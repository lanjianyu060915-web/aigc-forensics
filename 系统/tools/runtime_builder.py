"""固定来源下载、同级临时构建和原子替换；仅使用标准库。"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import msvcrt
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.parse
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_runtime import TARGETS, check_component, read_lock

TOOLS = Path(__file__).resolve().parent
REQUIREMENTS = TOOLS.parent / 'requirements'


def message(text):
    print(text, flush=True)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8-sig', newline='\r\n')


def ensure_inside(path, parent):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(parent).resolve()) or resolved == Path(parent).resolve():
        raise ValueError(f'拒绝操作越界路径：{path}')
    return resolved


def remove_tree(path, parent):
    if Path(path).exists():
        ensure_inside(path, parent)
        if Path(path).is_symlink() or Path(path).is_junction():
            raise ValueError(f'拒绝删除链接目录：{path}')
        shutil.rmtree(path)


@contextmanager
def exclusive(path):
    with Path(path).open('a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(f'另一个安装进程正在使用目录，请勿同时运行：{path}') from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def download_segments(url, opener, temporary):
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'})
    with opener.open(request, timeout=45) as response:
        header = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes 0-0/(\d+)', header)
        if response.status != 206 or not match:
            return False
        total = int(match[1])
    chunk_size = (1 if temporary.name.startswith('ffmpeg-') else 16) * 1024 * 1024
    pieces = temporary.with_name(temporary.name + '.chunks')
    pieces.mkdir(exist_ok=True)
    for old_piece in list(pieces.iterdir()):
        if not old_piece.stem.isdigit():
            continue
        offset = int(old_piece.stem)
        if offset % chunk_size or old_piece.stat().st_size <= chunk_size:
            continue
        data = old_piece.read_bytes()
        for relative in range(0, len(data), chunk_size):
            block = data[relative:relative + chunk_size]
            if len(block) == chunk_size or offset + relative + len(block) == total:
                (pieces / str(offset + relative)).write_bytes(block)
        if old_piece.suffix == '.tmp':
            old_piece.unlink()
    ranges = [(offset, min(offset + chunk_size, total) - 1) for offset in range(0, total, chunk_size)]
    if temporary.exists():
        with temporary.open('rb') as existing:
            for offset, final in ranges:
                if final >= temporary.stat().st_size:
                    break
                piece = pieces / str(offset)
                block = existing.read(final - offset + 1)
                if not piece.exists():
                    piece.write_bytes(block)
    message(f'[分段下载] {temporary.name}，总计 {total} 字节，12 个并发连接')

    def fetch_piece(bounds):
        offset, final = bounds
        piece = pieces / str(offset)
        expected_size = final - offset + 1
        if piece.exists() and piece.stat().st_size == expected_size:
            return
        staging = piece.with_suffix('.tmp')
        for retry in range(3):
            try:
                existing_size = staging.stat().st_size if staging.exists() else 0
                if existing_size > expected_size:
                    staging.unlink()
                    existing_size = 0
                if existing_size == expected_size:
                    staging.replace(piece)
                    return
                request_start = offset + existing_size
                request = urllib.request.Request(url, headers={'Range': f'bytes={request_start}-{final}', 'Accept-Encoding': 'identity'})
                with opener.open(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {request_start}-{final}/{total}':
                        raise ValueError('服务器未返回要求的分段范围')
                    with staging.open('ab') as output:
                        shutil.copyfileobj(response, output, 1024 * 1024)
                if staging.stat().st_size != expected_size:
                    raise ValueError('下载分段被截断')
                staging.replace(piece)
                return
            except Exception:
                if retry == 2:
                    raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch_piece, bounds) for bounds in ranges]
        for finished, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if finished % 8 == 0 or finished == len(ranges):
                message(f'[分段进度] {temporary.name}：{finished}/{len(ranges)}')
    with temporary.open('wb') as output:
        for offset, final in ranges:
            with (pieces / str(offset)).open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    remove_tree(pieces, temporary.parent)
    return True


def download(entry, cache):
    filename = entry['filename']
    if Path(filename).name != filename or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
        raise ValueError('下载清单的文件名或 SHA-256 不合法')
    if not entry['url'].startswith('https://'):
        raise ValueError('只允许 HTTPS 下载来源')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / filename
    with exclusive(cache / (filename + '.lock')):
        if target.exists() and digest(target) == entry['sha256']:
            message(f'[缓存] 哈希通过：{filename}')
            return target
        temporary = cache / (filename + '.part')
        last_error = None
        locations = []
        if entry['url'].startswith('https://files.pythonhosted.org/'):
            locations.append((entry['url'].replace('https://files.pythonhosted.org', 'https://pypi.tuna.tsinghua.edu.cn'), True))
        elif urllib.parse.urlparse(entry['url']).hostname in {'download.pytorch.org', 'download-r2.pytorch.org'}:
            backend = 'cu128' if '+cu128' in filename else 'cpu'
            locations.append(('https://mirrors.aliyun.com/pytorch-wheels/' + backend + '/' + urllib.parse.quote(filename), True))
        locations.append((entry['url'], False))
        for attempt in range(1, 5):
            try:
                url, direct = locations[0 if attempt <= 2 else -1]
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
                large = entry.get('size', 0) > 64 * 1024 * 1024 or filename.startswith(('torch-', 'ffmpeg-'))
                if large and download_segments(url, opener, temporary):
                    if digest(temporary) != entry['sha256']:
                        temporary.unlink()
                        raise ValueError('分段合并后的 SHA-256 不匹配')
                    temporary.replace(target)
                    message(f'[下载完成] {filename}，{target.stat().st_size} 字节，SHA-256 通过')
                    return target
                offset = temporary.stat().st_size if temporary.exists() else 0
                headers = {'User-Agent': 'AIGC-Runtime-Setup/1', 'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                message(f'[下载 {attempt}/4] {filename}，续传位置 {offset} 字节')
                message(f'[来源] {url}；直连：{direct}')
                request = urllib.request.Request(url, headers=headers)
                with opener.open(request, timeout=45) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('下载服务器返回错误的续传范围')
                    total = int(response.headers.get('Content-Length', '0'))
                    received = 0
                    with temporary.open('ab' if append else 'wb') as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            received += len(block)
                    if total and total != received:
                        raise ValueError(f'下载不完整：本次应接收 {total} 字节，实际 {received}')
                if digest(temporary) != entry['sha256']:
                    temporary.unlink()
                    raise ValueError('SHA-256 不匹配，已丢弃下载内容')
                temporary.replace(target)
                message(f'[下载完成] {filename}，{target.stat().st_size} 字节，SHA-256 通过')
                return target
            except Exception as error:
                last_error = error
                message(f'[下载重试] {filename}：{error}')
                time.sleep(min(attempt * 2, 6))
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'下载失败：{filename}；请检查代理 HTTPS_PROXY/HTTP_PROXY。原因：{last_error}')


def zip_members(archive, target, final_target=None, prefix=''):
    members = []
    seen = set()
    total = 0
    for info in archive.infolist():
        if prefix and not info.filename.startswith(prefix):
            continue
        name = info.filename[len(prefix):].replace('\\', '/')
        if not name:
            continue
        pure = PurePosixPath(name)
        parts = pure.parts
        if pure.is_absolute() or '..' in parts or ':' in name or not parts:
            raise ValueError(f'压缩包包含危险路径：{info.filename}')
        for part in parts:
            if part.endswith((' ', '.')) or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part, re.I):
                raise ValueError(f'压缩包包含 Windows 非法路径：{name}')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'压缩包包含符号链接：{name}')
        path = target.joinpath(*parts)
        ensure_inside(path, target)
        for base in [target, final_target or target]:
            if len(str(base.joinpath(*parts))) >= 248:
                raise ValueError(f'解压路径过长，请将系统目录移到更短的位置：{base.joinpath(*parts)}')
        normalized = str(pure).rstrip('/').casefold()
        if normalized in seen:
            raise ValueError(f'压缩包包含重复路径：{name}')
        seen.add(normalized)
        total += info.file_size
        members.append((info, path))
    if not members:
        raise ValueError('压缩包没有可用内容')
    if total > shutil.disk_usage(target.parent).free:
        raise ValueError(f'磁盘空间不足，解压至少需要 {total} 字节')
    return members


def extract(archive_path, target, final_target=None, prefix=''):
    with zipfile.ZipFile(archive_path) as archive:
        members = zip_members(archive, target, final_target, prefix)
        for info, path in members:
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def check_receipt(target, identity):
    try:
        receipt = json.loads((target / 'runtime-build.json').read_text(encoding='utf-8-sig'))
        return receipt['identity'] == identity
    except (OSError, ValueError, KeyError):
        return False


def rename_retry(source, destination):
    for attempt in range(8):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            message(f'[等待] Windows 暂时占用目录，重试重命名：{source.name}')
            time.sleep(0.25 * (attempt + 1))


def swap(stage, target):
    backup = target.with_name(target.name + '.old-' + uuid.uuid4().hex[:8])
    had_target = target.exists()
    if had_target:
        rename_retry(target, backup)
    try:
        rename_retry(stage, target)
    except BaseException:
        if had_target:
            rename_retry(backup, target)
        raise
    if had_target:
        remove_tree(backup, target.parent)


def install_python(root, component, backend, manifest, cache, force):
    started = time.perf_counter()
    target = root / TARGETS[component]
    ensure_inside(target, root)
    key = f'engine-{backend}' if component == 'engine' else component
    lock = REQUIREMENTS / f'{key}.lock.txt'
    expected = read_lock(lock)
    entries = manifest['components'][key]
    from check_runtime import canonical
    if {canonical(entry['name']): entry['version'] for entry in entries} != expected:
        raise ValueError(f'{key} 下载清单与锁文件不一致')
    identity = {'lock_sha256': digest(lock), 'python_sha256': manifest['python']['sha256'], 'layout': 1}
    if target.exists() and not force:
        if check_receipt(target, identity) and check_component(root, component, backend, REQUIREMENTS):
            message(f'[跳过] {component} 已存在且校验通过，耗时 {time.perf_counter()-started:.2f} 秒')
            return
        message(f'[重建] {component} 原目录校验未通过，将保留旧目录直到新环境验收完成')
    message(f'[构建] {component}，后端 {backend if component == "engine" else "无"}')
    base = download(manifest['python'], cache)
    pip_wheel = download(manifest['pip'], cache)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda entry: download(entry, cache), entries))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.build-', dir=target.parent))
    try:
        extract(base, stage, target, prefix='tools/')
        site = stage / 'Lib' / 'site-packages'
        if site.exists():
            remove_tree(site, stage)
        site.mkdir(parents=True)
        for relative in ['Scripts', 'include', 'libs', 'Lib/ensurepip']:
            remove_tree(stage / relative, stage)
        work = stage / '.install-work'
        work.mkdir()
        wheelhouse = work / 'wheelhouse'
        wheelhouse.mkdir()
        for entry in entries:
            source = cache / entry['filename']
            destination = wheelhouse / entry['filename']
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        packages = work / 'packages'
        packages.mkdir()
        environment = os.environ.copy()
        environment.update({'TEMP': str(work), 'TMP': str(work), 'PYTHONDONTWRITEBYTECODE': '1', 'PIP_CONFIG_FILE': os.devnull, 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
        code = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("pip",run_name="__main__")'
        command = [str(stage / 'python.exe'), '-I', '-S', '-B', '-c', code, str(pip_wheel), '--isolated', 'install', '--no-deps', '--no-index', '--find-links', str(wheelhouse), '--only-binary=:all:', '--require-hashes', '--no-cache-dir', '--no-compile', '--no-warn-script-location', '--target', str(packages), '-r', str(lock)]
        message(f'[安装] {component}：按哈希锁安装 {len(entries)} 个包，禁用依赖解析')
        subprocess.run(command, env=environment, check=True)
        destination = stage / 'Lib' if component == 'vit' else site
        for item in packages.iterdir():
            if item.name in {'bin', 'Scripts'}:
                continue
            if (destination / item.name).exists():
                raise ValueError(f'依赖覆盖标准库或重复文件：{item.name}')
            item.rename(destination / item.name)
        remove_tree(work, stage)
        save_json(stage / 'runtime-build.json', {'component': component, 'backend': backend if component == 'engine' else None, 'identity': identity, 'python': '3.12.10', 'packages': len(entries), 'pip': '仅临时从哈希锁定的 wheel 运行，未安装到最终环境'})
        if not check_component(root, component, backend, REQUIREMENTS, stage):
            raise RuntimeError(f'{component} 新环境验收失败，未替换旧目录')
        swap(stage, target)
        message(f'[完成] {component}，耗时 {time.perf_counter()-started:.2f} 秒')
    finally:
        remove_tree(stage, target.parent)


def ffmpeg_valid(target, entry):
    if not check_receipt(target, {'archive_sha256': entry['sha256']}):
        return False
    try:
        receipt = json.loads((target / 'runtime-build.json').read_text(encoding='utf-8-sig'))
        for name in ['ffmpeg.exe', 'ffprobe.exe']:
            if digest(target / name) != receipt['files'][name]:
                return False
            result = subprocess.run([str(target / name), '-version'], capture_output=True, text=True, timeout=30)
            if result.returncode or not result.stdout.startswith(name[:-4] + ' version ' + entry['version'] + '-'):
                return False
        return True
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return False


def install_ffmpeg(root, manifest, cache, force):
    target = root / 'ffmpeg'
    ensure_inside(target, root)
    entry = manifest['ffmpeg']
    if target.exists() and not force and ffmpeg_valid(target, entry):
        message('[跳过] FFmpeg、FFprobe 版本及文件哈希通过')
        return
    archive_path = download(entry, cache)
    stage = Path(tempfile.mkdtemp(prefix='.ffmpeg.build-', dir=root))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            zip_members(archive, stage, target)
            for filename in ['ffmpeg.exe', 'ffprobe.exe']:
                matches = [name for name in archive.namelist() if name.endswith('/bin/' + filename)]
                if len(matches) != 1:
                    raise ValueError(f'FFmpeg 压缩包中找不到唯一的 {filename}')
                with archive.open(matches[0]) as source, (stage / filename).open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
            for filename in ['LICENSE', 'README.txt']:
                matches = [name for name in archive.namelist() if name.endswith('/' + filename)]
                if matches:
                    (stage / filename).write_bytes(archive.read(matches[0]))
        save_json(stage / 'build-source.json', entry)
        save_json(stage / 'runtime-build.json', {'identity': {'archive_sha256': entry['sha256']}, 'files': {name: digest(stage / name) for name in ['ffmpeg.exe', 'ffprobe.exe']}})
        if not ffmpeg_valid(stage, entry):
            raise RuntimeError('FFmpeg/FFprobe 运行或版本校验失败')
        swap(stage, target)
        message('[完成] FFmpeg、FFprobe 固定版本安装并校验通过')
    finally:
        remove_tree(stage, root)


def tree_hashes(target):
    return {str(path.relative_to(target)).replace('\\', '/'): digest(path) for path in target.rglob('*') if path.is_file() and path.name != 'runtime-build.json'}


def install_frozen(root, force):
    for name in ['deepfake_runtime', 'portable_python']:
        archive = root / 'assets' / (name + '.zip')
        target = root / 'packaging_v0.3' / 'resources' / name
        ensure_inside(target, root)
        if not archive.is_file():
            message(f'[提示] 冻结人脸裁剪分支不可用：缺少 assets/{name}.zip（不是失败）')
            continue
        archive_hash = digest(archive)
        sidecar = archive.with_suffix('.zip.sha256')
        if sidecar.exists():
            expected = sidecar.read_text(encoding='utf-8-sig').strip().split()[0].lower()
            if archive_hash != expected:
                raise ValueError(f'冻结压缩包哈希不一致：{archive.name}')
        else:
            message(f'[提示] {archive.name} 未提供独立哈希，将记录本次 SHA-256；不声称已验证来源真实性')
        identity = {'archive_sha256': archive_hash}
        if target.exists() and not force and check_receipt(target, identity):
            receipt = json.loads((target / 'runtime-build.json').read_text(encoding='utf-8-sig'))
            if receipt.get('files') == tree_hashes(target):
                message(f'[跳过] 冻结运行时 {name} 文件哈希通过')
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.' + name + '.build-', dir=target.parent))
        try:
            with zipfile.ZipFile(archive) as package:
                files = [info.filename for info in package.infolist() if not info.is_dir()]
                prefix = name + '/' if files and all(path.startswith(name + '/') for path in files) else ''
            extract(archive, stage, target, prefix)
            hashes = tree_hashes(stage)
            if not hashes:
                raise ValueError(f'冻结压缩包没有文件：{archive}')
            save_json(stage / 'runtime-build.json', {'identity': identity, 'files': hashes})
            swap(stage, target)
            message(f'[完成] 冻结运行时 {name} 安全解压，SHA-256：{archive_hash}')
        finally:
            remove_tree(stage, target.parent)


def detect_backend():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and result.stdout.strip():
            message('[显卡] ' + result.stdout.strip())
            return 'cu128'
    except (OSError, subprocess.TimeoutExpired):
        pass
    message('[显卡] 未检测到可查询的 NVIDIA 显卡，选择 CPU')
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='建立可搬移的固定版本运行时')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--engine-backend', choices=['cu128', 'cpu'])
    parser.add_argument('--components', default='app,vit,engine,ffmpeg,frozen')
    parser.add_argument('--force', action='store_true')
    arguments = parser.parse_args()
    components = list(dict.fromkeys(part.strip() for part in arguments.components.split(',')))
    if not components or any(part not in [*TARGETS, 'ffmpeg', 'frozen'] for part in components):
        parser.error('组件只允许 app、vit、engine、ffmpeg、frozen')
    root = arguments.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt' or sys.version_info < (3, 12):
        raise RuntimeError('下载助手需要 Windows 上的 Python 3.12 或更高版本；最终环境固定为 3.12.10')
    manifest = json.loads((REQUIREMENTS / 'runtime-sources.json').read_text(encoding='utf-8-sig'))
    backend = arguments.engine_backend or detect_backend()
    cache = Path(os.environ.get('AIGC_RUNTIME_CACHE', str(root / '.runtime-cache'))).resolve()
    started = time.perf_counter()
    message(f'[开始] 系统目录：{root}；后端：{backend}；缓存：{cache}')
    with exclusive(root / '.runtime-setup.lock'):
        for component in components:
            if component in TARGETS:
                install_python(root, component, backend, manifest, cache, arguments.force)
            elif component == 'ffmpeg':
                install_ffmpeg(root, manifest, cache, arguments.force)
            else:
                install_frozen(root, arguments.force)
    message(f'[成功] 所选组件处理完毕，总耗时 {time.perf_counter()-started:.2f} 秒')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        message('[失败] 用户中断，未完成的构建目录已清理')
        sys.exit(130)
    except Exception as error:
        message(f'[失败] {type(error).__name__}：{error}')
        sys.exit(1)
