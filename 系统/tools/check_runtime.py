"""使用标准库检查便携运行时、精确版本及关键导入。"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TARGETS = {
    'app': 'python_runtime',
    'vit': 'packaging_v0.4/resources/portable_python_v2',
    'engine': 'engine_runtime',
}
IMPORTS = {
    'app': ['numpy', 'PIL', 'cv2', 'onnxruntime', 'insightface', 'streamlit', 'pytest', 'reportlab', 'rapidocr_onnxruntime'],
    'vit': ['numpy', 'PIL', 'onnxruntime'],
    'engine': ['numpy', 'PIL', 'torch', 'torchvision', 'timm', 'transformers'],
}


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_lock(path):
    expected = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)(?:\s+--hash=sha256:[0-9a-f]{64})+', line)
        if not match:
            raise ValueError(f'锁文件格式错误：{line}')
        name = canonical(match[1])
        if name in expected:
            raise ValueError(f'锁文件存在重复包：{name}')
        expected[name] = match[2]
    if not expected:
        raise ValueError('锁文件为空')
    return expected


def probe(component, lock, backend, require_cuda=False):
    errors = []
    expected = read_lock(lock)
    actual = {}
    for distribution in importlib.metadata.distributions():
        name = canonical(distribution.metadata['Name'])
        if name in actual:
            errors.append(f'重复安装：{name}')
        actual[name] = distribution.version
    if sys.version_info[:3] != (3, 12, 10):
        errors.append(f'Python 版本错误：{sys.version.split()[0]}')
    for name, version in expected.items():
        if actual.get(name) != version:
            errors.append(f'{name}：期望 {version}，实际 {actual.get(name, "未安装")}')
    for name in actual.keys() - expected.keys():
        errors.append(f'锁外依赖：{name}=={actual[name]}')
    prefix = Path(sys.executable).parent
    for relative in ['python.exe', 'python312.dll', 'DLLs', 'Lib']:
        if not (prefix / relative).exists():
            errors.append(f'缺少便携目录内容：{relative}')
    if (prefix / 'pyvenv.cfg').exists() or sys.prefix != sys.base_prefix:
        errors.append('检测到 venv，不符合便携目录要求')
    if Path(sys.prefix).resolve() != prefix.resolve():
        errors.append(f'Python 前缀不在便携目录：{sys.prefix}')
    imported = []
    for module in IMPORTS[component]:
        try:
            importlib.import_module(module)
            imported.append(module)
        except Exception as error:
            errors.append(f'导入 {module} 失败：{type(error).__name__}: {error}')
    result = {'组件': component, 'Python': sys.version.split()[0], '依赖数量': len(actual), '导入通过': imported}
    if component == 'engine' and 'torch' in imported:
        torch = sys.modules['torch']
        available = torch.cuda.is_available()
        result['CUDA可用'] = available
        result['CUDA构建版本'] = torch.version.cuda
        result['显卡'] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if available else []
        expected_cuda = '12.8' if backend == 'cu128' else None
        if torch.version.cuda != expected_cuda:
            errors.append(f'CUDA 构建版本不符：期望 {expected_cuda}，实际 {torch.version.cuda}')
        if require_cuda and not available:
            errors.append('要求 CUDA 可用，但 torch.cuda.is_available() 为 False')
        if available:
            try:
                result['CUDA计算结果'] = torch.ones(4, device='cuda').sum().item()
            except Exception as error:
                errors.append(f'CUDA 实际计算失败：{error}')
    result['错误'] = errors
    result['通过'] = not errors
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return not errors


def check_component(root, component, backend, requirements=None, target=None, require_cuda=False):
    root = Path(root).resolve()
    directory = Path(target) if target else root / TARGETS[component]
    requirements = Path(requirements) if requirements else Path(__file__).resolve().parent.parent / 'requirements'
    lock = requirements / (f'engine-{backend}.lock.txt' if component == 'engine' else f'{component}.lock.txt')
    executable = directory / 'python.exe'
    if not executable.is_file():
        print(f'[失败] {component} 缺少解释器：{executable}', flush=True)
        return False
    flags = ['-E', '-S', '-B'] if component == 'vit' else ['-I', '-B']
    command = [str(executable), *flags, str(Path(__file__).resolve()), '--probe', component, '--lock', str(lock), '--engine-backend', backend]
    if require_cuda:
        command.append('--require-cuda')
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['HF_HUB_OFFLINE'] = '1'
    environment['TRANSFORMERS_OFFLINE'] = '1'
    try:
        completed = subprocess.run(command, env=environment, cwd=directory, timeout=240)
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f'[失败] {component} 校验无法完成：{error}', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='检查三个便携 Python 环境及其锁文件。')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent, help='系统目录')
    parser.add_argument('--components', default='app,vit,engine', help='以逗号分隔的组件')
    parser.add_argument('--engine-backend', choices=['cu128', 'cpu'], help='省略时读取安装记录')
    parser.add_argument('--require-cuda', action='store_true', help='要求实际 CUDA 运算成功')
    parser.add_argument('--probe', choices=list(TARGETS), help=argparse.SUPPRESS)
    parser.add_argument('--lock', type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    backend = arguments.engine_backend
    if not backend:
        receipt = arguments.root / TARGETS['engine'] / 'runtime-build.json'
        backend = json.loads(receipt.read_text(encoding='utf-8-sig'))['backend'] if receipt.exists() else 'cu128'
    if arguments.probe:
        return 0 if probe(arguments.probe, arguments.lock, backend, arguments.require_cuda) else 1
    components = [component.strip() for component in arguments.components.split(',')]
    if not components or any(component not in TARGETS for component in components):
        parser.error('组件只能是 app、vit、engine')
    results = [check_component(arguments.root, component, backend, require_cuda=arguments.require_cuda and component == 'engine') for component in components]
    print('[通过] 全部运行时检查通过' if all(results) else '[失败] 存在未通过的运行时', flush=True)
    return 0 if all(results) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[失败] 检查异常：{type(error).__name__}: {error}', flush=True)
        sys.exit(1)
