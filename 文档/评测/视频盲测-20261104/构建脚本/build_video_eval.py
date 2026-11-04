import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import random
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261104
MAX_FILE = 100_000_000
MAX_TOTAL = 11_500_000_000
FFPROBE = Path('ffprobe')
FFMPEG = Path('ffmpeg')
LOCAL_MEDIA = ROOT / '已用媒体'  # 选模时用过的媒体，只读取哈希用于排除
PANELS = ROOT / '已用清单'  # 选模时用过的清单，只读取 sha256 用于排除
MEDIA_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.m4s', '.mts', '.m2ts', '.flv', '.gif', '.jpg', '.jpeg', '.png', '.webp', '.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
FIELDS = ['sample_id', 'label', 'source', 'generator', 'license', 'origin', 'label_evidence', 'sha256', 'path', 'width', 'height', 'duration_seconds', 'codec', 'container', 'split', 'transform', 'group_id', 'notes']
REPOS = {'rapidata': 'Rapidata/text-2-video-human-preferences-seedance-1-pro', 'veo': 'Rapidata/text-2-video-human-preferences-veo3.1', 'msvd': 'OmerXYZ/comgenvid', '4d': '4DCodeBench/Dataset-Real-World', 'yo': 'YouZhe/YoCausal-dataset'}
MODEL_MAP = {'alpha': 'Runway', 'hunyuan': 'Hunyuan', 'pika2.2': 'Pika', 'ray2': 'Luma', 'sora': 'Sora', 'wan2.1': 'Wan', 'seedance-1-pro-24-7-25': 'Seedance', 'veo31': 'Veo'}
FOUR_D_QUOTAS = {'WISA-80K': 28, 'physics-IQ-benchmark': 12, 'ABC-130K': 5, 'T-REX': 4, 'Phys101': 3, 'Phys-AD': 3, 'Robo360': 3, 'AgiBotWorld': 2}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stable(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def total_bytes():
    return sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())


def request(url, stream=False):
    for attempt in range(3):
        try:
            response = requests.get(url, stream=stream, timeout=(15, 20), headers={'User-Agent': 'VideoBlindResearch/1.0 (public anonymous research)'})
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status in (401, 403, 404, 429) or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def evidence_dir(repo):
    return ROOT / 'evidence' / repo.replace('/', '__')


def info_for(repo):
    return json.loads((evidence_dir(repo) / 'api.json').read_text(encoding='utf-8'))


def url_for(repo, name):
    return f"https://huggingface.co/datasets/{repo}/resolve/{info_for(repo)['sha']}/{quote(name, safe='/')}"


def ensure_evidence():
    for repo in REPOS.values():
        folder = evidence_dir(repo)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'api.json').exists():
            response = request('https://huggingface.co/api/datasets/' + repo)
            (folder / 'api.json').write_bytes(response.content)
        info = info_for(repo)
        if not (folder / 'api_blobs.json').exists():
            response = request('https://huggingface.co/api/datasets/' + repo + '/revision/' + info['sha'] + '?blobs=true')
            (folder / 'api_blobs.json').write_bytes(response.content)
        if info.get('gated') or info.get('private'):
            raise RuntimeError('Only ungated public repositories are permitted: ' + repo)
        names = ['README.md']
        if repo in [REPOS['rapidata'], REPOS['veo']]:
            names += ['data/train-00000-of-00001.parquet']
        if repo in [REPOS['msvd'], REPOS['4d']]:
            names += ['metadata.jsonl']
        if repo == REPOS['yo']:
            names += ['subset/animal/dataset_metadata.json', 'subset/general/dataset_metadata.json']
        for name in names:
            local_name = name.split('/')[1] + '.json' if name.startswith('subset/') else Path(name).name
            target = folder / local_name
            if not target.exists():
                response = request(url_for(repo, name))
                if len(response.content) > MAX_FILE:
                    raise RuntimeError('Oversized evidence')
                target.write_bytes(response.content)
    records = []
    for repo in REPOS.values():
        for path in sorted(evidence_dir(repo).iterdir()):
            if path.is_file():
                records.append({'repo': repo, 'revision': info_for(repo)['sha'], 'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'bytes': path.stat().st_size})
    save_json(ROOT / 'evidence/sources_lock.json', records)


def build_exclusions():
    if not PANELS.is_dir() or not LOCAL_MEDIA.is_dir():
        raise RuntimeError('Exclusion roots unavailable; cannot certify novelty')
    panel_files = sorted(PANELS.glob('*.jsonl'))
    if not panel_files:
        raise RuntimeError('No panel exclusion files')
    records = []
    for path in panel_files:
        for number, line in enumerate(path.open(encoding='utf-8-sig'), 1):
            for value in re.findall(r'"sha256"\s*:\s*"([0-9a-fA-F]{64})"', line):
                records.append({'kind': 'panel', 'path': str(path), 'line': number, 'sha256': value.lower()})
    files = sorted(path for path in LOCAL_MEDIA.rglob('*') if path.is_file() and path.suffix.lower() in MEDIA_EXTS)
    if not files:
        raise RuntimeError('No local media found; check path encoding and access')
    cache_path = ROOT / 'evidence/local_media_hashes.jsonl'
    cache = {row['path']: row for row in load_jsonl(cache_path)} if cache_path.exists() else {}
    def compute(path):
        stat = path.stat()
        cached = cache.get(str(path))
        if cached and cached['size'] == stat.st_size and cached['mtime_ns'] == stat.st_mtime_ns:
            return cached
        checksum = digest(path)
        after = path.stat()
        if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
            raise RuntimeError('Local media changed while hashing: ' + str(path))
        return {'kind': 'local_media', 'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': checksum}
    print('Hashing local media', len(files), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        local = list(pool.map(compute, files))
    save_jsonl(cache_path, local)
    records.extend(local)
    archive_path = ROOT / 'evidence/archive_media_hashes.jsonl'
    if archive_path.exists():
        records.extend(load_jsonl(archive_path))
    save_jsonl(ROOT / 'evidence/exclusions.jsonl', records)
    summary = {'time_utc': now(), 'panel_files': len(panel_files), 'panel_hash_records': sum(row['kind'] == 'panel' for row in records), 'archive_media_hash_records': sum(row['kind'] == 'archived_media' for row in records), 'local_media_files': len(local), 'local_media_bytes': sum(row['size'] for row in local), 'unique_hashes': len({row['sha256'] for row in records}), 'scope': 'All explicit sha256 values in designated panels, plus direct media files recursively under local data; archives handled separately by archive audit.'}
    save_json(ROOT / 'evidence/exclusion_summary.json', summary)
    print(summary, flush=True)
    return {row['sha256'] for row in records}


def probe(path):
    result = subprocess.run([str(FFPROBE), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True, text=True, encoding='utf-8', timeout=60, check=True)
    payload = json.loads(result.stdout)
    video = next(stream for stream in payload['streams'] if stream['codec_type'] == 'video')
    duration = float(payload['format'].get('duration', video.get('duration', 0)))
    numerator, denominator = map(float, video.get('avg_frame_rate', '0/1').split('/'))
    size = path.stat().st_size
    return {'width': int(video['width']), 'height': int(video['height']), 'duration_seconds': duration, 'codec': video['codec_name'], 'container': payload['format']['format_name'], 'fps': numerator / denominator if denominator else 0, 'bit_rate': int(payload['format'].get('bit_rate', size * 8 / duration if duration else 0)), 'file_size': size, 'audio_streams': sum(stream['codec_type'] == 'audio' for stream in payload['streams'])}


def candidate(repo, name, label, source, generator, group, evidence, notes, bucket):
    identity = hashlib.sha256((repo + '/' + name).encode()).hexdigest()[:20]
    source_slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', source)
    return {'sample_id': 'v_' + identity, 'label': label, 'source': source, 'generator': generator, 'license': '未核实', 'origin': url_for(repo, name), 'label_evidence': evidence, 'sha256': '', 'path': f'data/{source_slug}/{identity}{Path(name).suffix.lower()}', 'split': '', 'transform': 'original', 'group_id': group, 'notes': notes, '_repo': repo, '_remote_path': name, '_bucket': bucket}


def make_candidates():
    import pyarrow.parquet as parquet
    candidates = []
    for key in ['rapidata', 'veo']:
        repo = REPOS[key]
        files = [item['rfilename'] for item in info_for(repo)['siblings']]
        by_name = {Path(name).name: name for name in files if name.endswith('.mp4')}
        table = parquet.read_table(evidence_dir(repo) / 'train-00000-of-00001.parquet', columns=['prompt', 'file_name1', 'file_name2', 'model1', 'model2'])
        seen = set()
        for index, row in enumerate(table.to_pylist()):
            for side in [1, 2]:
                model = row['model' + str(side)]
                filename = row['file_name' + str(side)]
                if model not in MODEL_MAP or filename not in by_name or filename in seen:
                    continue
                if key == 'veo' and model != 'veo31':
                    continue
                if key == 'rapidata' and model == 'veo31':
                    continue
                seen.add(filename)
                prompt = ' '.join(row['prompt'].split())
                if re.search(r'\b(dragon|magical|fantasy|cartoon|anime|surreal|wizard|unicorn|alien|animated|cyberpunk)\b', prompt, re.I):
                    continue
                group = 'rapidata_prompt_' + hashlib.sha256(prompt.lower().encode()).hexdigest()[:20]
                evidence = f"{url_for(repo, 'data/train-00000-of-00001.parquet')} ; row={index}, file_name{side}={filename}, model{side}={model}; dataset card explicitly describes generated model-comparison videos."
                record = candidate(repo, by_name[filename], 1, 'Rapidata_' + MODEL_MAP[model], MODEL_MAP[model], group, evidence, {'model_version': model, 'prompt': prompt, 'dataset_card': url_for(repo, 'README.md'), 'license_scope': 'Dataset-card declaration; per-video creator rights not independently cleared.', 'source_processing': 'Unmodified bytes of released MP4; upstream processing unknown.'}, 'gen/' + MODEL_MAP[model])
                card = (evidence_dir(repo) / 'README.md').read_text(encoding='utf-8')
                match = re.search(r'^license:\s*(\S+)', card, re.M)
                record['license'] = (match.group(1) + ' (dataset card; underlying video rights not independently verified)') if match else '未核实'
                candidates.append(record)
    repo = REPOS['msvd']
    for row in load_jsonl(evidence_dir(repo) / 'metadata.jsonl'):
        if row.get('subset') != 'real' or row.get('source_model') != 'MSVD':
            continue
        if min(row['width'], row['height']) < 480 or not 5 <= row['duration_seconds'] <= 30:
            continue
        filename = row['file_name']
        if remote_file_info(repo, filename).get('size', MAX_FILE + 1) > 10_000_000:
            continue
        youtube_id = Path(filename).stem.rsplit('_', 2)[0]
        record = candidate(repo, filename, 0, 'ComGenVid_MSVD', 'real', 'youtube_' + youtube_id, f"{url_for(repo, 'metadata.jsonl')} ; file_name={filename}, subset=real, source_model=MSVD", {'dataset_card': url_for(repo, 'README.md'), 'source_processing': 'Dataset-released MSVD clip, not downloaded from YouTube; upstream clipping/transcoding possible.', 'upstream_id': youtube_id, 'sampling': 'Revision 2: shorter side >=480, duration 5-30 seconds, released size <=10 MB. Five previously accepted HD clips retained. Adjusted before baseline for bandwidth and size/resolution confounding.'}, 'real/MSVD')
        candidates.append(record)
    repo = REPOS['4d']
    for row in load_jsonl(evidence_dir(repo) / 'metadata.jsonl'):
        if row['availability'] != 'included' or row['dataset'] not in FOUR_D_QUOTAS:
            continue
        group = '4d_' + hashlib.sha256((row['dataset'] + ':' + row['source']).encode()).hexdigest()[:20]
        record = candidate(repo, 'videos/' + row['case'] + '.mp4', 0, '4DCodeBench_' + row['dataset'], 'real', group, f"{url_for(repo, 'README.md')} states real-world physical events; {url_for(repo, 'metadata.jsonl')}, case={row['case']}, dataset={row['dataset']}, availability=included", {'upstream_source': row['source'], 'description': row['description'], 'categories': row['categories'], 'dataset_card': url_for(repo, 'README.md'), 'source_processing': 'Release already trims/downsamples, caps long side at 1280 and fps at 30; these released bytes are retained, not camera originals.'}, 'real/4d/' + row['dataset'])
        candidates.append(record)
    repo = REPOS['yo']
    for subset in ['animal', 'general']:
        for row in json.loads((evidence_dir(repo) / (subset + '.json')).read_text(encoding='utf-8')):
            filename = Path(row['video_path_forward']).name
            record = candidate(repo, f'subset/{subset}/fwd/{filename}', 0, 'YoCausal_' + ('AnimalKingdom' if subset == 'animal' else 'MomentsInTime'), 'real', 'yocausal_' + subset + '_' + Path(filename).stem, f"{url_for(repo, 'README.md')} explicitly states fwd/ contains natural videos; {url_for(repo, 'subset/' + subset + '/dataset_metadata.json')}, id={row['id']}, video_path_forward={row['video_path_forward']}", {'prompt': row['prompt'], 'upstream_dataset': row['dataset_source'], 'source_processing': 'Released forward clip only; upstream fixed 3-second clips, no reversed bwd videos.', 'dataset_card': url_for(repo, 'README.md'), 'license_scope': 'CC BY 4.0 declared by dataset card; upstream per-clip rights not independently cleared.'}, 'real/yo/' + subset)
            record['license'] = 'cc-by-4.0 (dataset card; upstream per-clip rights not independently verified)'
            candidates.append(record)
    quotas = {'gen/' + value: 15 for value in MODEL_MAP.values()}
    quotas.update({'real/MSVD': 20, 'real/yo/animal': 10, 'real/yo/general': 10})
    quotas.update({'real/4d/' + name: count for name, count in FOUR_D_QUOTAS.items()})
    unique = {row['sample_id']: row for row in candidates}
    candidates = sorted(unique.values(), key=lambda row: (row['_bucket'], stable(row['sample_id'])))
    save_jsonl(ROOT / 'evidence/candidates.jsonl', candidates)
    save_json(ROOT / 'evidence/sampling_plan.json', {'seed': SEED, 'quotas': quotas, 'available': dict(collections.Counter(row['_bucket'] for row in candidates)), 'rules': 'Deterministic seeded SHA ordering; one released video per prompt per generator; group all models of same prompt; no video detection or preference scores loaded; excludes explicit fantasy prompt words; real MSVD HD prefilter; no changes after metadata test AUC.'})
    print('CANDIDATES', len(candidates), flush=True)
    return candidates, quotas


def remote_file_info(repo, filename):
    folder = evidence_dir(repo)
    entries = json.loads((folder / 'api_blobs.json').read_text(encoding='utf-8'))
    if entries['sha'] != info_for(repo)['sha']:
        raise RuntimeError('Remote revision mismatch')
    return next(item for item in entries['siblings'] if item['rfilename'] == filename)


def download_one(record):
    path = ROOT / record['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    started = time.monotonic()
    try:
        if not path.exists():
            with request(record['origin'] + '?download=true', stream=True) as response:
                expected = int(response.headers.get('Content-Length', 0))
                if expected > MAX_FILE:
                    raise ValueError('File exceeds 100 MB')
                amount = 0
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(64 * 1024):
                        if time.monotonic() - started > 300:
                            raise TimeoutError('Per-file download wall clock budget exceeded')
                        amount += len(chunk)
                        if amount > MAX_FILE:
                            raise ValueError('Stream exceeds 100 MB')
                        stream.write(chunk)
                if expected and expected != amount:
                    raise ValueError(f'Content-Length mismatch: {expected} != {amount}')
                temporary.replace(path)
        metadata = probe(path)
        if not 0 < metadata['duration_seconds'] <= 60 or metadata['file_size'] > MAX_FILE:
            raise ValueError('Duration or size outside limits')
        result = dict(record)
        result['sha256'] = digest(path)
        remote = remote_file_info(record['_repo'], record['_remote_path'])
        if remote.get('lfs', {}).get('sha256') != result['sha256'] or remote['size'] != metadata['file_size']:
            raise ValueError('Pinned upstream LFS SHA256 or size mismatch')
        result.update({key: metadata[key] for key in ['width', 'height', 'duration_seconds', 'codec', 'container']})
        result['notes'] = dict(result['notes'], downloaded_utc=now(), metadata=metadata, transfer='Python requests, TLS verified, no credentials; upstream LFS SHA256 and size verified; Content-Length checked when supplied')
        return result, None
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if path.exists():
            path.unlink()
        return None, {'sample_id': record['sample_id'], 'url': record['origin'], 'source': record['source'], 'reason': str(error)[:1000], 'time_utc': now()}


def collect(candidates, quotas, excluded):
    selected_path = ROOT / 'evidence/selected_originals.jsonl'
    selected = load_jsonl(selected_path) if selected_path.exists() else []
    errors_path = ROOT / 'evidence/acquisition_failures.jsonl'
    failures = load_jsonl(errors_path) if errors_path.exists() else []
    accepted_ids = {row['sample_id'] for row in selected}
    rejected_ids = {row['sample_id'] for row in failures}
    hashes = {row['sha256'] for row in selected}
    counts = collections.Counter(row['_bucket'] for row in selected)
    generator_groups = {(row['_bucket'], row['group_id']) for row in selected}
    for bucket, quota in sorted(quotas.items()):
        pool = [row for row in candidates if row['_bucket'] == bucket and row['sample_id'] not in accepted_ids | rejected_ids]
        while counts[bucket] < quota and pool:
            if total_bytes() + MAX_FILE * 12 > MAX_TOTAL:
                raise RuntimeError('Workspace budget exhausted')
            batch = []
            planned_groups = set()
            while pool and len(batch) < min(12, quota - counts[bucket]):
                row = pool.pop(0)
                pair = (bucket, row['group_id'])
                if pair in generator_groups or pair in planned_groups:
                    continue
                batch.append(row)
                planned_groups.add(pair)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(download_one, batch))
            for row, error in results:
                if error:
                    failures.append(error)
                    print('SKIP', bucket, error['reason'][:120], flush=True)
                    continue
                if row['sha256'] in hashes or row['sha256'] in excluded:
                    (ROOT / row['path']).unlink()
                    failures.append({'sample_id': row['sample_id'], 'url': row['origin'], 'source': row['source'], 'reason': 'Excluded previously seen hash or within-set duplicate', 'time_utc': now()})
                    continue
                selected.append(row)
                hashes.add(row['sha256'])
                counts[bucket] += 1
                generator_groups.add((bucket, row['group_id']))
                print('OK', bucket, counts[bucket], '/', quota, row['notes']['metadata']['file_size'], flush=True)
            save_jsonl(selected_path, selected)
            save_jsonl(errors_path, failures)
        print('BUCKET', bucket, counts[bucket], '/', quota, flush=True)
    return selected


def assign_splits(rows):
    groups = sorted({row['group_id'] for row in rows})
    strata = sorted({row['_bucket'] for row in rows})
    total = collections.Counter(row['_bucket'] for row in rows)
    group_rows = collections.defaultdict(list)
    for row in rows:
        group_rows[row['group_id']].append(row)
    rng = random.Random(SEED)
    best = None
    for attempt in range(3000):
        chosen = {group for group in groups if rng.random() < 0.30}
        count = collections.Counter(row['_bucket'] for group in chosen for row in group_rows[group])
        if any(count[stratum] == total[stratum] for stratum in strata):
            continue
        score = sum((count[stratum] - total[stratum] * .3) ** 2 / total[stratum] for stratum in strata)
        score += (sum(count.values()) - len(rows) * .3) ** 2 / len(rows)
        if best is None or score < best[0]:
            best = (score, chosen)
    if best is None:
        raise RuntimeError('Unable to construct group-disjoint split')
    for row in rows:
        row['split'] = 'calibration' if row['group_id'] in best[1] else 'test'
    save_json(ROOT / 'evidence/split_assignment.json', {'seed': SEED, 'iterations': 3000, 'objective': best[0], 'calibration_groups': sorted(best[1]), 'policy': 'Joint prompt groups across generator families; no metadata or classifier predictions used in split search.'})
    return rows


def derive(rows):
    derived = []
    vf = "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
    for label in [0, 1]:
        eligible = sorted((row for row in rows if row['split'] == 'test' and row['label'] == label), key=lambda row: stable('degrade:' + row['sample_id']))[:60]
        for index, original in enumerate(eligible, 1):
            row = dict(original)
            row['sample_id'] += '_d28'
            row['path'] = 'derived/' + row['sample_id'] + '.mp4'
            target = ROOT / row['path']
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                if total_bytes() + MAX_FILE > MAX_TOTAL:
                    raise RuntimeError('No budget for derived clip')
                temporary = target.with_suffix('.part.mp4')
                command = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(ROOT / original['path']), '-map', '0:v:0', '-map', '0:a?', '-vf', vf, '-c:v', 'libx264', '-crf', '28', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-threads', '2', '-c:a', 'aac', '-b:a', '128k', '-map_metadata', '-1', '-movflags', '+faststart', '-fs', str(MAX_FILE), str(temporary)]
                subprocess.run(command, check=True, capture_output=True, timeout=600)
                temporary.replace(target)
            metadata = probe(target)
            if abs(metadata['duration_seconds'] - original['duration_seconds']) > .25 or metadata['file_size'] >= MAX_FILE:
                raise RuntimeError('Derived clip truncated or too large')
            row.update({key: metadata[key] for key in ['width', 'height', 'duration_seconds', 'codec', 'container']})
            row['sha256'] = digest(target)
            row['split'] = 'test_degraded'
            row['transform'] = 'h264_crf28_720p'
            row['notes'] = dict(original['notes'], parent_sample_id=original['sample_id'], parent_sha256=original['sha256'], metadata=metadata, transformation={'video_filter': vf, 'codec': 'libx264', 'crf': 28, 'preset': 'veryfast', 'pixel_format': 'yuv420p', 'audio': 'aac 128k if present', 'fps': 'preserved', 'canvas': '1280x720, aspect preserved; letterbox and upscale allowed'})
            derived.append(row)
            print('DERIVED', label, index, '/', len(eligible), flush=True)
    return derived


def write_manifest(rows):
    save_jsonl(ROOT / 'manifest.jsonl', [{key: row[key] for key in FIELDS} for row in rows])


def validate(rows, excluded):
    errors = []
    hashes = set()
    ids = set()
    groups = collections.defaultdict(set)
    originals = {row['sample_id']: row for row in rows if row['transform'] == 'original'}
    refreshed = []
    for row in rows:
        path = ROOT / row['path']
        try:
            if list(row) != FIELDS:
                raise ValueError('Manifest field schema/order differs')
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError('Path outside workspace')
            if not path.is_file() or digest(path) != row['sha256']:
                raise ValueError('Missing file or mismatched SHA256')
            if row['sha256'] in hashes or row['sha256'] in excluded or row['sample_id'] in ids:
                raise ValueError('Duplicate or excluded SHA256/sample_id')
            hashes.add(row['sha256'])
            ids.add(row['sample_id'])
            metadata = probe(path)
            for key in ['width', 'height', 'codec', 'container']:
                if metadata[key] != row[key]:
                    raise ValueError('Metadata mismatch: ' + key)
            if abs(metadata['duration_seconds'] - row['duration_seconds']) > 1e-5:
                raise ValueError('Duration mismatch')
            if not 0 < metadata['duration_seconds'] <= 60 or metadata['file_size'] > MAX_FILE:
                raise ValueError('Duration/size limit exceeded')
            groups[row['group_id']].add('test' if row['split'] == 'test_degraded' else row['split'])
            if row['transform'] != 'original':
                parent = originals[row['notes']['parent_sample_id']]
                if parent['split'] != 'test' or parent['group_id'] != row['group_id'] or parent['sha256'] != row['notes']['parent_sha256']:
                    raise ValueError('Invalid derived lineage')
                if row['split'] != 'test_degraded' or metadata['codec'] != 'h264' or (metadata['width'], metadata['height']) != (1280, 720):
                    raise ValueError('Invalid derivative format')
            refreshed.append({'sample_id': row['sample_id'], **metadata})
        except Exception as error:
            errors.append({'sample_id': row['sample_id'], 'error': str(error)})
    overlap = [group for group, splits in groups.items() if 'test' in splits and 'calibration' in splits]
    if overlap:
        errors.append({'group_leakage': overlap})
    source_test = {row['source'] for row in rows if row['split'] == 'test'}
    missing_test = {row['source'] for row in originals.values()} - source_test
    if missing_test:
        errors.append({'sources_without_test': sorted(missing_test)})
    size = total_bytes()
    if size > 12_000_000_000:
        errors.append({'workspace_bytes': size})
    result = {'time_utc': now(), 'passed': not errors, 'checked_rows': len(rows), 'unique_hashes': len(hashes), 'excluded_unique_hashes': len(excluded), 'group_overlap_count': len(overlap), 'workspace_bytes_at_validation': size, 'errors': errors}
    save_json(ROOT / 'validation.json', result)
    save_jsonl(ROOT / 'evidence/ffprobe_metadata.jsonl', refreshed)
    print('VALIDATION', json.dumps(result, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError('Validation failed')
    return result


def summary_numbers(values):
    import numpy as np
    array = np.array(values, dtype=float)
    return dict(zip(['min', 'p25', 'median', 'p75', 'max', 'mean'], map(float, [array.min(), *np.quantile(array, [.25, .5, .75]), array.max(), array.mean()]))) if len(array) else {}


def stats(rows):
    originals = [row for row in rows if row['transform'] == 'original']
    result = {'seed': SEED, 'time_utc': now(), 'all_count': len(rows), 'original_count': len(originals), 'derived_count': len(rows) - len(originals), 'sources': dict(collections.Counter(row['source'] for row in originals)), 'generators': dict(collections.Counter(row['generator'] for row in originals)), 'splits': dict(collections.Counter(row['split'] for row in rows)), 'split_labels': {}, 'by_label_original': {}, 'source_splits': {}, 'generator_splits': {}, 'metadata_shortcut': {}}
    for split in ['calibration', 'test', 'test_degraded']:
        result['split_labels'][split] = dict(collections.Counter(str(row['label']) for row in rows if row['split'] == split))
    for field, dest in [('source', 'source_splits'), ('generator', 'generator_splits')]:
        for name in sorted({row[field] for row in originals}):
            result[dest][name] = dict(collections.Counter(row['split'] for row in originals if row[field] == name))
    for label in [0, 1]:
        subset = [row for row in originals if row['label'] == label]
        result['by_label_original'][str(label)] = {'count': len(subset), 'codec': dict(collections.Counter(row['codec'] for row in subset)), 'container': dict(collections.Counter(row['container'] for row in subset)), 'resolution': dict(collections.Counter(f"{row['width']}x{row['height']}" for row in subset)), 'fps': dict(collections.Counter(str(round(row['notes']['metadata']['fps'], 3)) for row in subset)), 'duration_seconds': summary_numbers([row['duration_seconds'] for row in subset]), 'bit_rate': summary_numbers([row['notes']['metadata']['bit_rate'] for row in subset]), 'file_size': summary_numbers([row['notes']['metadata']['file_size'] for row in subset]), 'preferred_5_to_30_seconds': sum(5 <= row['duration_seconds'] <= 30 for row in subset)}
    baseline_path = ROOT / 'metadata_baseline.json'
    if baseline_path.exists():
        result['metadata_shortcut'] = json.loads(baseline_path.read_text(encoding='utf-8'))
    save_json(ROOT / 'stats.json', result)
    return result


def baseline(rows):
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import roc_auc_score, accuracy_score
    train = [row for row in rows if row['split'] == 'calibration']
    test = [row for row in rows if row['split'] == 'test']
    def features(subset):
        return [[row['width'], row['height'], row['duration_seconds'], row['codec'], row['container'], (ROOT / row['path']).stat().st_size] for row in subset]
    preprocess = ColumnTransformer([('numeric', 'passthrough', [0, 1, 2, 5]), ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [3, 4])])
    classifier = Pipeline([('features', preprocess), ('tree', DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=SEED))])
    classifier.fit(features(train), [row['label'] for row in train])
    probabilities = classifier.predict_proba(features(test))[:, 1]
    predictions = classifier.predict(features(test))
    auc = roc_auc_score([row['label'] for row in test], probabilities)
    importances = list(map(float, classifier.named_steps['tree'].feature_importances_))
    transformed_names = ['width', 'height', 'duration_seconds', 'file_size'] + list(preprocess.named_transformers_['categorical'].get_feature_names_out(['codec', 'container']))
    result = {'model': 'DecisionTreeClassifier', 'hyperparameters': {'max_depth': 3, 'min_samples_leaf': 5, 'random_state': SEED}, 'allowed_features': ['width', 'height', 'duration_seconds', 'codec', 'container', 'file_size'], 'train_split': 'calibration', 'evaluation_split': 'test', 'train_count': len(train), 'test_count': len(test), 'test_auc': float(auc), 'test_accuracy': float(accuracy_score([row['label'] for row in test], predictions)), 'feature_importance': dict(zip(transformed_names, importances)), 'sklearn_version': sklearn.__version__, 'policy': 'One fit, one test evaluation; no hyperparameter search and no sample changes after AUC; no pixels/audio/features from detection models.', 'tree_rules': export_text(classifier.named_steps['tree'], feature_names=transformed_names)}
    save_json(ROOT / 'metadata_baseline.json', result)
    save_jsonl(ROOT / 'evidence/metadata_baseline_predictions.jsonl', [{'sample_id': row['sample_id'], 'label': row['label'], 'metadata_only_probability': float(score)} for row, score in zip(test, probabilities)])
    summary = json.loads((ROOT / 'stats.json').read_text(encoding='utf-8'))
    summary['metadata_shortcut'] = result
    save_json(ROOT / 'stats.json', summary)
    print('METADATA BASELINE', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'prepare', 'exclude', 'collect', 'finalize', 'validate', 'baseline'], default='all')
    stage = parser.parse_args().stage
    if stage in ['all', 'prepare']:
        ensure_evidence()
        make_candidates()
    if stage in ['all', 'exclude']:
        build_exclusions()
    if stage in ['all', 'collect']:
        candidates = load_jsonl(ROOT / 'evidence/candidates.jsonl')
        plan = json.loads((ROOT / 'evidence/sampling_plan.json').read_text(encoding='utf-8'))
        excluded = {row['sha256'] for row in load_jsonl(ROOT / 'evidence/exclusions.jsonl')}
        collect(candidates, plan['quotas'], excluded)
    if stage in ['all', 'finalize']:
        rows = assign_splits(load_jsonl(ROOT / 'evidence/selected_originals.jsonl'))
        derived = derive(rows)
        write_manifest(rows + derived)
    if stage in ['all', 'finalize', 'validate']:
        rows = load_jsonl(ROOT / 'manifest.jsonl')
        excluded = {row['sha256'] for row in load_jsonl(ROOT / 'evidence/exclusions.jsonl')}
        validate(rows, excluded)
        stats(rows)
    if stage in ['all', 'baseline']:
        baseline(load_jsonl(ROOT / 'manifest.jsonl'))


if __name__ == '__main__':
    main()
