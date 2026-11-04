import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from parquet_ranges import range_bytes, locate_images
from probe_sources import RemoteFile, pq, get

SEED = 20261104
REVISION = '3247b942f18d39d1fd476b1077852af033d35d8c'
REPO = 'ComplexDataLab/OpenFake'
FIELDS = ['sample_id','label','source','generator','license','origin','sha256','path','width','height','format','split','transform','group_id','notes']
REAL_QUOTAS = {'docci':333,'imagenet':334,'pexels':333}
FAKE_QUOTAS = {'FLUX':200,'Z-Image':180,'GPT-Image':160,'Midjourney':120,'Recraft':70,'ERNIE':60,'Seedream':50,'Stable-Diffusion':100,'Imagen':35,'DALL-E':25}
MODELS = {
    'flux.2-klein-9b':'FLUX', 'z-image-turbo':'Z-Image', 'gpt-image-1.5':'GPT-Image',
    'midjourney-7':'Midjourney', 'recraft-v2':'Recraft','recraft-v3':'Recraft',
    'ernie-image':'ERNIE','ernie-image-turbo':'ERNIE',
    'seedream-v4.0':'Seedream','seedream-v4.5':'Seedream','seedream-v5.0':'Seedream',
    'sd-3.5':'Stable-Diffusion','sd-2.1':'Stable-Diffusion','sdxl':'Stable-Diffusion','sdxl-1.0':'Stable-Diffusion',
    'imagen-3.0-002':'Imagen','imagen-4.0':'Imagen','dalle-3':'DALL-E',
}

def dump(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def rank(value):
    return digest(f'{SEED}:{value}'.encode())

def budget():
    size = sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())
    if size > 14_000_000_000:
        raise RuntimeError(f'Workspace budget reserve reached: {size}')
    return size

def record_event(kind, **details):
    with (ROOT/'evidence'/'events.jsonl').open('a',encoding='utf-8') as output:
        output.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'kind':kind,**details},ensure_ascii=False)+'\n')

def catalog():
    candidates = []
    for upstream_split in ['test','validation']:
        metadata_path = ROOT/'evidence'/f'openfake_{upstream_split}_metadata.json'
        if not metadata_path.exists():
            raise RuntimeError('Run tools/probe_sources.py first to create pinned source metadata.')
        metadata = load(metadata_path)
        assert metadata['revision'] == REVISION
        for index,row in enumerate(metadata['rows']):
            if row['label']=='real' and row['model'] in REAL_QUOTAS:
                category = 'real:'+row['model']
            elif row['label']=='fake' and row['model'] in MODELS and str(row['type']).lower() not in {'video','base-watermarked'}:
                category = 'fake:'+MODELS[row['model']]
            else:
                continue
            candidates.append({**row,'upstream_split':upstream_split,'upstream_row':index,'category':category,'shard':metadata['entry']['path'],'shard_size':metadata['entry']['size']})
    dump('evidence/eligible_catalog.json',candidates)
    return candidates

def candidate_plan():
    path=ROOT/'evidence'/'candidate_plan.json'
    if path.exists(): return load(path)
    candidates = catalog()
    categories = defaultdict(list)
    for row in candidates: categories[row['category']].append(row)
    plan=[]
    for category, rows in sorted(categories.items()):
        family=category.split(':',1)[1]
        quota=(REAL_QUOTAS if category.startswith('real:') else FAKE_QUOTAS)[family]
        reserve=math.ceil(quota*0.25)+12
        if family=='imagenet': reserve=quota
        rows.sort(key=lambda row:rank(row['row_id']))
        plan.extend(rows[:quota+reserve])
    dump(path,plan)
    print('Candidate plan:',dict(Counter(row['category'] for row in plan)),flush=True)
    return plan

def index_sources(workers):
    plan=candidate_plan()
    jobs=[]
    for upstream_split in ['test','validation']:
        rows=[row for row in plan if row['upstream_split']==upstream_split]
        metadata=load(f'evidence/openfake_{upstream_split}_metadata.json')
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{metadata["entry"]["path"]}'
        remote=RemoteFile(url,metadata['entry']['size'])
        parquet=pq.ParquetFile(remote)
        cursor=0
        for group_index in range(parquet.num_row_groups):
            group=parquet.metadata.row_group(group_index)
            wanted={row['upstream_row']-cursor for row in rows if cursor <= row['upstream_row'] < cursor+group.num_rows}
            if wanted: jobs.append((upstream_split,group_index,cursor,url,group.column(0),wanted))
            cursor+=group.num_rows
    def execute(job):
        upstream_split,group_index,cursor,url,column,wanted=job
        path=ROOT/'evidence'/'offsets'/f'{upstream_split}_{group_index}.json'
        if path.exists():
            result=load(path)
            if set(map(int,result))=={cursor+index for index in wanted}: return upstream_split,group_index,len(result),'cached'
        result={str(cursor+index):{'offset':offset,'length':length} for index,(offset,length) in locate_images(url,column,wanted).items()}
        dump(path,result)
        return upstream_split,group_index,len(result),'indexed'
    failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try: print('INDEX',future.result(),flush=True)
            except Exception as error:
                failures.append({'split':job[0],'group':job[1],'error':str(error)})
                print('INDEX_ERROR',failures[-1],flush=True)
    dump('evidence/index_failures.json',failures)
    if failures: raise RuntimeError(f'{len(failures)} indexing jobs failed; rerun --stage index')

def acquire(workers):
    plan=candidate_plan()
    offsets={}
    for path in (ROOT/'evidence'/'offsets').glob('*.json'):
        upstream_split=path.stem.split('_')[0]
        for index,location in load(path).items(): offsets[(upstream_split,int(index))]=location
    def execute(row):
        path=ROOT/'.cache'/'candidates'/f'{row["row_id"]}.bin'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and digest(path.read_bytes())==row['image_sha256']: return row['row_id'],'cached',path.stat().st_size
        location=offsets[(row['upstream_split'],row['upstream_row'])]
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row["shard"]}'
        payload=range_bytes(url,location['offset'],location['length'],False)
        if digest(payload)!=row['image_sha256']: raise RuntimeError('Upstream SHA-256 mismatch')
        with Image.open(io.BytesIO(payload)) as image: image.load()
        temporary=path.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(path)
        return row['row_id'],'downloaded',len(payload)
    failures=[]
    completed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,row):row for row in plan}
        for future in as_completed(futures):
            row=futures[future]
            try:
                result=future.result();completed+=1
                if completed%100==0: print('ACQUIRE',completed,'/',len(plan),result,flush=True)
                if completed%250==0: budget()
            except Exception as error:
                failure={'row_id':row['row_id'],'error':str(error)};failures.append(failure)
                print('DOWNLOAD_ERROR',failure,flush=True)
    dump('evidence/download_failures.json',failures)
    print('ACQUIRE_COMPLETE',completed,'failures',len(failures),flush=True)
    if failures: raise RuntimeError('Downloads incomplete; rerun --stage acquire')

def fingerprint(image):
    rgb=ImageOps.exif_transpose(image).convert('RGB')
    gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    coefficients=dctn(gray,type=2,norm='ortho')[:8,:8].flatten()[1:]
    bits=coefficients>np.median(coefficients)
    phash=sum(int(bit)<<index for index,bit in enumerate(bits))
    thumb=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS),dtype=np.int16)
    pixels=digest(rgb.tobytes()+str(rgb.size).encode())
    return pixels,phash,thumb

def select():
    plan=candidate_plan()
    plan.sort(key=lambda row:rank(row['row_id']))
    chosen=[];excluded=[];counts=Counter();hashes={};pixels={};visuals=[]
    for row in plan:
        category=row['category'];family=category.split(':',1)[1]
        quota=(REAL_QUOTAS if category.startswith('real:') else FAKE_QUOTAS)[family]
        if counts[category]>=quota:continue
        path=ROOT/'.cache'/'candidates'/f'{row["row_id"]}.bin'
        if not path.exists():continue
        payload=path.read_bytes()
        assert digest(payload)==row['image_sha256']
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width,height=image.size;fmt=image.format
            if min(width,height)<256:
                excluded.append({'row_id':row['row_id'],'reason':'short_edge_below_256','width':width,'height':height});continue
            pixel_hash,phash,thumb=fingerprint(image)
        duplicate=hashes.get(row['image_sha256']) or pixels.get(pixel_hash)
        if not duplicate:
            for earlier_hash,earlier_thumb,earlier_id in visuals:
                if (phash^earlier_hash).bit_count()<=4 and float(np.mean(np.abs(thumb-earlier_thumb)))<=5.0:
                    duplicate=earlier_id;break
        if duplicate:
            excluded.append({'row_id':row['row_id'],'reason':'duplicate_sha_pixels_or_phash4_rgbmae5','duplicate_of':duplicate});continue
        hashes[row['image_sha256']]=row['row_id'];pixels[pixel_hash]=row['row_id'];visuals.append((phash,thumb,row['row_id']))
        chosen.append({**row,'width':width,'height':height,'format':fmt,'size_bytes':len(payload),'pixel_sha256':pixel_hash,'phash':f'{phash:016x}'})
        counts[category]+=1
    dump('evidence/selection_exclusions.json',excluded)
    expected={'real:'+key:value for key,value in REAL_QUOTAS.items()}|{'fake:'+key:value for key,value in FAKE_QUOTAS.items()}
    dump('evidence/selection_counts.json',dict(counts))
    if dict(counts)!=expected:
        raise RuntimeError(f'Insufficient unique images: actual={dict(counts)}, expected={expected}; extend fixed candidate reserve without detector scores')
    dump('evidence/selected_originals.json',chosen)
    print('SELECTED',dict(counts),flush=True)
    return chosen

def assign_splits(rows):
    group_keys=[];strata=[]
    for row in rows:
        prompt=' '.join((row.get('prompt') or '').casefold().split())
        key='prompt:'+prompt if prompt else 'image:'+row['image_sha256']
        group_keys.append('g_'+digest(key.encode())[:24])
        strata.append(row['category'] if row['label']=='real' else 'fake:'+row['model'])
    folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    calibration=set()
    for fold,(_,held_out) in enumerate(folds.split(np.zeros(len(rows)),strata,group_keys)):
        if fold<2:calibration.update(held_out.tolist())
    group_split={group:('calibration' if index in calibration else 'test') for index,group in enumerate(group_keys)}
    for stratum in sorted(set(strata)):
        groups=sorted({group_keys[index] for index,value in enumerate(strata) if value==stratum},key=rank)
        if len(groups)==1:group_split[groups[0]]='test'
        elif len({group_split[group] for group in groups})==1:
            group_split[groups[0]]='calibration';group_split[groups[-1]]='test'
    for row,group in zip(rows,group_keys):row['group_id']=group;row['split']=group_split[group]
    return rows

def metadata_auc(rows):
    formats=sorted({row['format'] for row in rows})
    features=np.array([[row['width'],row['height'],row['size_bytes']]+[int(row['format']==fmt) for fmt in formats] for row in rows])
    labels=np.array([int(row['label']=='fake') if isinstance(row['label'],str) else row['label'] for row in rows])
    calibration=np.array([row['split']=='calibration' for row in rows]);test=~calibration
    classifier=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,random_state=SEED)
    classifier.fit(features[calibration],labels[calibration])
    auc=float(roc_auc_score(labels[test],classifier.predict_proba(features[test])[:,1]))
    return {'auc':auc,'model':'DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=20261104)','features':['width','height','file_size_bytes']+['format_'+fmt for fmt in formats], 'calibration_n':int(calibration.sum()),'test_n':int(test.sum()),'feature_importances':classifier.feature_importances_.tolist()}

def standardize(image):
    image=ImageOps.exif_transpose(image).convert('RGB')
    image=ImageOps.fit(image,(512,512),method=Image.Resampling.LANCZOS,centering=(0.5,0.5))
    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=95,subsampling=0,optimize=False)
    payload=buffer.getvalue()
    return payload

def summarize(rows):
    result={'count':len(rows),'label':dict(Counter(str(row['label']) for row in rows)),'source':dict(Counter(row['source'] for row in rows)),'generator':dict(Counter(row['generator'] for row in rows)),'split':dict(Counter(row['split'] for row in rows)),'format':dict(Counter(row['format'] for row in rows)),'by_label':{},'source_split':{},'generator_split':{}}
    for label in [0,1]:
        subset=[row for row in rows if row['label']==label]
        result['by_label'][str(label)]={'count':len(subset),'format':dict(Counter(row['format'] for row in subset)), 'resolution':dict(Counter(f'{row["width"]}x{row["height"]}' for row in subset))}
        for field in ['width','height','size_bytes']:
            values=[row[field] for row in subset]
            result['by_label'][str(label)][field]={'min':int(min(values)),'p25':float(np.percentile(values,25)),'median':float(np.median(values)),'p75':float(np.percentile(values,75)),'max':int(max(values))}
    for field in ['source','generator']:
        for value in sorted({row[field] for row in rows}):result[field+'_split'][value]=dict(Counter(row['split'] for row in rows if row[field]==value))
    return result

def manifest_row(row):
    generated=row['label']=='fake'
    source='OpenFake' if generated else 'OpenFake_'+row['model'].upper()
    notes=['upstream_label='+row['label'],'upstream_type='+str(row['type']),'upstream_row_id='+row['row_id'],'upstream_image_sha256='+row['image_sha256'],'family='+MODELS.get(row['model'],'real'),'Original bytes extracted from UNCOMPRESSED PLAIN Parquet pages and verified against publisher image_sha256.','Aggregator CC-BY-NC-4.0; individual image rights not independently verified.']
    if row['model']=='imagenet':notes.append('KNOWN SOURCE-LEVEL OVERLAP: ImageNet is listed among Community Forensics real-image sources; exact item/training overlap unverified. Do not claim contamination-free evaluation.')
    if generated:notes.append('Generator family overlap with Community Forensics may exist; exact sample overlap not verified. No Community Forensics images downloaded.')
    return {'sample_id':'b_'+row['row_id'].removeprefix('openfake-'),'label':int(generated),'source':source,'generator':row['model'] if generated else 'real','license':'CC-BY-NC-4.0 (OpenFake v2 compilation); underlying image rights 未核实','origin':f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row["shard"]}#row={row["upstream_row"]}','sha256':row['image_sha256'],'path':'','width':row['width'],'height':row['height'],'format':row['format'],'split':row['split'],'transform':'original','group_id':row['group_id'],'notes':'; '.join(notes),'size_bytes':row['size_bytes']}

def build():
    chosen=assign_splits(select())
    before=metadata_auc(chosen)
    normalized=before['auc']>0.75
    rows=[];before_rows=[];raw_index=[]
    for build_index, chosen_row in enumerate(chosen, 1):
        row=manifest_row(chosen_row);before_rows.append(dict(row))
        original=(ROOT/'.cache'/'candidates'/f'{chosen_row["row_id"]}.bin').read_bytes()
        extension={'JPEG':'jpg','PNG':'png','WEBP':'webp'}.get(row['format'],row['format'].lower())
        raw_path=Path('raw')/row['source']/(row['sample_id']+'.'+extension)
        (ROOT/raw_path).parent.mkdir(parents=True,exist_ok=True)
        (ROOT/raw_path).write_bytes(original)
        raw_index.append({'sample_id':row['sample_id'],'path':raw_path.as_posix(),'sha256':row['sha256'],'size_bytes':len(original),'width':row['width'],'height':row['height'],'format':row['format'],'label':row['label']})
        if normalized:
            with Image.open(io.BytesIO(original)) as image:payload=standardize(image)
            row.update(width=512,height=512,format='JPEG',size_bytes=len(payload),sha256=digest(payload))
            row['notes']+='; PREPROCESSING (triggered by metadata AUC>0.75): EXIF transpose, RGB, center crop/resize 512x512 Lanczos, JPEG q95 4:4:4, remove metadata; NO file-size padding; raw bytes at '+raw_path.as_posix()+'. transform=original means primary set, NOT untouched upstream bytes.'
            extension='jpg'
        else:payload=original
        path=Path('data')/row['source']/(row['sample_id']+'.'+extension)
        (ROOT/path).parent.mkdir(parents=True,exist_ok=True);(ROOT/path).write_bytes(payload)
        row['path']=path.as_posix();rows.append(row)
        if build_index%250==0: print('BUILD_PRIMARY',build_index,flush=True)
    after=metadata_auc(rows)
    dump('evidence/raw_index.json',raw_index)
    dump('evidence/metadata_baseline.json',{'before':before,'after':after,'normalized':normalized,'policy':'Equal processing when raw metadata AUC>0.75; no detector or image-content model used. Plain JPEG without padding. Metadata-only test inspection authorized; no detector blind-test scores observed. Earlier padding trial and removal decision retained in metadata_trials.json.'})
    original_rows=list(rows)
    parent_ids=[]
    for label in [0,1]:
        candidates=sorted([row for row in rows if row['label']==label and row['split']=='test'],key=lambda row:rank('degradation:'+row['sample_id']))
        parents=candidates[:200]
        assert len(parents)==200
        for parent in parents:
            parent_ids.append(parent['sample_id'])
            for transform in ['jpeg_q75','resize50_jpeg_q85']:
                with Image.open(ROOT/parent['path']) as original:
                    image=ImageOps.exif_transpose(original).convert('RGB')
                    if transform=='resize50_jpeg_q85':image=image.resize((max(1,image.width//2),max(1,image.height//2)),Image.Resampling.LANCZOS)
                    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=75 if transform=='jpeg_q75' else 85,subsampling=2,optimize=False)
                    payload=buffer.getvalue();width,height=image.size
                derived={**parent,'sample_id':parent['sample_id']+'__'+transform,'sha256':digest(payload),'width':width,'height':height,'format':'JPEG','split':'test_degraded','transform':transform,'size_bytes':len(payload)}
                path=Path('derived')/parent['source']/(derived['sample_id']+'.jpg')
                (ROOT/path).parent.mkdir(parents=True,exist_ok=True);(ROOT/path).write_bytes(payload)
                derived['path']=path.as_posix();derived['notes']+='; parent_sample_id='+parent['sample_id']+'; derived from delivered test primary, not from raw upstream image; Pillow JPEG subsampling=2.'
                rows.append(derived)
    dump('evidence/degraded_parent_ids.json',parent_ids)
    with (ROOT/'manifest.jsonl').open('w',encoding='utf-8',newline='\n') as output:
        for row in sorted(rows,key=lambda item:item['sample_id']):output.write(json.dumps({field:row[field] for field in FIELDS},ensure_ascii=False)+'\n')
    stats={'seed':SEED,'revision':REVISION,'before_processing':summarize(before_rows),'primary':summarize(original_rows),'all':summarize(rows),'generator_families':dict(Counter(MODELS[row['generator']] for row in original_rows if row['label']==1)),'metadata_baseline':load('evidence/metadata_baseline.json'),'workspace_bytes':budget()}
    dump('stats.json',stats)
    print('BUILD_COMPLETE',len(rows),before,after,flush=True)
    validate()

def validate():
    rows=[json.loads(line) for line in (ROOT/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    hashes={};groups=defaultdict(set);ids={};errors=[];visuals=[];pixel_hashes={}
    for row in rows:
        if list(row)!=FIELDS:errors.append('incorrect_fields:'+row['sample_id'])
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT):errors.append('path_escape:'+row['sample_id']);continue
        if not path.exists():errors.append('missing:'+row['sample_id']);continue
        payload=path.read_bytes();checksum=digest(payload)
        if checksum!=row['sha256']:errors.append('sha_mismatch:'+row['sample_id'])
        if checksum in hashes:errors.append('duplicate_sha:'+row['sample_id']+':'+hashes[checksum])
        hashes[checksum]=row['sample_id']
        if row['sample_id'] in ids:errors.append('duplicate_sample_id:'+row['sample_id'])
        ids[row['sample_id']]=row
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if (image.width,image.height,image.format)!=(row['width'],row['height'],row['format']):errors.append('image_metadata_mismatch:'+row['sample_id'])
                if row['transform']=='original':
                    pixel_hash,phash,thumb=fingerprint(image)
                    if pixel_hash in pixel_hashes:errors.append('duplicate_primary_pixels:'+row['sample_id'])
                    pixel_hashes[pixel_hash]=row['sample_id']
                    for earlier_hash,earlier_thumb,earlier_id in visuals:
                        if (phash^earlier_hash).bit_count()<=4 and float(np.mean(np.abs(thumb-earlier_thumb)))<=5.0:errors.append('duplicate_primary_perceptual:'+row['sample_id']+':'+earlier_id)
                    visuals.append((phash,thumb,row['sample_id']))
        except Exception as error:errors.append('decode:'+row['sample_id']+':'+str(error))
        groups[row['group_id']].add(row['split'])
    for group,splits in groups.items():
        if 'calibration' in splits and ('test' in splits or 'test_degraded' in splits):errors.append('group_leakage:'+group)
    primary=[row for row in rows if row['transform']=='original'];derived=[row for row in rows if row['split']=='test_degraded']
    for row in derived:
        parent=ids.get(row['sample_id'].split('__')[0])
        if not parent or parent['split']!='test' or parent['group_id']!=row['group_id'] or parent['label']!=row['label']:errors.append('invalid_derived_parent:'+row['sample_id'])
    for entry in load('evidence/raw_index.json'):
        if digest((ROOT/entry['path']).read_bytes())!=entry['sha256']:errors.append('raw_sha_mismatch:'+entry['sample_id'])
    for key in ['source','generator']:
        for value in {row[key] for row in primary}:
            splits={row['split'] for row in primary if row[key]==value}
            if splits!={'calibration','test'}:errors.append('stratum_missing_split:'+key+':'+value)
    if Counter(row['label'] for row in primary)!=Counter({0:1000,1:1000}):errors.append('primary_label_counts')
    if len(derived)!=800:errors.append('derived_count')
    parent_transforms=defaultdict(list)
    for row in derived:parent_transforms[row['sample_id'].split('__')[0]].append(row['transform'])
    if any(sorted(transforms)!=['jpeg_q75','resize50_jpeg_q85'] for transforms in parent_transforms.values()):errors.append('incorrect_parent_transforms')
    if Counter(ids[parent]['label'] for parent in parent_transforms)!=Counter({0:200,1:200}):errors.append('degraded_parent_label_counts')
    raw_entries=load('evidence/raw_index.json')
    if len({entry['sha256'] for entry in raw_entries})!=2000:errors.append('raw_hash_count')
    families=Counter(MODELS[row['generator']] for row in primary if row['label']==1)
    if len(families)<8 or max(families.values())>250:errors.append('generator_family_constraint')
    if load('evidence/metadata_baseline.json')['after']['auc']>0.75:errors.append('metadata_auc_threshold')
    result={'passed':not errors,'checked_manifest_files':len(rows),'checked_raw_files':len(primary),'unique_hashes':len(hashes),'groups':len(groups),'primary_perceptual_pairs_checked':len(primary)*(len(primary)-1)//2,'perceptual_rule':'63-bit pHash Hamming <=4 AND 32x32 RGB mean absolute difference <=5; exact decoded pixels also checked; intentional degraded-vs-parent similarity excluded','errors':errors,'workspace_bytes':budget(),'checked_at':datetime.now(timezone.utc).isoformat()}
    dump('evidence/validation.json',result)
    print('VALIDATION',json.dumps(result,ensure_ascii=False),flush=True)
    if errors:raise RuntimeError('Validation failed')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','index','acquire','build','validate'],default='all')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    budget()
    if args.stage in ['all','index']:
        if any(not (ROOT/'evidence'/f'openfake_{split}_metadata.json').exists() for split in ['test','validation']):
            from probe_sources import main as probe_metadata
            probe_metadata()
        index_sources(args.workers)
    if args.stage in ['all','acquire']:acquire(args.workers)
    if args.stage in ['all','build']:
        build()
        from audit_content import main as audit_content
        from write_report import main as write_report
        from verify_reproducibility import main as verify_reproducibility
        audit_content()
        verify_reproducibility()
        write_report()
    if args.stage=='validate':validate()
if __name__=='__main__':main()
