import io
import json
import os
import sys
import time
import threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import requests
import pyarrow.parquet as pq
EVIDENCE = ROOT / 'evidence'
EVIDENCE.mkdir(exist_ok=True)
class AnonymousAuth(requests.auth.AuthBase):
    def __call__(self, request):
        return request
HTTP_LOCAL = threading.local()
def get(url, **kwargs):
    if not hasattr(HTTP_LOCAL, 'session'):
        HTTP_LOCAL.session = requests.Session()
        HTTP_LOCAL.session.auth = AnonymousAuth()
    for attempt in range(5):
        try:
            response = HTTP_LOCAL.session.get(url, timeout=(20, 90), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.cache = url, size, 0, {}
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not size:
            return b''
        key = (self.position, size)
        for (start, length), payload in self.cache.items():
            if start <= self.position and self.position + size <= start + length:
                result = payload[self.position-start:self.position-start+size]
                self.position += size
                return result
        if key not in self.cache:
            response = get(self.url + f'?range_start={self.position}&range_size={size}', headers={'Range': f'bytes={self.position}-{self.position+size-1}'})
            if response.status_code != 206 or len(response.content) != size:
                raise RuntimeError(f'Range not honored: {response.status_code} {len(response.content)} expected {size}')
            self.cache[key] = response.content
        self.position += size
        return self.cache[key]
def main():
    repo = 'ComplexDataLab/OpenFake'
    pinned_revision = '3247b942f18d39d1fd476b1077852af033d35d8c'
    info = get(f'https://huggingface.co/api/datasets/{repo}/revision/{pinned_revision}').json()
    revision = info['sha']
    assert revision == pinned_revision
    (EVIDENCE / 'openfake_api.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    card = get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md').text
    (EVIDENCE / 'openfake_README.md').write_text(card, encoding='utf-8')
    entries = get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/core?limit=1000').json()
    (EVIDENCE / 'openfake_tree.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    for split in ['test', 'validation']:
        entry = next(item for item in entries if item['path'].startswith(f'core/{split}-'))
        remote = RemoteFile(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry["path"]}', entry['size'])
        parquet = pq.ParquetFile(remote)
        print(split, revision, parquet.schema, 'groups', parquet.num_row_groups, flush=True)
        from concurrent.futures import ThreadPoolExecutor
        def prefetch(index):
            group = parquet.metadata.row_group(index)
            columns = [group.column(column) for column in range(2, group.num_columns)]
            start = min(min(column.dictionary_page_offset or column.data_page_offset, column.data_page_offset) for column in columns)
            stop = max(min(column.dictionary_page_offset or column.data_page_offset, column.data_page_offset) + column.total_compressed_size for column in columns)
            url = remote.url + f'?range_start={start}&range_size={stop-start}'
            response = get(url, headers={'Range': f'bytes={start}-{stop-1}'})
            assert response.status_code == 206 and len(response.content) == stop-start
            return (start, stop-start), response.content
        with ThreadPoolExecutor(max_workers=12) as pool:
            remote.cache.update(pool.map(prefetch, range(parquet.num_row_groups)))
        metadata = parquet.read(columns=['model', 'label', 'type', 'prompt', 'row_id', 'image_sha256']).to_pylist()
        from collections import Counter
        print(Counter((row['label'], row['model'], row['type']) for row in metadata), flush=True)
        print([(index, parquet.metadata.row_group(index).num_rows, parquet.metadata.row_group(index).total_byte_size) for index in range(min(4,parquet.num_row_groups))], flush=True)
        (EVIDENCE / f'openfake_{split}_metadata.json').write_text(json.dumps({'entry': entry,'revision':revision,'rows':metadata}, ensure_ascii=False), encoding='utf-8')
if __name__ == '__main__':
    main()
