import hashlib
import struct
import threading
import time
SIGNED_URLS = {}
SIGNED_LOCK = threading.Lock()
from concurrent.futures import ThreadPoolExecutor
from probe_sources import ROOT, get, pq, RemoteFile

class CompactReader:
    def __init__(self, data):
        self.data, self.pos = data, 0
    def byte(self):
        value = self.data[self.pos]
        self.pos += 1
        return value
    def varint(self):
        value, shift = 0, 0
        while True:
            byte = self.byte()
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7
    def value(self, kind):
        if kind in (1,2): return kind == 1
        if kind == 3: return self.byte()
        if kind in (4,5,6):
            value = self.varint()
            return (value >> 1) ^ -(value & 1)
        if kind == 7:
            result = self.data[self.pos:self.pos+8];self.pos += 8;return result
        if kind == 8:
            size = self.varint();result = self.data[self.pos:self.pos+size];self.pos += size;return result
        if kind in (9,10):
            header = self.byte();size = header >> 4
            if size == 15: size = self.varint()
            return [self.value(header & 15) for _ in range(size)]
        if kind == 12: return self.structure()
        raise ValueError(f'Unsupported compact kind {kind}')
    def structure(self):
        result, field = {}, 0
        while True:
            header = self.byte()
            if header == 0: return result
            field = field + (header >> 4) if header >> 4 else self.value(4)
            result[field] = self.value(header & 15)

def range_bytes(url, start, length, cache=True):
    if not cache and length > 1048576:
        parts = [(offset, min(1048576, start+length-offset)) for offset in range(start,start+length,1048576)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            return b''.join(pool.map(lambda part: range_bytes(url,part[0],part[1],True), parts))
    key = hashlib.sha256(f'{url}:{start}:{length}'.encode()).hexdigest()
    path = ROOT / '.cache' / 'ranges' / key
    if cache and path.exists():
        data = path.read_bytes()
        if len(data) == length: return data
    with SIGNED_LOCK:
        if url not in SIGNED_URLS or time.monotonic()-SIGNED_URLS[url][1]>1800:
            resolved = get(url + f'?resolve_public_range={int(time.time())}', headers={'Range':'bytes=0-3'})
            SIGNED_URLS[url] = (resolved.url, time.monotonic())
        signed_url = SIGNED_URLS[url][0]
    response = get(signed_url, headers={'Range':f'bytes={start}-{start+length-1}'})
    if response.headers.get('Content-Range','').split('/')[0] != f'bytes {start}-{start+length-1}':
        raise ValueError('Unexpected Content-Range')
    if response.status_code != 206 or len(response.content) != length:
        raise ValueError(f'Invalid range response {response.status_code}, {len(response.content)}, expected {length}')
    data = response.content
    if cache:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
    return data

def levels(data, count, width=2):
    reader = CompactReader(data)
    values = []
    while len(values) < count:
        header = reader.varint()
        if header & 1:
            groups = header >> 1
            number = int.from_bytes(data[reader.pos:reader.pos+groups*width], 'little')
            reader.pos += groups*width
            values.extend((number >> (index*width)) & ((1<<width)-1) for index in range(groups*8))
        else:
            run = header >> 1
            value = reader.byte()
            values.extend([value]*run)
    return values[:count]

def locate_images(url, column, wanted):
    assert column.compression == 'UNCOMPRESSED', column.compression
    assert column.dictionary_page_offset is None
    cursor = column.data_page_offset
    end = cursor + column.total_compressed_size
    row_index = 0
    results = {}
    while cursor < end and row_index <= max(wanted):
        probe = range_bytes(url,cursor,min(4096,end-cursor))
        reader = CompactReader(probe)
        header = reader.structure()
        assert header[1] == 0, header
        assert header[2] == header[3], header
        page = header[5]
        count = page[1]
        if any(index in wanted for index in range(row_index,row_index+count)):
            assert page[2] == 0, page
            payload_start = cursor+reader.pos
            level_length = struct.unpack_from('<I',probe,reader.pos)[0]
            definitions = levels(probe[reader.pos+4:reader.pos+4+level_length],count)
            image_start = payload_start + 4 + level_length
            for local_index, definition in enumerate(definitions):
                absolute_index = row_index+local_index
                if definition == 2:
                    if cursor <= image_start and image_start+4 <= cursor+len(probe):
                        length = struct.unpack_from('<I',probe,image_start-cursor)[0]
                    else:
                        length = int.from_bytes(range_bytes(url,image_start,4),'little')
                    assert 0 < length < 100_000_000, length
                    if absolute_index in wanted:
                        results[absolute_index] = (image_start+4,length)
                    image_start += 4+length
                elif absolute_index in wanted:
                    raise ValueError('Selected null image')
                if absolute_index >= max(wanted): break
            assert image_start <= payload_start+header[3]
        row_index += count
        cursor += reader.pos + header[3]
    assert set(results)==set(wanted),(results.keys(),wanted)
    return results
