"""只读、受限的文件元数据解析；标识与内容真实性无等价关系。"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
import zlib

from PIL import Image


MAX_VALUE = 2048
MAX_PAYLOAD = 1024 * 1024
MAX_METADATA = 8 * 1024 * 1024
MAX_NODES = 100000
C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")
JUMBF_UUID = bytes.fromhex("6332706100110010800000aa00389b71")
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
AI_SOURCE_TYPES = {
    prefix + term
    for prefix in ("http://cv.iptc.org/newscodes/digitalsourcetype/",
                   "https://cv.iptc.org/newscodes/digitalsourcetype/")
    for term in ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")
}
LABEL_MEANINGS = {
    "1": "属于人工智能生成合成内容",
    "2": "可能为人工智能生成合成内容",
    "3": "疑似为人工智能生成合成内容",
}
FIELDS = {
    "ContentProducer": "content_producer", "ProduceID": "produce_id",
    "ContentPropagator": "content_propagator", "PropagateID": "propagate_id",
    "ReservedCode1": "reserved_code1", "ReservedCode2": "reserved_code2",
}
TOOL_PATTERN = re.compile(
    r"(?<![\w])(?:ComfyUI|AUTOMATIC1111|A1111|Stable[ -]Diffusion|"
    r"InvokeAI|Fooocus|NovelAI|Midjourney|DALL[· -]?E(?:[ -]?[23])?)(?![\w])",
    re.IGNORECASE,
)
COMFY_NODE_TYPES = {
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
    "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "FluxGuidance",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts", ".m2ts", ".wmv", ".flv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif", ".bmp", ".avif", ".heic", ".heif"}


def _clip(value):
    return str(value).encode("utf-8", "replace")[:MAX_VALUE].decode("utf-8", "ignore")


def _decode(value):
    if isinstance(value, bytes):
        if value.startswith(b"ASCII\0\0\0"):
            value = value[8:]
        elif value.startswith(b"UNICODE\0"):
            return value[8:].decode("utf-16", "replace").rstrip("\0")
        try:
            return value.decode("utf-8").rstrip("\0")
        except UnicodeDecodeError:
            return value.decode("latin-1").rstrip("\0")
    return str(value)


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("重复 JSON 字段")
            result[key] = value
        return result

    text = text.strip().strip("\ufeff").rstrip("\0")
    for _ in range(4):
        try:
            value = json.loads(text, object_pairs_hook=pairs)
        except (ValueError, RecursionError):
            decoded = html.unescape(text)
            if decoded != text:
                text = decoded
                continue
            if '\\"' in text:
                try:
                    text = json.loads('"' + text + '"')
                    continue
                except (ValueError, RecursionError):
                    pass
            return None
        if isinstance(value, str):
            text = value
            continue
        return value
    return None


def _credit_codes(value):
    alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    weights = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)
    candidates = []
    for token in re.findall(r"[A-Z0-9]+", value):
        if len(token) == 27 and token.startswith("0011"):
            candidate = token[4:22]
        elif len(token) == 18:
            candidate = token
        else:
            continue
        if not re.fullmatch(r"[159Y][1239][0-9]{6}[0-9A-HJ-NPQRTUWXY]{10}", candidate):
            continue
        check = (31 - sum(alphabet.index(char) * weight for char, weight in zip(candidate[:17], weights)) % 31) % 31
        if candidate[-1] == alphabet[check] and candidate not in candidates:
            candidates.append(candidate)
    return candidates


class _Parser:
    def __init__(self, result):
        self.result = result
        self.used = 0
        self.nodes = 0
        self.label_keys = set()
        self.source_types = []

    def warn(self, message):
        if message not in self.result["warnings"] and len(self.result["warnings"]) < 100:
            self.result["warnings"].append(message)

    def tick(self, depth=0):
        self.nodes += 1
        if self.nodes > MAX_NODES or depth > 24:
            raise ValueError("容器结构数量或嵌套深度超过安全上限")

    def payload(self, reader, offset, length):
        if length > MAX_PAYLOAD or self.used + length > MAX_METADATA:
            self.warn("元数据超过读取上限，已跳过；未发现不代表不存在。")
            return None
        data = reader(offset, length)
        if data is not None:
            self.used += length
        return data

    def c2pa(self, location):
        self.result["c2pa"]["present"] = True
        if location not in self.result["c2pa"]["locations"]:
            self.result["c2pa"]["locations"].append(location)

    def label(self, text, location, named=False):
        value = _json(text)
        if not isinstance(value, dict):
            return
        wrapped = "AIGC" in value
        if wrapped:
            value = value["AIGC"]
            if isinstance(value, str):
                value = _json(value)
        if not (named or wrapped) or not isinstance(value, dict):
            return
        normalized = dict(value)
        if "PropagateID" not in normalized and "PropagatorID" in normalized:
            normalized["PropagateID"] = normalized["PropagatorID"]
            self.warn("发现非标准字段 PropagatorID，兼容映射至 propagate_id；原文保留。")
        required = {"Label", *FIELDS}
        if not required.issubset(normalized) or any(not isinstance(normalized[key], str) for key in required):
            self.warn("AIGC JSON 缺字段或字段类型不符，未作为有效标识。")
            return
        label = normalized["Label"]
        if label not in LABEL_MEANINGS:
            self.warn("AIGC Label 不属于已核实的 1、2、3，未作为有效标识。")
            return
        identity = tuple(normalized[key] for key in sorted(required))
        if identity in self.label_keys:
            return
        if len(self.label_keys) >= 128:
            self.warn("标识数量超过上限，后续标识已跳过。")
            return
        self.label_keys.add(identity)
        codes = []
        for key in ("ContentProducer", "ContentPropagator"):
            for code in _credit_codes(normalized[key]):
                if code not in codes:
                    codes.append(code)
        self.result["aigc_labels"].append({
            "standard": "GB 45438-2025", "location": location,
            "label": label, "label_meaning": LABEL_MEANINGS[label],
            **{target: _clip(normalized[source]) for source, target in FIELDS.items()},
            "credit_codes": codes, "raw": _clip(text),
        })

    def hint(self, source, value):
        item = {"source": source, "value": _clip(value)}
        hints = self.result["generator_hints"]
        if item not in hints and len(hints) < 128:
            hints.append(item)

    def metadata(self, group, key, value, location, target=None):
        value = _decode(value)
        key = str(key)
        target = self.result["metadata"][group] if target is None else target
        if len(target) < 256:
            target[_clip(key)] = _clip(value)
        local = key.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()
        named = "aigc" in local
        self.label(value, location, named)
        if named:
            return
        source = f"{group}:{key}"
        if group == "png_text" and local == "parameters":
            if all(re.search(pattern, value) for pattern in (r"\bSteps:\s*\d+", r"\bSampler:\s*\S+", r"\bSeed:\s*-?\d+")):
                self.hint(source, value)
        if group == "png_text" and local in {"prompt", "workflow"}:
            data = _json(value)
            valid_prompt = isinstance(data, dict) and any(
                isinstance(node, dict) and isinstance(node.get("class_type"), str)
                and node["class_type"] in COMFY_NODE_TYPES and isinstance(node.get("inputs"), dict) for node in data.values()
            )
            valid_workflow = isinstance(data, dict) and isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list) and any(
                isinstance(node, dict) and "id" in node and isinstance(node.get("type"), str)
                and node["type"] in COMFY_NODE_TYPES for node in data["nodes"]
            )
            if valid_prompt or valid_workflow:
                self.hint(source, value)
        if local in {"software", "creatortool"} and TOOL_PATTERN.search(value):
            self.hint(source, value)
        elif local in {"comment", "description", "imagedescription", "usercomment"}:
            match = re.search(r"(?:generated|created|made|生成)\s*(?:by|with|using|使用|自)?\s*[:：]?\s*" + TOOL_PATTERN.pattern, value, re.IGNORECASE)
            if match and not re.search(r"\b(?:not|never|isn't|wasn't)\b|不是|并非|未|非", value[max(0, match.start() - 24):match.start()], re.IGNORECASE):
                self.hint(source, value)

    def xmp(self, data):
        if not data:
            return
        if b"\0" in data.rstrip(b"\0"):
            self.warn("XMP 不是支持的 UTF-8 XML 编码，已跳过。")
            return
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            self.warn("XMP 含 DTD 或实体声明，出于安全原因跳过。")
            return
        try:
            root = ET.fromstring(data.rstrip(b"\0"))
        except (ET.ParseError, ValueError):
            self.warn("XMP XML 无法解析，未用任意文本匹配代替结构解析。")
            return
        for element in root.iter():
            self.tick()
            entries = list(element.attrib.items())
            if not list(element):
                entries.append((element.tag, element.text or ""))
            elif element.tag == "{" + IPTC_NS + "}DigitalSourceType":
                entries.append((element.tag, "".join(element.itertext()).strip()))
            for key, value in entries:
                self.metadata("xmp", key, value, "xmp")
                if key == "{" + IPTC_NS + "}DigitalSourceType":
                    self.source_type(value)
            if element.tag == "{" + IPTC_NS + "}DigitalSourceType":
                resource = element.get("{" + RDF_NS + "}resource")
                if resource:
                    self.source_type(resource)

    def source_type(self, value):
        value = value.strip()
        if value and value not in self.source_types:
            self.source_types.append(value)
            if self.result["digital_source_type"] is None or value in AI_SOURCE_TYPES:
                self.result["digital_source_type"] = _clip(value)
            if len(self.source_types) > 1:
                self.warn("存在多个 DigitalSourceType，优先返回明确生成类；全部值保留在 metadata.xmp。")
            self.result["metadata"]["xmp"]["DigitalSourceType_values"] = [_clip(item) for item in self.source_types[:128]]

    def exif(self, data):
        try:
            exif = Image.Exif()
            exif.load(data)
            values = dict(exif.items())
            try:
                values.update(exif.get_ifd(34665))
            except (KeyError, ValueError, TypeError, SyntaxError, OSError):
                pass
            for code, key in {305: "Software", 270: "ImageDescription", 37510: "UserComment", 271: "Make", 272: "Model"}.items():
                if code in values:
                    self.metadata("exif", key, values[code], "xmp")
        except Exception as exc:
            self.warn(f"EXIF 解析失败：{type(exc).__name__}。")

def _inflate(data):
    decoder = zlib.decompressobj()
    output = decoder.decompress(data, MAX_PAYLOAD + 1)
    if len(output) > MAX_PAYLOAD or not decoder.eof or decoder.unused_data:
        raise ValueError("压缩文本不完整或解压后超过上限")
    return output


def _png(parser, read, size):
    offset = 8
    while offset < size:
        parser.tick()
        header = read(offset, 8)
        if not header:
            raise ValueError("PNG 块头截断")
        length, kind = struct.unpack(">I4s", header)
        if offset + 12 + length > size:
            raise ValueError("PNG 块长度越界")
        if kind in {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"caBX"}:
            if kind == b"caBX" and length > MAX_PAYLOAD:
                parser.c2pa("png:caBX")
                parser.warn("较大 PNG caBX 仅检查块边界，未读取内容或核对 CRC。")
            data = parser.payload(read, offset + 8, length)
            if data is not None:
                checksum = int.from_bytes(read(offset + 8 + length, 4), "big")
                if zlib.crc32(kind + data) & 0xffffffff != checksum:
                    parser.warn("PNG 元数据块 CRC 错误，已跳过。")
                elif kind == b"caBX":
                    if data:
                        parser.c2pa("png:caBX")
                elif kind == b"eXIf":
                    parser.exif(data)
                else:
                    try:
                        key, body = data.split(b"\0", 1)
                        compressed_length = len(body)
                        if not 1 <= len(key) <= 79:
                            raise ValueError("无效关键词")
                        if kind == b"zTXt":
                            if not body or body[0] != 0:
                                raise ValueError("无效压缩方法")
                            body = _inflate(body[1:])
                        elif kind == b"iTXt":
                            flag, method = body[:2]
                            if flag not in (0, 1) or method != 0:
                                raise ValueError("无效压缩标记")
                            body = body[2:].split(b"\0", 2)[2]
                            if flag:
                                body = _inflate(body)
                        parser.used += max(0, len(body) - compressed_length)
                        if parser.used > MAX_METADATA:
                            raise ValueError("累计解压元数据超过上限")
                        key = key.decode("latin-1")
                        value = body.decode("utf-8" if kind == b"iTXt" else "latin-1")
                        parser.metadata("png_text", key, value, "png_text")
                        if key == "XML:com.adobe.xmp":
                            parser.xmp(body)
                    except (ValueError, IndexError, zlib.error):
                        parser.warn("PNG 文本块格式错误或解压超限，已跳过。")
        offset += length + 12
        if kind == b"IEND":
            if length:
                raise ValueError("PNG IEND 长度无效")
            return
    raise ValueError("PNG 缺少 IEND")


def _is_jumbf(data):
    if len(data) < 16 or data[4:8] != b"jumb":
        return False
    length = int.from_bytes(data[:4], "big")
    if length < 16 or length > len(data):
        return False
    offset = 8
    while offset + 8 <= length:
        child_length = int.from_bytes(data[offset:offset + 4], "big")
        if child_length < 8 or offset + child_length > length:
            return False
        if data[offset + 4:offset + 8] == b"jumd":
            body = data[offset + 8:offset + child_length]
            return len(body) >= 22 and body[:16] == JUMBF_UUID and bool(body[16] & 2) and body[17:].split(b"\0", 1)[0] == b"c2pa" and b"\0" in body[17:]
        offset += child_length
    return False


def _jpeg(parser, read, size):
    offset = 2
    fragments = {}
    extended = {}
    terminated = False
    while offset < size:
        parser.tick()
        marker = read(offset, 2)
        if marker is None or marker[0] != 255:
            raise ValueError("JPEG 段标记无效")
        if marker[1] == 255:
            offset += 1
            continue
        if marker[1] in (0xda, 0xd9):
            terminated = True
            break
        if marker[1] in {0x01, *range(0xd0, 0xd8)}:
            offset += 2
            continue
        length_data = read(offset + 2, 2)
        if length_data is None:
            raise ValueError("JPEG 段头截断")
        length = int.from_bytes(length_data, "big")
        if length < 2 or offset + length + 2 > size:
            raise ValueError("JPEG 段长度越界")
        if marker[1] in (0xe1, 0xeb, 0xfe):
            data = parser.payload(read, offset + 4, length - 2)
            if data is not None:
                if marker[1] == 0xfe:
                    parser.metadata("exif", "Comment", data, "xmp")
                elif marker[1] == 0xe1:
                    prefix = b"http://ns.adobe.com/xap/1.0/\0"
                    extension = b"http://ns.adobe.com/xmp/extension/\0"
                    if data.startswith(b"Exif\0\0"):
                        parser.exif(data)
                    elif data.startswith(prefix):
                        parser.xmp(data[len(prefix):])
                    elif data.startswith(extension):
                        part = data[len(extension):]
                        if len(part) >= 40:
                            guid = part[:32]
                            total, start = struct.unpack(">II", part[32:40])
                            if total <= MAX_PAYLOAD:
                                extended.setdefault((guid, total), []).append((start, part[40:]))
                            else:
                                parser.warn("扩展 XMP 超过读取上限，已跳过。")
                elif data[:2] == b"JP" and len(data) >= 16:
                    instance = data[2:4]
                    sequence = int.from_bytes(data[4:8], "big")
                    fragments.setdefault(instance, []).append((sequence, data[8:]))
        offset += length + 2
    if not terminated:
        raise ValueError("JPEG 缺少扫描或结束标记")
    for parts in fragments.values():
        parts.sort()
        if [number for number, _ in parts] != list(range(1, len(parts) + 1)):
            parser.warn("JPEG APP11 分片序号不连续，未确认 C2PA。")
            continue
        first = parts[0][1]
        pieces = [first]
        for _, part in parts[1:]:
            pieces.append(part[8:] if part[:8] == first[:8] else part)
        if _is_jumbf(b"".join(pieces)):
            parser.c2pa("jpeg:APP11/JUMBF:c2pa")
    for (_, total), parts in extended.items():
        body = bytearray()
        for start, part in sorted(parts):
            if start != len(body):
                break
            body.extend(part)
        if len(body) == total:
            parser.xmp(bytes(body))
        else:
            parser.warn("扩展 XMP 不完整，已跳过。")


def _riff(parser, read, size):
    header = read(0, 12)
    end = int.from_bytes(header[4:8], "little") + 8
    if end > size or end < 12:
        raise ValueError("RIFF 声明长度越界")
    offset = 12
    while offset + 8 <= end:
        parser.tick()
        header = read(offset, 8)
        if header is None:
            return
        kind, length = struct.unpack("<4sI", header)
        if offset + 8 + length + length % 2 > end:
            raise ValueError("RIFF 块长度越界")
        if kind in {b"XMP ", b"EXIF", b"C2PA"}:
            if kind == b"C2PA" and length:
                parser.c2pa("riff:C2PA")
            else:
                data = parser.payload(read, offset + 8, length)
                if data is not None:
                    if kind == b"XMP ":
                        parser.xmp(data)
                    else:
                        parser.exif(data)
        offset += 8 + length + length % 2


def _bmff(parser, read, size):
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta", b"ilst", b"dinf", b"mvex", b"moof", b"traf"}

    def walk(start, end, depth=0, parent=b"", keys=None):
        offset = start
        keys = {} if keys is None else keys
        while offset + 8 <= end:
            parser.tick(depth)
            header = read(offset, 8)
            if header is None:
                return
            length, kind = struct.unpack(">I4s", header)
            header_size = 8
            if length == 1:
                extended = read(offset + 8, 8)
                if extended is None:
                    return
                length = int.from_bytes(extended, "big")
                header_size = 16
            elif length == 0:
                length = end - offset
            if length < header_size or offset + length > end:
                raise ValueError("ISO-BMFF box 长度越界")
            body_start = offset + header_size
            body_length = length - header_size
            if kind == b"uuid":
                if body_length < 16:
                    raise ValueError("ISO-BMFF uuid 长度不足")
                identifier = read(body_start, 16)
                if identifier == C2PA_UUID:
                    parser.c2pa("iso-bmff:uuid:d8fec3d6-1b0e-483c-9297-5828877ec481")
                elif identifier == XMP_UUID:
                    data = parser.payload(read, body_start + 16, body_length - 16)
                    if data is not None:
                        parser.xmp(data)
            elif kind == b"keys" and parent == b"meta":
                data = parser.payload(read, body_start, body_length)
                if data is not None and len(data) >= 8:
                    entry_offset = 8
                    count = int.from_bytes(data[4:8], "big")
                    for index in range(1, min(count, 256) + 1):
                        if entry_offset + 8 > len(data):
                            break
                        entry_length = int.from_bytes(data[entry_offset:entry_offset + 4], "big")
                        if entry_length < 8 or entry_offset + entry_length > len(data):
                            break
                        keys[index] = _decode(data[entry_offset + 8:entry_offset + entry_length])
                        entry_offset += entry_length
            elif parent == b"ilst":
                data = parser.payload(read, body_start, body_length)
                if data is not None:
                    name = keys.get(int.from_bytes(kind, "big"), _decode(kind))
                    cursor = 0
                    values = []
                    while cursor + 8 <= len(data):
                        child_length, child_kind = struct.unpack(">I4s", data[cursor:cursor + 8])
                        if child_length < 8 or cursor + child_length > len(data):
                            break
                        payload = data[cursor + 8:cursor + child_length]
                        if child_kind == b"name" and len(payload) >= 4:
                            name = _decode(payload[4:])
                        elif child_kind == b"data" and len(payload) >= 8 and payload[:4] in (b"\0\0\0\x01", b"\0\0\0\0"):
                            values.append(payload[8:])
                        cursor += child_length
                    for value in values:
                        parser.metadata("format_tags", name, value, "mp4")
            elif kind in containers:
                skip = 4 if kind == b"meta" else 0
                if body_length >= skip:
                    walk(body_start + skip, offset + length, depth + 1, kind, {} if kind == b"meta" else keys)
            elif kind == b"XMP_":
                data = parser.payload(read, body_start, body_length)
                if data is not None:
                    parser.xmp(data)
            elif parent == b"udta" and kind in {b"AIGC", b"aigc", b"\xa9cmt", b"\xa9swr", b"desc"}:
                data = parser.payload(read, body_start, body_length)
                if data is not None:
                    parser.metadata("format_tags", _decode(kind), data, "mp4")
            offset += length
    walk(0, size)

def _ebml(parser, read, size):
    def vint(offset, identifier=False):
        first = read(offset, 1)
        if first is None:
            return None
        if first[0] == 0:
            raise ValueError("EBML 可变长度整数无效")
        width = 1
        mask = 128
        while not first[0] & mask:
            width += 1
            mask >>= 1
        if width > (4 if identifier else 8):
            raise ValueError("EBML 整数宽度超限")
        data = read(offset, width)
        if data is None:
            return None
        value = int.from_bytes(data, "big")
        if not identifier:
            value &= (1 << (7 * width)) - 1
            if value == (1 << (7 * width)) - 1:
                value = None
        return value, width

    def walk(start, end, depth=0, parent=None, context=None):
        offset = start
        tag_name = None
        tag_values = []
        if parent == 0x7373:
            context = {"values": [], "track_uids": []}
        while offset < end:
            parser.tick(depth)
            identity = vint(offset, True)
            if identity is None:
                break
            kind, id_width = identity
            length_info = vint(offset + id_width)
            if length_info is None:
                break
            length, width = length_info
            body_start = offset + id_width + width
            if length is None:
                if kind != 0x18538067:
                    parser.warn("EBML 未知长度元素限制结构遍历，已保留 ffprobe 结果。")
                    break
                length = end - body_start
            body_end = body_start + length
            if body_end > end:
                raise ValueError("EBML 元素长度越界")
            if kind in {0x1a45dfa3, 0x18538067, 0x1254c367, 0x7373, 0x67c8, 0x63c0}:
                walk(body_start, body_end, depth + 1, kind, context)
            elif kind == 0x63c5 and parent == 0x63c0 and context is not None and length <= 8:
                data = read(body_start, length)
                if data is not None and any(data):
                    context["track_uids"].append(int.from_bytes(data, "big"))
            elif kind in {0x45a3, 0x4487} and parent == 0x67c8:
                data = parser.payload(read, body_start, length)
                if data is not None:
                    if kind == 0x45a3:
                        tag_name = _decode(data)
                    else:
                        tag_values.append(_decode(data))
            elif kind == 0x4282 and parent == 0x1a45dfa3:
                data = parser.payload(read, body_start, length)
                if data == b"webm":
                    parser.result["container"] = "webm"
            offset = body_end
        if tag_name is not None and context is not None:
            context["values"].extend((tag_name, value) for value in tag_values)
        if parent == 0x7373:
            tags = {} if context["track_uids"] else parser.result["metadata"]["format_tags"]
            for name, value in context["values"]:
                parser.metadata("stream_tags" if context["track_uids"] else "format_tags", name, value, "mkv_tag", tags)
            if context["track_uids"] and len(parser.result["metadata"]["stream_tags"]) < 256:
                parser.result["metadata"]["stream_tags"].append({"index": None, "track_uids": context["track_uids"], "tags": tags})
    walk(0, size)


def _probe(parser, path, executable, timeout):
    executable = os.fspath(executable) if executable is not None else shutil.which("ffprobe")
    if not executable:
        parser.warn("未找到 ffprobe，继续使用受限容器解析。")
        return
    command = [executable, "-v", "error", "-protocol_whitelist", "file,pipe",
               "-format_whitelist", "mov,matroska,webm,avi,mpeg,mpegts,asf,flv",
               "-show_entries", "format=format_name,duration,size:format_tags:stream=index,codec_type,codec_name,width,height,duration:stream_tags",
               "-of", "json", "-i", str(path)]
    buffers = [bytearray(), bytearray()]
    exceeded = threading.Event()
    workers = []
    try:
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            def consume(pipe, target, limit):
                try:
                    while True:
                        chunk = pipe.read(65536)
                        if not chunk:
                            break
                        available = limit - len(target)
                        target.extend(chunk[:max(available, 0)])
                        if len(chunk) > available:
                            exceeded.set()
                            try:
                                process.kill()
                            except OSError:
                                pass
                finally:
                    pipe.close()
            for pipe, target, limit in ((process.stdout, buffers[0], MAX_METADATA), (process.stderr, buffers[1], MAX_PAYLOAD)):
                worker = threading.Thread(target=consume, args=(pipe, target, limit), daemon=True)
                worker.start()
                workers.append(worker)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                parser.warn("ffprobe 超时，继续使用受限容器解析。")
                return
            finally:
                for worker in workers:
                    worker.join()
            if exceeded.is_set():
                parser.warn("ffprobe 输出超过上限，已终止并忽略输出。")
                return
            if process.returncode:
                parser.warn("ffprobe 返回失败，继续使用受限容器解析。")
                return
        data = json.loads(buffers[0])
        if not isinstance(data, dict):
            raise ValueError("无效 ffprobe 对象")
        container = parser.result["container"]
        location = "mkv_tag" if container in {"mkv", "webm"} else "mp4" if container in {"mp4", "mov"} else "byte_scan"
        format_info = data.get("format", {})
        for key, value in format_info.get("tags", {}).items():
            parser.metadata("format_tags", key, value, location)
        streams = []
        parser.result["metadata"]["stream_tags"] = []
        for stream in data.get("streams", [])[:256]:
            tags = {}
            for key, value in stream.get("tags", {}).items():
                parser.metadata("stream_tags", key, value, location, tags)
            parser.result["metadata"]["stream_tags"].append({"index": stream.get("index"), "tags": tags})
            if stream.get("codec_type") == "video":
                streams.append({key: stream[key] for key in ("index", "codec_name", "width", "height", "duration") if key in stream})
        parser.result["metadata"]["video"] = {
            "format_name": format_info.get("format_name"), "duration": format_info.get("duration"), "streams": streams,
        }
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.warn(f"ffprobe 无法执行或输出无效（{type(exc).__name__}），继续使用受限容器解析。")


def _identify(header, suffix):
    if header.startswith(b"\xff\xd8"):
        return "image", "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "png"
    if header.startswith(b"RIFF") and header[8:12] in {b"WEBP", b"AVI "}:
        return ("image", "webp") if header[8:12] == b"WEBP" else ("video", "avi")
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "video", "mkv"
    if header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"uuid"}:
        brand = header[8:12]
        if brand in {b"avif", b"avis", b"heic", b"heix", b"mif1"}:
            return "image", "avif" if brand in {b"avif", b"avis"} else "heif"
        return "video", "mov" if brand == b"qt  " or suffix == ".mov" else "mp4"
    if header.startswith((b"II*\0", b"MM\0*")):
        return "image", "tiff"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image", "gif"
    if header.startswith(b"BM"):
        return "image", "bmp"
    if suffix in VIDEO_SUFFIXES:
        return "video", suffix[1:]
    if suffix in IMAGE_SUFFIXES:
        return "image", "unknown"
    return "unknown", "unknown"


def analyze_file(path, *, ffprobe=None, timeout_seconds=20.0, scan_bytes=4 * 1024 * 1024) -> dict:
    """解析一个本地文件；普通输入/文件/子进程异常转换为中文错误或警告。"""
    started = time.perf_counter()
    result = {
        "schema_version": "1.0", "module": "provenance", "status": "success",
        "error_message": None, "input_file": "", "input_kind": "unknown",
        "size_bytes": 0, "sha256": "", "container": "unknown",
        "verdict": "no_label", "verdict_label": "未发现可识别的生成标识或明确工具痕迹",
        "aigc_labels": [], "c2pa": {"present": False, "locations": [], "verified": False, "note": "只做结构检测，未验证签名"},
        "digital_source_type": None, "generator_hints": [],
        "metadata": {"exif": {}, "xmp": {}, "png_text": {}, "format_tags": {}, "stream_tags": [], "video": {}},
        "warnings": [], "limitations": [
            "没有标识不代表不是生成内容：重新编码、截图、转发都会丢失标识。",
            "标识可以被伪造，有标识也要结合其他证据判断。",
        ], "runtime_seconds": 0.0,
    }
    parser = _Parser(result)
    try:
        path = Path(path)
        result["input_file"] = path.name
        if not isinstance(scan_bytes, int) or isinstance(scan_bytes, bool) or not 32 <= scan_bytes <= 64 * 1024 * 1024:
            raise ValueError("scan_bytes 必须为 32 至 67108864 的整数")
        timeout = float(timeout_seconds)
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds 必须是有限正数")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("输入不是普通文件")
        before = path.stat()
        result["size_bytes"] = size = before.st_size
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            result["sha256"] = digest.hexdigest()
            stream.seek(0)
            header = stream.read(min(size, 32))
            kind, container = _identify(header, path.suffix.lower())
            result["input_kind"], result["container"] = kind, container
            if size == 0:
                raise ValueError("输入文件为空")
            windows = []
            if kind == "video":
                stream.seek(0)
                windows.append((0, stream.read(min(scan_bytes, size))))
                tail = max(scan_bytes, size - scan_bytes)
                if tail < size:
                    stream.seek(tail)
                    windows.append((tail, stream.read(size - tail)))
                if len(windows) == 2 and windows[1][0] == len(windows[0][1]):
                    windows = [(0, windows[0][1] + windows[1][1])]
                parser.warn("视频字节解析仅覆盖头尾窗口，窗口外元数据可能遗漏；SHA-256 单独分块读取全文件。")

            def read(offset, length):
                if offset < 0 or length < 0 or offset + length > size:
                    return None
                if kind == "video":
                    for start, data in windows:
                        if start <= offset and offset + length <= start + len(data):
                            return data[offset - start:offset - start + length]
                    parser.warn("容器元素不完整地落在头尾窗口内，已跳过该结构分支。")
                    return None
                stream.seek(offset)
                data = stream.read(length)
                return data if len(data) == length else None

            if container == "jpeg":
                _jpeg(parser, read, size)
            elif container == "png":
                _png(parser, read, size)
            elif container in {"webp", "avi"}:
                _riff(parser, read, size)
            elif container in {"mp4", "mov", "avif", "heif"}:
                _bmff(parser, read, size)
            elif container in {"mkv", "webm"}:
                _ebml(parser, read, size)
            elif container == "tiff":
                data = parser.payload(read, 0, min(size, MAX_PAYLOAD))
                if data is not None:
                    parser.exif(data)
                parser.warn("TIFF 仅尝试受限 EXIF，未实现 TIFF XMP/C2PA。")
            else:
                parser.warn("此容器没有实现专用标识解析；不对任意文件字节作关键词推断。")
        if kind == "video":
            _probe(parser, path, ffprobe, timeout)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("解析期间文件发生变化，哈希与元数据可能不一致")
        labels = {item["label"] for item in result["aigc_labels"]}
        if "1" in labels or any(value in AI_SOURCE_TYPES for value in parser.source_types):
            result["verdict"] = "labelled_ai"
            result["verdict_label"] = "元数据声明属于生成合成内容（未核验声明真实性）"
        elif labels & {"2", "3"}:
            result["verdict"] = "labelled_possible"
            result["verdict_label"] = "元数据声明可能或疑似为生成合成内容"
        elif result["generator_hints"]:
            result["verdict"] = "generator_hint"
            result["verdict_label"] = "发现明确生成工具痕迹，仅作线索"
    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = f"文件标识解析失败（{type(exc).__name__}）：{_clip(exc)}"
        result["verdict_label"] = "解析未完成，不能据此判断是否存在标识"
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    return result
