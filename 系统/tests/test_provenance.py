"""全部素材由用例创建，不依赖本机媒体库。"""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import zlib

import pytest
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.provenance import analyze_file
from modules.provenance import parser as implementation
from modules.provenance.run_provenance import main


def label_json(label="1", alias=False):
    payload = {
        "Label": label, "ContentProducer": "001191310110779301025N00000",
        "ProduceID": "generated-001", "ReservedCode1": "",
        "ContentPropagator": "91310110779301025N", "PropagateID": "shared-001",
        "ReservedCode2": "",
    }
    if alias:
        payload["PropagatorID"] = payload.pop("PropagateID")
    return json.dumps(payload, ensure_ascii=False)


def box(kind, data=b"", extended=False):
    if extended:
        return struct.pack(">I4sQ", 1, kind, 16 + len(data)) + data
    return struct.pack(">I4s", 8 + len(data), kind) + data


def png_chunk(kind, data=b""):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def png(*chunks):
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + b"".join(chunks) + png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0")) + png_chunk(b"IEND")


def segment(marker, data):
    return bytes([255, marker]) + struct.pack(">H", len(data) + 2) + data


def jpeg(*segments):
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(stream, "JPEG")
    data = stream.getvalue()
    return data[:2] + b"".join(segments) + data[2:]


def xmp(value=None, source=None, attribute=False, resource=False, namespace=implementation.IPTC_NS):
    content = ""
    attrs = ""
    if value is not None:
        if attribute:
            attrs = ' TC260:AIGC="' + html.escape(value, quote=True) + '"'
        else:
            content += "<TC260:AIGC>" + html.escape(value) + "</TC260:AIGC>"
    if source:
        if resource:
            content += '<iptc:DigitalSourceType rdf:resource="' + source + '"/>'
        else:
            content += "<iptc:DigitalSourceType>" + source + "</iptc:DigitalSourceType>"
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="{implementation.RDF_NS}" '
            f'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/" xmlns:iptc="{namespace}">'
            f'<rdf:RDF><rdf:Description{attrs}>{content}</rdf:Description></rdf:RDF></x:xmpmeta>').encode()


def xmp_jpeg(data):
    return jpeg(segment(0xe1, b"http://ns.adobe.com/xap/1.0/\0" + data))


def riff_chunk(kind, data):
    return kind + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def webp(*chunks):
    data = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(data)) + data


def ebml(kind, payload):
    for width in range(1, 9):
        if len(payload) < (1 << (7 * width)) - 1:
            length = ((1 << (7 * width)) | len(payload)).to_bytes(width, "big")
            break
    return kind + length + payload


def mkv(value, doctype=b"matroska", unknown=False):
    tag = ebml(bytes.fromhex("67c8"), ebml(bytes.fromhex("45a3"), b"AIGC") + ebml(bytes.fromhex("4487"), value.encode()))
    tags = ebml(bytes.fromhex("1254c367"), ebml(bytes.fromhex("7373"), tag))
    head = ebml(bytes.fromhex("1a45dfa3"), ebml(bytes.fromhex("4282"), doctype))
    return head + (bytes.fromhex("18538067ff") + tags if unknown else ebml(bytes.fromhex("18538067"), tags))


def jumbf(label=b"c2pa", identifier=implementation.JUMBF_UUID):
    return box(b"jumb", box(b"jumd", identifier + b"\x03" + label + b"\0"))


def run_bytes(tmp_path, data, suffix, **kwargs):
    path = tmp_path / ("素材" + suffix)
    path.write_bytes(data)
    kwargs.setdefault("ffprobe", "missing-provenance-ffprobe.exe")
    return analyze_file(path, **kwargs)


@pytest.mark.parametrize("label,verdict", [("1", "labelled_ai"), ("2", "labelled_possible"), ("3", "labelled_possible")])
@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt"])
def test_png_labels(tmp_path, label, verdict, kind):
    value = label_json(label).encode()
    prefix = b"AIGC\0"
    if kind == b"zTXt":
        value = b"\0" + zlib.compress(value)
    elif kind == b"iTXt":
        value = b"\1\0\0\0" + zlib.compress(value)
    result = run_bytes(tmp_path, png(png_chunk(kind, prefix + value)), ".png")
    assert result["status"] == "success"
    assert result["verdict"] == verdict
    assert result["aigc_labels"][0]["label_meaning"] == implementation.LABEL_MEANINGS[label]
    assert result["aigc_labels"][0]["credit_codes"] == ["91310110779301025N"]


@pytest.mark.parametrize("container", ["jpeg", "png", "webp"])
@pytest.mark.parametrize("escaping", ["normal", "entities", "backslash", "json_string", "wrapper"])
@pytest.mark.parametrize("attribute", [True, False])
def test_xmp_labels(tmp_path, container, escaping, attribute):
    value = label_json()
    if escaping == "entities":
        value = html.escape(value)
    elif escaping == "backslash":
        value = value.replace('"', '\\"')
    elif escaping == "json_string":
        value = json.dumps(value)
    elif escaping == "wrapper":
        value = json.dumps({"AIGC": json.loads(value)})
    packet = xmp(value, attribute=attribute)
    if container == "jpeg":
        data = xmp_jpeg(packet)
    elif container == "png":
        data = png(png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + packet))
    else:
        data = webp(riff_chunk(b"XMP ", packet))
    result = run_bytes(tmp_path, data, "." + container)
    assert result["status"] == "success"
    assert result["verdict"] == "labelled_ai"
    assert result["aigc_labels"][0]["location"] == "xmp"


@pytest.mark.parametrize("doctype", [b"matroska", b"webm"])
@pytest.mark.parametrize("unknown", [True, False])
def test_mkv_and_alias_without_probe(tmp_path, doctype, unknown):
    result = run_bytes(tmp_path, mkv(label_json("2", alias=True), doctype, unknown), ".mkv")
    assert result["status"] == "success"
    assert result["verdict"] == "labelled_possible"
    assert result["aigc_labels"][0]["propagate_id"] == "shared-001"
    assert result["aigc_labels"][0]["location"] == "mkv_tag"
    assert any("PropagatorID" in warning for warning in result["warnings"])
    assert any("ffprobe" in warning for warning in result["warnings"])


@pytest.mark.parametrize("layout", ["udta", "freeform", "mdta"])
def test_mp4_labels(tmp_path, layout):
    value = label_json().encode()
    if layout == "udta":
        meta = box(b"udta", box(b"AIGC", value))
    elif layout == "freeform":
        meta = box(b"meta", b"\0" * 4 + box(b"ilst", box(b"----", box(b"name", b"\0" * 4 + b"AIGC") + box(b"data", b"\0\0\0\1" + b"\0" * 4 + value))))
    else:
        keys = box(b"keys", b"\0" * 4 + struct.pack(">I", 1) + box(b"mdta", b"AIGC"))
        values = box(b"ilst", box(b"\0\0\0\1", box(b"data", b"\0\0\0\1" + b"\0" * 4 + value)))
        meta = box(b"meta", b"\0" * 4 + keys + values)
    result = run_bytes(tmp_path, box(b"ftyp", b"isom\0\0\0\0") + box(b"moov", meta), ".mp4")
    assert result["status"] == "success"
    assert result["verdict"] == "labelled_ai"
    assert result["aigc_labels"][0]["location"] == "mp4"


@pytest.mark.parametrize("container", ["jpeg", "jpeg_fragments", "png", "webp", "mp4", "mp4_extended"])
def test_c2pa_structure_only(tmp_path, container):
    manifest = jumbf()
    if container == "jpeg":
        data = jpeg(segment(0xeb, b"JP\0\1\0\0\0\1" + manifest))
    elif container == "jpeg_fragments":
        data = jpeg(segment(0xeb, b"JP\0\1\0\0\0\1" + manifest[:24]), segment(0xeb, b"JP\0\1\0\0\0\2" + manifest[:8] + manifest[24:]))
    elif container == "png":
        data = png(png_chunk(b"caBX", manifest))
    elif container == "webp":
        data = webp(riff_chunk(b"C2PA", manifest))
    else:
        data = box(b"ftyp", b"isom\0\0\0\0") + box(b"uuid", implementation.C2PA_UUID + b"\0" * 8 + manifest, container.endswith("extended"))
    result = run_bytes(tmp_path, data, "." + container.split("_")[0])
    assert result["status"] == "success"
    assert result["c2pa"]["present"] is True
    assert result["c2pa"]["verified"] is False
    assert result["verdict"] == "no_label"


@pytest.mark.parametrize("case", ["mdat", "free", "app11_text", "wrong_jumbf", "png_text", "webp_pixels", "mkv_cluster"])
def test_incidental_markers_not_evidence(tmp_path, case):
    fake = b"c2pa" + box(b"uuid", implementation.C2PA_UUID) + b"AIGC\0" + label_json().encode()
    if case in {"mdat", "free"}:
        data, suffix = box(b"ftyp", b"isom\0\0\0\0") + box(case.encode(), fake), ".mp4"
    elif case == "app11_text":
        data, suffix = jpeg(segment(0xeb, fake)), ".jpg"
    elif case == "wrong_jumbf":
        data, suffix = jpeg(segment(0xeb, b"JP\0\1\0\0\0\1" + jumbf(identifier=b"\0" * 16))), ".jpg"
    elif case == "png_text":
        data, suffix = png(png_chunk(b"tEXt", b"Comment\0" + fake)), ".png"
    elif case == "webp_pixels":
        data, suffix = webp(riff_chunk(b"VP8 ", fake)), ".webp"
    else:
        data = ebml(bytes.fromhex("1a45dfa3"), b"") + ebml(bytes.fromhex("18538067"), ebml(bytes.fromhex("1f43b675"), mkv(label_json())))
        suffix = ".mkv"
    result = run_bytes(tmp_path, data, suffix)
    assert result["status"] == "success"
    assert result["c2pa"]["present"] is False
    assert result["verdict"] == "no_label"


@pytest.mark.parametrize("source,expected", [
    ("trainedAlgorithmicMedia", "labelled_ai"),
    ("compositeWithTrainedAlgorithmicMedia", "labelled_ai"),
    ("digitalCapture", "no_label"), ("algorithmicallyEnhanced", "no_label"),
    ("digitalArt", "no_label"), ("algorithmicMedia", "no_label"),
    ("compositeSynthetic", "no_label"),
])
@pytest.mark.parametrize("resource", [False, True])
def test_digital_source_type(tmp_path, source, expected, resource):
    uri = "http://cv.iptc.org/newscodes/digitalsourcetype/" + source
    result = run_bytes(tmp_path, xmp_jpeg(xmp(source=uri, resource=resource)), ".jpg")
    assert result["digital_source_type"] == uri
    assert result["verdict"] == expected


def test_wrong_namespace_or_uri(tmp_path):
    for namespace, uri in [("urn:not-iptc", next(iter(implementation.AI_SOURCE_TYPES))), (implementation.IPTC_NS, "https://example.com/trainedAlgorithmicMedia")]:
        result = run_bytes(tmp_path, xmp_jpeg(xmp(source=uri, namespace=namespace)), ".jpg")
        assert result["verdict"] == "no_label"


@pytest.mark.parametrize("key,value,expected", [
    ("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 42", "generator_hint"),
    ("parameters", "camera parameters: aperture 2.8", "no_label"),
    ("prompt", '{"4":{"class_type":"KSampler","inputs":{"seed":42}}}', "generator_hint"),
    ("workflow", '{"nodes":[{"id":1,"type":"KSampler"}],"links":[]}', "generator_hint"),
    ("prompt", "a cat on a desk", "no_label"),
    ("workflow", '{"instructions":"review photo"}', "no_label"),
    ("Software", "ComfyUI", "generator_hint"),
    ("Software", "Adobe Photoshop", "no_label"),
    ("Comment", "Generated with Stable Diffusion", "generator_hint"),
    ("Description", "Article about Stable Diffusion", "no_label"),
])
def test_generator_hints(tmp_path, key, value, expected):
    result = run_bytes(tmp_path, png(png_chunk(b"tEXt", key.encode() + b"\0" + value.encode())), ".png")
    assert result["verdict"] == expected
    assert result["metadata"]["png_text"][key] == value


def test_exif_and_utf8_truncation(tmp_path):
    exif = Image.Exif()
    exif[271] = "Canon"
    exif[272] = "EOS 5D"
    exif[305] = "Photoshop"
    exif[270] = "Camera portrait"
    result = run_bytes(tmp_path, jpeg(segment(0xe1, exif.tobytes())), ".jpg")
    assert result["verdict"] == "no_label"
    assert result["metadata"]["exif"]["Model"] == "EOS 5D"
    value = "ComfyUI " + "中文" * 2000
    result = run_bytes(tmp_path, png(png_chunk(b"iTXt", b"Software\0\0\0\0\0" + value.encode())), ".png")
    assert result["verdict"] == "generator_hint"
    assert len(result["generator_hints"][0]["value"].encode()) <= 2048

@pytest.mark.parametrize("value", [
    '{"Label":"1"}',
    label_json("4"),
    label_json().replace('"Label": "1"', '"Label": 1'),
    label_json().replace('"Label": "1"', '"Label": "1", "Label": "2"'),
    label_json()[:-1],
    'prefix ' + label_json(),
])
def test_reject_invalid_aigc(tmp_path, value):
    result = run_bytes(tmp_path, png(png_chunk(b"tEXt", b"AIGC\0" + value.encode())), ".png")
    assert result["verdict"] == "no_label"
    assert result["aigc_labels"] == []


def test_require_named_field_or_wrapper(tmp_path):
    for value, verdict in [(label_json(), "no_label"), (json.dumps({"AIGC": json.loads(label_json())}), "labelled_ai")]:
        result = run_bytes(tmp_path, png(png_chunk(b"tEXt", b"Comment\0" + value.encode())), ".png")
        assert result["verdict"] == verdict


def test_precedence_deduplication(tmp_path):
    result = run_bytes(tmp_path, png(
        png_chunk(b"tEXt", b"AIGC\0" + label_json("2").encode()),
        png_chunk(b"tEXt", b"AIGC\0" + label_json("2").encode()),
        png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp(label_json("1"))),
        png_chunk(b"tEXt", b"Software\0ComfyUI"),
    ), ".png")
    assert result["verdict"] == "labelled_ai"
    assert len(result["aigc_labels"]) == 2


def test_extended_xmp(tmp_path):
    packet = xmp(label_json())
    prefix = b"http://ns.adobe.com/xmp/extension/\0" + b"A" * 32
    middle = len(packet) // 2
    data = jpeg(segment(0xe1, prefix + struct.pack(">II", len(packet), middle) + packet[middle:]),
                segment(0xe1, prefix + struct.pack(">II", len(packet), 0) + packet[:middle]))
    assert run_bytes(tmp_path, data, ".jpg")["verdict"] == "labelled_ai"


def test_bad_crc_and_compression_bomb(tmp_path):
    bad = png_chunk(b"tEXt", b"AIGC\0" + label_json().encode())
    bad = bad[:-4] + b"\0" * 4
    result = run_bytes(tmp_path, png(bad), ".png")
    assert result["verdict"] == "no_label"
    assert any("CRC" in warning for warning in result["warnings"])
    bomb = b"Software\0\0" + zlib.compress(b"ComfyUI " + b"x" * (implementation.MAX_PAYLOAD + 1))
    result = run_bytes(tmp_path, png(png_chunk(b"zTXt", bomb)), ".png")
    assert result["verdict"] == "no_label"
    assert any("解压" in warning for warning in result["warnings"])


def test_payload_limit_and_xml_entities(tmp_path):
    value = b"Software\0ComfyUI" + b"x" * implementation.MAX_PAYLOAD
    result = run_bytes(tmp_path, png(png_chunk(b"tEXt", value)), ".png")
    assert result["verdict"] == "no_label"
    assert any("上限" in warning for warning in result["warnings"])
    packet = b'<!DOCTYPE x [<!ENTITY foo "ComfyUI">]><x>&foo;</x>'
    result = run_bytes(tmp_path, xmp_jpeg(packet), ".jpg")
    assert result["verdict"] == "no_label"
    assert any("DTD" in warning for warning in result["warnings"])


def test_video_window_and_full_hash(tmp_path):
    head = box(b"ftyp", b"isom\0\0\0\0")
    label_box = box(b"moov", box(b"udta", box(b"AIGC", label_json().encode())))
    middle = box(b"free", b"x" * 8192)
    data = head + middle + label_box
    result = run_bytes(tmp_path, data, ".mp4", scan_bytes=1024)
    assert result["verdict"] == "labelled_ai"
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    result = run_bytes(tmp_path, head + middle + label_box + middle, ".mp4", scan_bytes=1024)
    assert result["verdict"] == "no_label"
    assert any("窗口" in warning for warning in result["warnings"])


def test_image_reads_skip_pixels(tmp_path, monkeypatch):
    pixel_size = 5 * 1024 * 1024
    data = png(png_chunk(b"IDAT", b"x" * pixel_size), png_chunk(b"tEXt", b"AIGC\0" + label_json().encode()))
    path = tmp_path / "bounded.png"
    path.write_bytes(data)
    real_open = Path.open
    reads = []

    class Watched:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, length=-1):
            reads.append((self.stream.tell(), length))
            return self.stream.read(length)
        def seek(self, *args):
            return self.stream.seek(*args)

    def opened(target, *args, **kwargs):
        stream = real_open(target, *args, **kwargs)
        return Watched(stream) if target == path else stream
    monkeypatch.setattr(Path, "open", opened)
    result = analyze_file(path)
    assert result["verdict"] == "labelled_ai"
    assert all(0 <= length <= implementation.MAX_PAYLOAD for _, length in reads)
    hash_reads = (len(data) + 1024 * 1024 - 1) // (1024 * 1024) + 1
    assert sum(length for _, length in reads[hash_reads:]) < 4096


@pytest.mark.parametrize("bad", [None, 123, [], {}, "\0", "missing-file"])
def test_any_bad_path_returns_error(bad):
    result = analyze_file(bad)
    assert result["status"] == "error"
    assert result["error_message"].startswith("文件标识解析失败")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"scan_bytes": 0}, {"scan_bytes": -1}, {"scan_bytes": "4"}, {"scan_bytes": True}, {"timeout_seconds": 0}, {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")}, {"timeout_seconds": None}])
def test_bad_options(tmp_path, kwargs):
    assert run_bytes(tmp_path, png(), ".png", **kwargs)["status"] == "error"


@pytest.mark.parametrize("data,suffix", [(b"", ".png"), (b"\x89PNG\r\n\x1a\n\xff\xff\xff\xfftEXt", ".png"), (b"\xff\xd8\xff\xe1\x00\xff", ".jpg"), (box(b"ftyp", b"isom") + b"\xff\xff\xff\xffuuid", ".mp4"), (b"\x1aE\xdf\xa3\0", ".mkv")])
def test_truncated_files(tmp_path, data, suffix):
    result = run_bytes(tmp_path, data, suffix)
    assert result["status"] == "error"
    assert result["error_message"]


def test_unknown_and_deterministic_fuzz(tmp_path):
    generator = random.Random(1729)
    for index in range(100):
        data = generator.randbytes(index + 1)
        result = run_bytes(tmp_path, data, ".bin")
        assert result["verdict"] == "no_label"
        assert result["c2pa"]["present"] is False
        json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_cli_error_result_is_successful_write(tmp_path):
    output = tmp_path / "nested" / "error.json"
    assert main(["--input", str(tmp_path / "missing"), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "error"
    source = tmp_path / "source.png"
    source.write_bytes(png())
    before = source.read_bytes()
    assert main(["--input", str(source), "--output", str(source)]) == 1
    assert source.read_bytes() == before
    assert main(["--input", str(source), "--output", str(tmp_path)]) == 1


def test_cli_subprocess(tmp_path):
    source = tmp_path / "输入.png"
    output = tmp_path / "结果.json"
    source.write_bytes(png())
    script = Path(implementation.__file__).with_name("run_provenance.py")
    process = subprocess.run([sys.executable, "-B", str(script), "--input", str(source), "--output", str(output)], capture_output=True, timeout=20)
    assert process.returncode == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["input_file"] == source.name
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_missing_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(implementation.shutil, "which", lambda _: None)
    result = run_bytes(tmp_path, box(b"ftyp", b"isom\0\0\0\0"), ".mp4", ffprobe=None)
    assert result["status"] == "success"
    assert any("未找到 ffprobe" in warning for warning in result["warnings"])


@pytest.mark.parametrize("mode", ["timeout", "invalid", "overflow", "success", "failure"])
def test_probe_contract(tmp_path, monkeypatch, mode):
    real_popen = subprocess.Popen
    seen = []
    payload = {"format": {"format_name": "mov,mp4", "duration": "2.0", "tags": {"AIGC": label_json("2"), "encoder": "Lavf"}}, "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264", "width": 128, "height": 72, "tags": {"AIGC": label_json("2"), "handler_name": "VideoHandler"}}]}
    scripts = {
        "timeout": "import time;time.sleep(10)",
        "invalid": "print('invalid-json')",
        "overflow": "print('x'*100000)",
        "success": "print(" + repr(json.dumps(payload)) + ")",
        "failure": "raise SystemExit(3)",
    }
    if mode == "overflow":
        monkeypatch.setattr(implementation, "MAX_METADATA", 1024)
    def launch(command, **kwargs):
        seen.append(command)
        return real_popen([sys.executable, "-B", "-c", scripts[mode]], **kwargs)
    monkeypatch.setattr(implementation.subprocess, "Popen", launch)
    result = run_bytes(tmp_path, box(b"ftyp", b"isom\0\0\0\0"), ".mp4", ffprobe="custom-probe", timeout_seconds=0.1 if mode == "timeout" else 5)
    assert seen[0][seen[0].index("-protocol_whitelist") + 1] == "file,pipe"
    assert "-i" in seen[0]
    assert result["status"] == "success"
    if mode == "success":
        assert result["verdict"] == "labelled_possible"
        assert len(result["aigc_labels"]) == 1
        assert result["metadata"]["video"]["streams"][0]["width"] == 128
        assert result["metadata"]["stream_tags"][0]["tags"]["handler_name"] == "VideoHandler"
    else:
        assert result["verdict"] == "no_label"
        assert any("ffprobe" in warning for warning in result["warnings"])


def test_real_ffprobe_generated_avi(tmp_path):
    executable = os.environ.get("FFPROBE") or shutil.which("ffprobe")
    if not executable:
        pytest.skip("未找到 ffprobe，可通过 FFPROBE 环境变量指定")
    frame = jpeg()
    avih = struct.pack("<14I", 1000000, 0, 0, 0, 1, 0, 1, len(frame), 2, 2, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0, 0, 1, 1, 0, 1, len(frame), 0xffffffff, 0, 0, 0, 2, 2)
    strf = struct.pack("<IiiHHIIiiII", 40, 2, 2, 1, 24, int.from_bytes(b"MJPG", "little"), len(frame), 0, 0, 0, 0)
    hdrl = riff_chunk(b"LIST", b"hdrl" + riff_chunk(b"avih", avih) + riff_chunk(b"LIST", b"strl" + riff_chunk(b"strh", strh) + riff_chunk(b"strf", strf)))
    body = b"AVI " + hdrl + riff_chunk(b"LIST", b"movi" + riff_chunk(b"00dc", frame))
    data = b"RIFF" + struct.pack("<I", len(body)) + body
    result = run_bytes(tmp_path, data, ".avi", ffprobe=executable)
    assert result["status"] == "success"
    assert result["metadata"]["video"]["streams"][0]["codec_name"] == "mjpeg"
    assert result["metadata"]["video"]["streams"][0]["width"] == 2
    assert result["verdict"] == "no_label"

@pytest.mark.parametrize("value,expected", [
    ("001191110108MACC4D63XF10306", ["91110108MACC4D63XF"]),
    ("00119144030008867405X210001", ["9144030008867405X2"]),
    ("91310110779301025N", ["91310110779301025N"]),
    ("91310110779301025X", []),
    ("prefix91310110779301025Nsuffix", ["91310110779301025N"]),
    ("002111010519491231002X10832", []),
    ("001191350100M000100Y4300000", ["91350100M000100Y43"]),
])
def test_credit_codes_no_sliding_ambiguity(value, expected):
    assert implementation._credit_codes(value) == expected


def test_mkv_tag_scope(tmp_path):
    def simple(name, value):
        return ebml(bytes.fromhex("67c8"), ebml(bytes.fromhex("45a3"), name) + ebml(bytes.fromhex("4487"), value))
    global_tag = ebml(bytes.fromhex("7373"), simple(b"ENCODER", b"global-encoder"))
    targets = ebml(bytes.fromhex("63c0"), ebml(bytes.fromhex("63c5"), b"\x01"))
    stream_tag = ebml(bytes.fromhex("7373"), targets + simple(b"ENCODER", b"stream-encoder"))
    data = ebml(bytes.fromhex("1a45dfa3"), b"") + ebml(bytes.fromhex("18538067"), ebml(bytes.fromhex("1254c367"), global_tag + stream_tag))
    result = run_bytes(tmp_path, data, ".mkv")
    assert result["metadata"]["format_tags"]["ENCODER"] == "global-encoder"
    assert result["metadata"]["stream_tags"][0]["tags"]["ENCODER"] == "stream-encoder"
    assert result["metadata"]["stream_tags"][0]["track_uids"] == [1]


def test_large_c2pa_chunk_detection_without_payload(tmp_path):
    result = run_bytes(tmp_path, png(png_chunk(b"caBX", b"x" * (implementation.MAX_PAYLOAD + 1))), ".png")
    assert result["c2pa"]["present"] is True
    assert result["verdict"] == "no_label"


def test_utf16_xml_entities_not_expanded(tmp_path):
    packet = '<!DOCTYPE x [<!ENTITY foo "ComfyUI">]><x>&foo;</x>'.encode("utf-16")
    result = run_bytes(tmp_path, xmp_jpeg(packet), ".jpg")
    assert result["verdict"] == "no_label"
    assert result["metadata"]["xmp"] == {}


def test_empty_jpeg_is_error(tmp_path):
    assert run_bytes(tmp_path, b"\xff\xd8", ".jpg")["status"] == "error"


def test_missing_jpeg_fragment_not_c2pa(tmp_path):
    result = run_bytes(tmp_path, jpeg(segment(0xeb, b"JP\0\1\0\0\0\2" + jumbf())), ".jpg")
    assert result["c2pa"]["present"] is False


def test_wrong_uuid_and_truncated_uuid(tmp_path):
    result = run_bytes(tmp_path, box(b"uuid", b"x" * 16 + b"c2pa"), ".mp4")
    assert result["c2pa"]["present"] is False
    result = run_bytes(tmp_path, box(b"uuid", implementation.C2PA_UUID[:10]), ".mp4")
    assert result["status"] == "error"
    assert result["c2pa"]["present"] is False


def test_multiple_source_types(tmp_path):
    uri = "http://cv.iptc.org/newscodes/digitalsourcetype/"
    packet = xmp(source=uri + "trainedAlgorithmicMedia").replace(b"</rdf:Description>", b"<iptc:DigitalSourceType>" + (uri + "digitalCapture").encode() + b"</iptc:DigitalSourceType></rdf:Description>")
    result = run_bytes(tmp_path, xmp_jpeg(packet), ".jpg")
    assert result["verdict"] == "labelled_ai"
    assert len(result["metadata"]["xmp"]["DigitalSourceType_values"]) == 2
    assert result["digital_source_type"] == uri + "trainedAlgorithmicMedia"


def test_readonly_file_and_directory(tmp_path):
    path = tmp_path / "readonly.png"
    path.write_bytes(png())
    path.chmod(0o444)
    try:
        before = path.stat()
        result = analyze_file(path)
        after = path.stat()
        assert result["status"] == "success"
        assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    finally:
        path.chmod(0o666)
    assert analyze_file(tmp_path)["status"] == "error"


def test_file_changed_during_probe(tmp_path, monkeypatch):
    path = tmp_path / "changing.mp4"
    path.write_bytes(box(b"ftyp", b"isom\0\0\0\0"))
    def changed(*args):
        with path.open("ab") as stream:
            stream.write(b"x")
    monkeypatch.setattr(implementation, "_probe", changed)
    result = analyze_file(path)
    assert result["status"] == "error"
    assert "发生变化" in result["error_message"]


def test_cumulative_decompression_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(implementation, "MAX_METADATA", 1024)
    text = b"x" * 800
    data = png(png_chunk(b"zTXt", b"Comment\0\0" + zlib.compress(text)),
               png_chunk(b"zTXt", b"Software\0\0" + zlib.compress(b"ComfyUI " + text)))
    result = run_bytes(tmp_path, data, ".png")
    assert result["verdict"] == "no_label"
    assert any("解压" in warning for warning in result["warnings"])


def test_nesting_limit(tmp_path):
    data = box(b"uuid", implementation.C2PA_UUID)
    for _ in range(30):
        data = box(b"moov", data)
    result = run_bytes(tmp_path, data, ".mp4")
    assert result["status"] == "error"
    assert "深度" in result["error_message"]

@pytest.mark.parametrize("key,value", [
    ("prompt", '{"4":{"class_type":"Task","inputs":{}}}'),
    ("workflow", '{"nodes":[{"id":1,"type":"SendEmail"}],"links":[]}'),
    ("Comment", "Not generated with Stable Diffusion"),
    ("Description", "This was never created using Midjourney"),
])
def test_non_generative_workflow_and_negation(tmp_path, key, value):
    result = run_bytes(tmp_path, png(png_chunk(b"tEXt", key.encode() + b"\0" + value.encode())), ".png")
    assert result["verdict"] == "no_label"


@pytest.mark.parametrize("container", ["png", "webp", "jpeg"])
def test_exif_generating_software(tmp_path, container):
    exif = Image.Exif()
    exif[305] = "AUTOMATIC1111"
    exif[37510] = b"ASCII\0\0\0Generated with Stable Diffusion"
    if container == "png":
        data = png(png_chunk(b"eXIf", exif.tobytes()[6:]))
    elif container == "webp":
        data = webp(riff_chunk(b"EXIF", exif.tobytes()))
    else:
        data = jpeg(segment(0xe1, exif.tobytes()))
    result = run_bytes(tmp_path, data, "." + container)
    assert result["verdict"] == "generator_hint"
    assert result["metadata"]["exif"]["Software"] == "AUTOMATIC1111"
    assert any(hint["source"] == "exif:UserComment" for hint in result["generator_hints"])


def test_cli_hardlink_cannot_overwrite_source(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(png())
    output = tmp_path / "alias.png"
    try:
        os.link(source, output)
    except OSError:
        pytest.skip("临时文件系统不支持硬链接")
    before = source.read_bytes()
    assert main(["--input", str(source), "--output", str(output)]) == 1
    assert source.read_bytes() == before


def test_no_ffprobe_for_image(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("图片不得启动 ffprobe")
    monkeypatch.setattr(implementation.subprocess, "Popen", forbidden)
    result = run_bytes(tmp_path, png(), ".png")
    assert result["status"] == "success"


def test_zero_length_and_large_bmff_headers(tmp_path):
    data = box(b"ftyp", b"isom\0\0\0\0") + struct.pack(">I4s", 0, b"uuid") + implementation.C2PA_UUID
    result = run_bytes(tmp_path, data, ".mp4")
    assert result["c2pa"]["present"] is True
    result = run_bytes(tmp_path, struct.pack(">I4sQ", 1, b"uuid", 2**63) + implementation.C2PA_UUID, ".mp4")
    assert result["status"] == "error"
    assert result["c2pa"]["present"] is False
