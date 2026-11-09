"""只在独立子进程加载随安装包分发的离线模型。"""

import contextlib
import json
from pathlib import Path
import socket
import sys
import time

from core import convert_hit, deduplicate, load_keywords, restore_exif_coordinates, scan_views


def _deny_network(*args, **kwargs):
    raise RuntimeError("画面文字识别禁止联网")


def run(inputs: list[dict]) -> dict:
    socket.socket.connect = _deny_network
    socket.create_connection = _deny_network
    import numpy as np
    from PIL import Image, ImageOps
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR

    models = Path(rapidocr_onnxruntime.__file__).parent / "models"
    names = {"det": "ch_PP-OCRv4_det_infer.onnx", "rec": "ch_PP-OCRv4_rec_infer.onnx",
             "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx"}
    if not all((models / name).is_file() for name in names.values()):
        raise FileNotFoundError("本地文字识别模型不完整")
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                     det_limit_side_len=736, det_limit_type="max",
                     **{f"{kind}_model_path": str(models / name) for kind, name in names.items()})
    keywords = load_keywords()
    hits, frames = [], []
    for frame_index, item in enumerate(inputs):
        started = time.perf_counter()
        with Image.open(item["path"]) as source:
            raw_width, raw_height = source.size
            orientation = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        for view, x, y, sx, sy in scan_views(image):
            rows, _ = engine(np.asarray(view)[:, :, ::-1])
            for row in rows or []:
                hit = convert_hit(row, width=width, height=height, scale_x=sx, scale_y=sy,
                                  offset_x=x, offset_y=y, frame_index=frame_index,
                                  timestamp_seconds=item.get("timestamp_seconds"), keywords=keywords)
                if hit:
                    hits.append(restore_exif_coordinates(hit, raw_width, raw_height, orientation))
        frames.append({"frame_index": frame_index, "timestamp_seconds": item.get("timestamp_seconds"),
                       "width": raw_width, "height": raw_height, "exif_orientation": orientation,
                       "runtime_seconds": round(time.perf_counter() - started, 6)})
    hits = deduplicate(hits)
    kinds = {hit["kind"] for hit in hits}
    verdict = "visible_label_found" if "visible_label_found" in kinds else "tool_hint_found" if kinds else "no_label"
    return {"status": "success", "verdict": verdict, "hits": hits, "frames": frames}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request["inputs"])
    except Exception as error:
        result = {"status": "error", "verdict": "unavailable", "hits": [],
                  "error_message": f"离线文字识别未完成（{type(error).__name__}）。"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
