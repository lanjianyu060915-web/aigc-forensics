"""多人脸查询的主脸选择：默认仍报错，显式开启才选择，而且必须留下记录。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "modules" / "identity_search", ROOT / "modules" / "identity_search_v2"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from identity_search.errors import MultipleFacesError
from scale_fallback import (
    MULTI_FACE_LARGEST,
    annotate_main_face,
    attempt_scale_chain,
    select_main_face,
    write_main_face_query,
)


SMALL = {"bbox": [10.0, 10.0, 30.0, 30.0], "det_score": 0.99}
LARGE = {"bbox": [50.0, 50.0, 150.0, 150.0], "det_score": 0.70}


def _query(tmp_path: Path) -> Path:
    path = tmp_path / "查询 图.png"
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))
    return path


class _Searcher:
    """第一次查询报多人脸，换成涂抹后的图才成功。"""

    def __init__(self, faces: list[dict]) -> None:
        self.faces = faces
        self.queries: list[str] = []
        self.extractor = None

    def search(self, query, top_k, progress_reporter=None, **kwargs):
        self.queries.append(str(query))
        if len(self.queries) == 1:
            raise MultipleFacesError(
                "图片中检测到多张人脸",
                diagnostics={"face_count": len(self.faces), "detected_faces": self.faces},
            )
        return {
            "status": "success",
            "query_image": str(query),
            "method": {},
            "metrics": {"query_face_count": 1},
            "warnings": [],
            "candidates": [],
        }


def test_largest_face_wins_even_with_a_lower_detection_score() -> None:
    assert select_main_face([SMALL, LARGE]) == 1


def test_equal_area_falls_back_to_detection_score_deterministically() -> None:
    first = {"bbox": [0, 0, 10, 10], "det_score": 0.6}
    second = {"bbox": [20, 20, 30, 30], "det_score": 0.9}
    assert select_main_face([first, second]) == 1
    assert select_main_face([second, first]) == 0


def test_faces_without_bbox_cannot_be_selected() -> None:
    with pytest.raises(Exception, match="人脸框"):
        select_main_face([{"det_score": 0.9}])


def test_masking_covers_other_faces_and_keeps_the_main_face(tmp_path: Path) -> None:
    query = _query(tmp_path)
    image = cv2.imdecode(np.fromfile(str(query), dtype=np.uint8), cv2.IMREAD_COLOR)
    image[50:150, 50:150] = (10, 20, 30)
    image[10:30, 10:30] = (250, 0, 0)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(query))

    path = write_main_face_query(query, [SMALL, LARGE], 1, tmp_path / "out")

    written = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert written.shape == image.shape
    assert tuple(written[100, 100]) == (10, 20, 30)
    assert tuple(written[20, 20]) != (250, 0, 0)
    # 原始查询图不能被改动。
    original = cv2.imdecode(np.fromfile(str(query), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert tuple(original[20, 20]) == (250, 0, 0)


def test_strict_policy_still_raises(tmp_path: Path) -> None:
    searcher = _Searcher([SMALL, LARGE])
    with pytest.raises(MultipleFacesError):
        attempt_scale_chain(searcher, str(_query(tmp_path)), (1280,), lambda scale: None, top_k=5)
    assert len(searcher.queries) == 1


def test_policy_without_output_directory_still_raises(tmp_path: Path) -> None:
    searcher = _Searcher([SMALL, LARGE])
    with pytest.raises(MultipleFacesError):
        attempt_scale_chain(
            searcher, str(_query(tmp_path)), (1280,), lambda scale: None, top_k=5,
            multi_face_policy=MULTI_FACE_LARGEST,
        )


def test_largest_face_policy_records_the_selection(tmp_path: Path) -> None:
    query = _query(tmp_path)
    searcher = _Searcher([SMALL, LARGE])

    result, attempts = attempt_scale_chain(
        searcher, str(query), (1280, 640), lambda scale: None, top_k=5,
        multi_face_policy=MULTI_FACE_LARGEST, main_face_dir=tmp_path / "task",
    )

    assert attempts[-1]["status"] == "success"
    assert len(attempts) == 1, "主脸选择成功后不应再回退尺度"
    assert result["query_image"] == str(query), "结果必须仍指向原始查询图"
    selection = result["method"]["face_selection"]
    assert selection["policy"] == MULTI_FACE_LARGEST
    assert selection["face_count"] == 2
    assert selection["selected_bbox"] == LARGE["bbox"]
    assert selection["other_bboxes"] == [SMALL["bbox"]]
    assert result["metrics"]["query_face_count_original"] == 2
    assert any("2 张人脸" in item for item in result["warnings"])
    assert (tmp_path / "task" / "query_main_face_only.png").is_file()
    json.dumps(result, ensure_ascii=False)


def test_annotate_tolerates_missing_blocks(tmp_path: Path) -> None:
    result: dict = {}
    annotate_main_face(result, "q.jpg", [SMALL, LARGE], 1, tmp_path / "m.png")
    assert result["method"]["face_selection"]["selected_index"] == 1
    assert result["warnings"]
