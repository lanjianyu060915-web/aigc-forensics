"""GenD 包完整性、策略对应校准和未校准边界，不加载真实大模型。"""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "modules/general_aigc"))
from aigc_engine import gend_face
from general_aigc_core import enrich_face_branch, face_branch_payload


def make_package(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test-weight")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(gend_face, "CHECKPOINT_SHA256", digest)
    manifest = {"checkpoint": checkpoint.name, "checkpoint_sha256": digest, "files_sha256": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return checkpoint, manifest


def test_modified_checkpoint_is_rejected_before_loading(tmp_path, monkeypatch):
    checkpoint, manifest = make_package(tmp_path, monkeypatch)
    assert gend_face.verify_package(tmp_path) == manifest
    checkpoint.write_bytes(b"changed-weight")
    with pytest.raises(ValueError, match="SHA-256"):
        gend_face.verify_package(tmp_path)


def test_manifest_cannot_change_the_pinned_weight(tmp_path, monkeypatch):
    _, manifest = make_package(tmp_path, monkeypatch)
    manifest["checkpoint_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="固定版本"):
        gend_face.verify_package(tmp_path)


def test_configuration_path_cannot_escape_package(tmp_path, monkeypatch):
    _, manifest = make_package(tmp_path, monkeypatch)
    manifest["files_sha256"] = {"../outside.json": "0" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="超出"):
        gend_face.verify_package(tmp_path)


def records():
    return [{"file":f"face_{i}.jpg","source":"image.jpg","face_id":i,
             "short_edge_px":100,"legacy_selected":i==1} for i in (1,2)]


def test_gend_without_calibration_emits_score_but_no_band():
    original = face_branch_payload([0.5], ["face_1.jpg"], {})
    result = enrich_face_branch(original, records(), [0.5,0.95], {}, backend="gend")
    assert result["status"] == "success"
    assert result["score"] == 0.5
    assert result["band"] is None
    assert result["band_thresholds"] is None
    assert result["per_face"][0]["score_kind"] == "model_probability"


def test_gend_calibration_matches_the_selected_aggregation():
    calibration = {"aggregations":{"largest":{"low_max":0.2,"medium_max":0.8},
                                    "max_all":{"low_max":0.96,"medium_max":0.99}}}
    original = face_branch_payload([0.5], ["face_1.jpg"], {})
    largest = enrich_face_branch(original, records(), [0.5,0.95], calibration, backend="gend")
    all_faces = enrich_face_branch(original, records(), [0.5,0.95], calibration,
                                  backend="gend", band_aggregation="max_all")
    assert largest["band"] == "medium"
    assert all_faces["score"] == 0.95
    assert all_faces["band"] == "low"
    assert all_faces["band_thresholds"]["low_max"] == 0.96


def test_gend_does_not_inherit_the_old_small_face_exception():
    original = face_branch_payload([0.99], ["face_1.jpg"], {})
    result = enrich_face_branch(original, [{**records()[0],"short_edge_px":50}], [0.99], {}, backend="gend")
    assert result["status"] == "no_eligible_face"
    assert result["score"] is None
    assert not result["per_face"][0]["used_for_band"]
