"""Task-local visual evidence for a completed identity-search inference.

The functions in this module are deliberately a side channel: they consume the
already selected query face and the already ranked best reference, but never
participate in embedding normalization, cosine scoring, person aggregation, or
ranking.  A visualization failure is returned as a warning and cannot change a
genuine search result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .extractor import face_diagnostics, read_image_unicode


ALIGNED_FACE_SIZE = 112


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return cleaned[:80] or "identity_task"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_image(path: Path, image) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
    parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 94] if extension != ".png" else []
    ok, encoded = cv2.imencode(extension, np.asarray(image), parameters)
    if not ok:
        raise OSError(f"OpenCV 无法编码证据图像: {path.name}")
    name_digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:8]
    temporary = path.parent / f".tmp-{os.getpid()}-{name_digest}"
    try:
        # NumPy ``tofile(str_path)`` still routes through a narrow C filename on
        # some Windows builds.  A pathlib-opened binary stream is Unicode-safe.
        with temporary.open("wb") as stream:
            stream.write(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _artifact(
    task_root: Path,
    path: Path,
    *,
    artifact_id: str,
    kind: str,
    description: str,
) -> dict[str, Any]:
    relative_path = path.resolve(strict=False).relative_to(
        task_root.resolve(strict=False)
    ).as_posix()
    return {
        "id": artifact_id,
        "kind": kind,
        "path": relative_path,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "status": "success",
        "description": description,
    }


def _bbox_crop(image, diagnostics: Mapping[str, Any]):
    import numpy as np

    bbox = diagnostics.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("人脸对象没有可用的四点检测框")
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(np.floor(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(np.floor(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(np.ceil(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(np.ceil(float(bbox[3])))))
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("人脸检测框裁剪结果为空")
    return crop, [x1, y1, x2, y2]


def _annotate_face(image, diagnostics: Mapping[str, Any]):
    import cv2

    annotated = image.copy()
    bbox = diagnostics.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        thickness = max(2, round(min(image.shape[:2]) / 350))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), thickness)
        cv2.putText(
            annotated,
            "FACE 1",
            (max(0, x1), max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.65, min(image.shape[:2]) / 1400),
            (255, 180, 0),
            thickness,
            cv2.LINE_AA,
        )
    keypoints = diagnostics.get("keypoints_5")
    colors = ((50, 220, 50), (50, 220, 50), (0, 220, 255), (255, 80, 80), (255, 80, 80))
    if isinstance(keypoints, list):
        radius = max(3, round(min(image.shape[:2]) / 300))
        for index, point in enumerate(keypoints[:5]):
            if not isinstance(point, list) or len(point) != 2:
                continue
            x, y = (int(round(float(value))) for value in point)
            cv2.circle(annotated, (x, y), radius, colors[index], -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                str(index + 1),
                (x + radius + 2, y - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[index],
                1,
                cv2.LINE_AA,
            )
    return annotated


def _align_face(image, diagnostics: Mapping[str, Any]):
    import numpy as np
    from insightface.utils import face_align

    keypoints = diagnostics.get("keypoints_5")
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ValueError("人脸对象没有可用的五点关键点，无法生成标准对齐图")
    return face_align.norm_crop(
        image,
        landmark=points,
        image_size=ALIGNED_FACE_SIZE,
    )


def _square_panel(image, size: int = 320):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    panel = np.full((size, size, 3), 244, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _thumbnail(image, max_side: int = 768):
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _comparison_image(query_aligned, reference_aligned, candidate: Mapping[str, Any]):
    import cv2
    import numpy as np

    query_panel = _square_panel(query_aligned)
    reference_panel = _square_panel(reference_aligned)
    header = np.full((58, 650, 3), 28, dtype=np.uint8)
    body = np.full((320, 650, 3), 236, dtype=np.uint8)
    body[:, :320] = query_panel
    body[:, 330:650] = reference_panel
    cv2.putText(header, "QUERY ALIGNED", (22, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(header, "TOP-1 BEST REFERENCE", (350, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    score = candidate.get("best_reference_score")
    mean_score = candidate.get("mean_similarity", candidate.get("score"))
    caption = f"cosine={float(score):.6f}  person-mean={float(mean_score):.6f}"
    cv2.putText(header, caption, (22, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 210, 255), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def build_identity_visual_evidence(
    *,
    query_path: Path,
    query_face: Any,
    query_feature: Any,
    candidates: list[Mapping[str, Any]],
    extractor: Any,
    task_root: Path,
    task_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Generate task-relative evidence and its integrity manifest.

    The returned mapping contains only relative POSIX paths.  Any optional
    artifact that cannot be created is represented by a warning rather than an
    exception, preserving the completed identity ranking.
    """

    import numpy as np

    task_root = Path(task_root).expanduser().resolve(strict=False)
    query_path = Path(query_path).expanduser().resolve(strict=False)
    query_sha256 = _sha256_file(query_path)
    namespace = _safe_component(task_id or task_root.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    evidence_dir = (
        task_root
        / "visualizations"
        / "identity"
        # The enclosing task root already binds the artifact to ``namespace``.
        # Keeping the run directory compact preserves Windows MAX_PATH
        # compatibility even when both the project and task names are Chinese.
        / f"{run_id}_{query_sha256[:12]}"
    )
    evidence_dir.mkdir(parents=True, exist_ok=False)

    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    visual: dict[str, Any] = {
        "status": "partial",
        "query_sha256": query_sha256,
        "query_face_count": 1,
        "embedding_status": "completed",
        "embedding_dimension": int(np.asarray(query_feature).reshape(-1).shape[0]),
        "embedding_normalization": "l2",
        "face_alignment_method": "insightface.utils.face_align.norm_crop",
        "aligned_face_size": [ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE],
        "artifact_directory": evidence_dir.relative_to(task_root).as_posix(),
        "task_namespace": namespace,
    }

    query_image = read_image_unicode(query_path)
    query_diagnostics = face_diagnostics(query_face)
    visual["query_face"] = query_diagnostics
    query_aligned = None
    try:
        annotated_path = evidence_dir / "query_face_detection.jpg"
        _write_image(annotated_path, _annotate_face(query_image, query_diagnostics))
        visual["face_detection_image"] = annotated_path.relative_to(task_root).as_posix()
        artifacts.append(
            _artifact(
                task_root,
                annotated_path,
                artifact_id="identity.query.face_detection",
                kind="identity_face_detection",
                description="查询原图上的真实人脸检测框与五点关键点。",
            )
        )
    except Exception as exc:
        warnings.append(f"查询人脸标注图生成失败: {type(exc).__name__}: {exc}")

    try:
        crop, crop_bbox = _bbox_crop(query_image, query_diagnostics)
        crop_path = evidence_dir / "query_face_before_alignment.jpg"
        _write_image(crop_path, crop)
        visual["face_crop_before_alignment"] = crop_path.relative_to(task_root).as_posix()
        visual["face_crop_bbox_pixels"] = crop_bbox
        artifacts.append(
            _artifact(
                task_root,
                crop_path,
                artifact_id="identity.query.face_before_alignment",
                kind="identity_face_crop_before_alignment",
                description="按本次真实检测框裁出的对齐前查询人脸。",
            )
        )
    except Exception as exc:
        warnings.append(f"查询人脸裁剪图生成失败: {type(exc).__name__}: {exc}")

    try:
        query_aligned = _align_face(query_image, query_diagnostics)
        aligned_path = evidence_dir / "query_face_aligned_112.jpg"
        _write_image(aligned_path, query_aligned)
        visual["aligned_face_image"] = aligned_path.relative_to(task_root).as_posix()
        artifacts.append(
            _artifact(
                task_root,
                aligned_path,
                artifact_id="identity.query.face_aligned",
                kind="identity_face_aligned",
                description="由本次真实五点关键点生成的 112×112 标准对齐查询人脸。",
            )
        )
    except Exception as exc:
        warnings.append(f"查询人脸对齐图生成失败: {type(exc).__name__}: {exc}")

    candidate_card_paths: dict[int, str] = {}
    candidate_card_images: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            rank = int(candidate.get("rank"))
            identity_id = str(candidate.get("identity_id"))
            reference_value = candidate.get("best_reference_image") or candidate.get(
                "reference_image"
            )
            reference_path = Path(str(reference_value)).expanduser().resolve(
                strict=False
            )
            candidate_image = read_image_unicode(reference_path)
            candidate_path = evidence_dir / (
                f"candidate_rank_{rank:02d}_{_safe_component(identity_id)}_best_reference.jpg"
            )
            card_image = candidate_image if rank == 1 else _thumbnail(candidate_image)
            _write_image(candidate_path, card_image)
            relative_candidate_path = candidate_path.relative_to(task_root).as_posix()
            candidate_card_paths[rank] = relative_candidate_path
            candidate_card_images.append(
                {
                    "rank": rank,
                    "identity_id": identity_id,
                    "path": relative_candidate_path,
                    "source_file_name": reference_path.name,
                    "image_role": (
                        "full_evidence_copy" if rank == 1 else "candidate_thumbnail"
                    ),
                    "image_size": [int(card_image.shape[1]), int(card_image.shape[0])],
                    "status": "success",
                }
            )
            artifacts.append(
                _artifact(
                    task_root,
                    candidate_path,
                    artifact_id=(
                        "identity.top1.best_reference_original"
                        if rank == 1
                        else f"identity.candidate.rank_{rank:02d}.best_reference"
                    ),
                    kind=(
                        "identity_best_reference_original"
                        if rank == 1
                        else "identity_candidate_best_reference"
                    ),
                    description=(
                        "当前库内第一候选的真实最佳参考图任务内副本。"
                        if rank == 1
                        else f"当前库内第 {rank} 候选的真实最佳参考图任务内副本。"
                    ),
                )
            )
        except Exception as exc:
            warnings.append(
                "候选卡片最佳参考图生成失败: "
                f"rank={candidate.get('rank')} {type(exc).__name__}: {exc}"
            )
            candidate_card_images.append(
                {
                    "rank": candidate.get("rank"),
                    "identity_id": candidate.get("identity_id"),
                    "path": None,
                    "source_file_name": Path(
                        str(
                            candidate.get("best_reference_image")
                            or candidate.get("reference_image")
                            or ""
                        )
                    ).name,
                    "status": "failed",
                }
            )
    visual["candidate_card_images"] = candidate_card_images

    top_candidate = candidates[0] if candidates else None
    if top_candidate:
        reference_value = top_candidate.get("best_reference_image") or top_candidate.get("reference_image")
        reference_path = Path(str(reference_value)).expanduser().resolve(strict=False)
        visual["best_reference"] = {
            "identity_id": top_candidate.get("identity_id"),
            "source_file_name": reference_path.name,
            "best_single_image_similarity": top_candidate.get("best_reference_score"),
            "person_mean_similarity": top_candidate.get("mean_similarity", top_candidate.get("score")),
            "original_image": candidate_card_paths.get(1),
        }
        try:
            reference_feature, reference_face = extractor.extract(reference_path)
            reference_image = read_image_unicode(reference_path)
            reference_diagnostics = face_diagnostics(reference_face)
            visual["best_reference"]["face"] = reference_diagnostics
            recomputed = float(
                np.dot(
                    np.asarray(query_feature, dtype=np.float32).reshape(-1),
                    np.asarray(reference_feature, dtype=np.float32).reshape(-1),
                )
            )
            expected = float(top_candidate.get("best_reference_score"))
            visual["best_reference"]["evidence_recomputed_similarity"] = recomputed
            visual["best_reference"]["evidence_similarity_delta"] = recomputed - expected
            if abs(recomputed - expected) > 1e-6:
                warnings.append(
                    "最佳参考图证据重算余弦相似度与缓存分数差值超过 1e-6；"
                    "候选排序仍保留原始缓存分数。"
                )

            annotated_reference_path = evidence_dir / "best_reference_face_detection.jpg"
            _write_image(
                annotated_reference_path,
                _annotate_face(reference_image, reference_diagnostics),
            )
            visual["best_reference"]["face_detection_image"] = annotated_reference_path.relative_to(task_root).as_posix()
            artifacts.append(
                _artifact(
                    task_root,
                    annotated_reference_path,
                    artifact_id="identity.top1.best_reference_face_detection",
                    kind="identity_best_reference_face_detection",
                    description="最佳参考图上的真实人脸检测框与五点关键点。",
                )
            )

            reference_aligned = _align_face(reference_image, reference_diagnostics)
            aligned_reference_path = evidence_dir / "best_reference_face_aligned_112.jpg"
            _write_image(aligned_reference_path, reference_aligned)
            visual["best_reference"]["aligned_face_image"] = aligned_reference_path.relative_to(task_root).as_posix()
            artifacts.append(
                _artifact(
                    task_root,
                    aligned_reference_path,
                    artifact_id="identity.top1.best_reference_aligned",
                    kind="identity_best_reference_aligned",
                    description="由最佳参考图真实五点关键点生成的 112×112 标准对齐人脸。",
                )
            )

            if query_aligned is not None:
                comparison_path = evidence_dir / "query_vs_best_reference.jpg"
                _write_image(
                    comparison_path,
                    _comparison_image(query_aligned, reference_aligned, top_candidate),
                )
                visual["best_reference_comparison_image"] = comparison_path.relative_to(task_root).as_posix()
                artifacts.append(
                    _artifact(
                        task_root,
                        comparison_path,
                        artifact_id="identity.top1.query_best_comparison",
                        kind="identity_query_best_reference_comparison",
                        description="查询对齐人脸与第一候选最佳参考对齐人脸的并排证据，不含虚构对应线。",
                    )
                )
        except Exception as exc:
            warnings.append(f"最佳参考图对比证据生成失败: {type(exc).__name__}: {exc}")
    else:
        warnings.append("本次任务没有身份候选，未生成最佳参考图对比证据。")

    visual["warnings"] = warnings
    visual["status"] = "complete" if not warnings else "partial"
    metadata_path = evidence_dir / "identity_evidence.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(visual, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    visual["metadata_file"] = metadata_path.relative_to(task_root).as_posix()
    artifacts.append(
        _artifact(
            task_root,
            metadata_path,
            artifact_id="identity.evidence.metadata",
            kind="identity_evidence_metadata",
            description="身份检索视觉证据的任务内结构化元数据。",
        )
    )
    return visual, artifacts, warnings


__all__ = ["ALIGNED_FACE_SIZE", "build_identity_visual_evidence"]
