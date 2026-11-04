from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

import cv2

from .face_mask import FaceMasker
from .feature_cache import LibraryEntry, SourceFeatureCache
from .global_features import extract_global, global_similarity_with_components
from .image_io import read_image, scan_library, write_image
from .local_matcher import LocalMatcher
from .visualizer import (
    make_all_matches_visualization,
    make_homography_overlay,
    make_inlier_outlier_visualization,
    make_keypoints_visualization,
    make_query_face_detection,
    make_visualization,
)


RESULT_SCHEMA_VERSION = "3.0"
GLOBAL_WEIGHT = 0.45
LOCAL_WEIGHT = 0.55
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_relative(path: Path, task_dir: Path) -> str:
    return path.resolve(strict=False).relative_to(
        task_dir.resolve(strict=False)
    ).as_posix()


def _optional_task_relative(path: Path, task_dir: Path) -> str | None:
    try:
        return _task_relative(path, task_dir)
    except ValueError:
        return None


def _artifact_entry(
    *,
    artifact_id: str,
    kind: str,
    path: Path | None,
    task_dir: Path,
    description: str,
    status: str = "success",
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "id": artifact_id,
            "kind": kind,
            "path": None,
            "sha256": None,
            "size_bytes": 0,
            "status": "not_generated" if status == "success" else status,
            "description": description,
        }
    return {
        "id": artifact_id,
        "kind": kind,
        "path": _task_relative(path, task_dir),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "status": status,
        "description": description,
    }


def _write_optional_visual(
    path: Path,
    producer,
    warnings: list[str],
    label: str,
) -> Path | None:
    """Write supplementary evidence without invalidating a valid ranking."""

    try:
        image = producer()
        if image is None:
            return None
        write_image(path, image)
    except Exception as exc:  # OpenCV/image encoding is evidence-only here.
        warnings.append(f"{label}生成失败：{type(exc).__name__}: {exc}")
        return None
    return path


def _emit_progress(
    reporter, stage: str, status: str, message: str, **kwargs: Any
) -> None:
    """Keep observability failures isolated from the frozen search algorithm."""

    if reporter is None:
        return
    try:
        reporter.emit(stage, status, message, **kwargs)
    except Exception:
        # Progress is supplementary evidence. A disk/permission error here must
        # not alter a valid source-search result or its ranking.
        return


@dataclass
class Candidate:
    path: Path
    image: object
    mask: object
    global_score: float
    global_components: dict[str, Any] | None = None
    local: object | None = None
    final_score: float = 0.0
    cache_entry: LibraryEntry | None = None


def search_source_images(
    query_path: Path,
    library_path: Path,
    top_k: int,
    output_dir: Path,
    visualization_stem: str | None = None,
    progress_reporter=None,
    feature_cache_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    if top_k < 1:
        raise ValueError("top-k must be at least 1")

    _emit_progress(
        progress_reporter,
        "read_source_query",
        "running",
        "正在读取底图检索查询图片。",
        progress_percent=None,
        details={"query_file_name": query_path.name},
        force=True,
    )
    query = read_image(query_path)
    if query is None:
        raise ValueError(f"query image cannot be decoded: {query_path}")
    _emit_progress(
        progress_reporter,
        "read_source_query",
        "success",
        "底图检索查询图片读取完成。",
        progress_percent=None,
        details={
            "query_file_name": query_path.name,
            "image_width": int(query.shape[1]),
            "image_height": int(query.shape[0]),
        },
        force=True,
    )

    masker = FaceMasker()
    _emit_progress(
        progress_reporter,
        "detect_source_query_face",
        "running",
        "正在检测查询图中的主要人脸区域。",
        progress_percent=None,
        details={
            "image_width": int(query.shape[1]),
            "image_height": int(query.shape[0]),
        },
        force=True,
    )
    detected_query_box = masker.detect(query)
    q_box = (
        tuple(int(value) for value in detected_query_box)
        if detected_query_box is not None
        else None
    )
    _emit_progress(
        progress_reporter,
        "detect_source_query_face",
        "success" if q_box is not None else "warning",
        (
            "查询图人脸检测完成，已定位主要人脸区域。"
            if q_box is not None
            else "查询图未检测到人脸，将使用完整图像继续检索。"
        ),
        progress_percent=None,
        details={
            "query_face_detected": q_box is not None,
            "query_face_bbox": list(q_box) if q_box is not None else None,
        },
        force=True,
    )
    q_mask = masker.mask(query.shape, q_box)

    visualization_dir = output_dir / "visualizations" / "source"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    visualization_prefix = f"{visualization_stem}_" if visualization_stem else ""
    artifact_manifest: list[dict[str, Any]] = []
    evidence_warnings: list[str] = []
    masked_query_path = (
        visualization_dir / f"{visualization_prefix}query_face_masked.jpg"
    )
    face_detection_path = (
        visualization_dir / f"{visualization_prefix}query_face_detection.jpg"
    )
    _emit_progress(
        progress_reporter,
        "mask_query_face",
        "running",
        "正在按真实检测区域生成查询图检索掩膜。",
        progress_percent=None,
        details={"face_mask_applied": q_box is not None},
        force=True,
    )
    masked_query = cv2.bitwise_and(query, query, mask=q_mask)
    write_image(masked_query_path, masked_query)
    face_detection_artifact_path = _write_optional_visual(
        face_detection_path,
        lambda: make_query_face_detection(query, q_box, q_mask),
        evidence_warnings,
        "查询图人脸检测证据",
    )
    artifact_manifest.extend(
        [
            _artifact_entry(
                artifact_id="source_query_face_detection",
                kind="source_face_detection",
                path=face_detection_artifact_path,
                task_dir=output_dir,
                description="查询图真实人脸检测框与实际屏蔽区域轮廓。",
            ),
            _artifact_entry(
                artifact_id="source_query_face_masked",
                kind="source_face_masked_query",
                path=masked_query_path,
                task_dir=output_dir,
                description="实际用于底图全局检索的查询图人脸屏蔽结果。",
            ),
        ]
    )
    _emit_progress(
        progress_reporter,
        "mask_query_face",
        "success",
        "查询图人脸区域屏蔽证据已生成。",
        progress_percent=None,
        details={
            "face_mask_applied": q_box is not None,
            "masked_pixel_count": int((q_mask == 0).sum()),
            "masked_query_artifact": _task_relative(
                masked_query_path, output_dir
            ),
        },
        force=True,
    )

    _emit_progress(
        progress_reporter,
        "extract_query_global_features",
        "running",
        "正在提取查询图的颜色、灰度、边缘和外围区域全局特征。",
        progress_percent=None,
        details={
            "query_face_detected": q_box is not None,
            "query_face_bbox": list(q_box) if q_box is not None else None,
            "face_mask_applied": q_box is not None,
        },
        force=True,
    )
    q_global = extract_global(query, q_mask)

    def report_decode(current: int, total: int, path: Path, decoded: bool) -> None:
        _emit_progress(
            progress_reporter,
            "decode_source_library",
            "running",
            f"正在读取底图库：{current} / {total}",
            progress_current=current,
            progress_total=total,
            # Decoding is preparatory work and is not assigned an invented share
            # of the measured coarse/local matching progress.
            progress_percent=None,
            details={
                "library_current": current,
                "library_total": total,
                "candidate_file_name": path.name,
                "decoded": decoded,
            },
        )

    feature_cache: SourceFeatureCache | None = None
    cached_entries: list[LibraryEntry] = []
    library = []
    if feature_cache_dir is not None:
        try:
            feature_cache = SourceFeatureCache(Path(feature_cache_dir), library_path)
            cached_entries, skipped = feature_cache.load_or_build(
                masker, progress_callback=report_decode
            )
        except ValueError:
            raise
        except Exception as exc:  # 缓存只是加速手段，失败时回到逐图实时扫描。
            feature_cache = None
            cached_entries = []
            evidence_warnings.append(
                f"底图特征缓存不可用，已回退实时扫描：{type(exc).__name__}: {exc}"
            )
    if feature_cache is None:
        library, skipped = scan_library(
            library_path, progress_callback=report_decode
        )
    library_decoded = len(cached_entries) if feature_cache is not None else len(library)
    library_file_count = library_decoded + len(skipped)
    shortlist_target = min(max(top_k * 4, 20), library_decoded)
    total_measured_work = library_decoded + shortlist_target

    def iterate_library():
        if feature_cache is not None:
            for entry in cached_entries:
                yield entry.path, None, None, entry.features, entry
            return
        for path, image in library:
            mask = masker.mask(image.shape, masker.detect(image))
            yield path, image, mask, extract_global(image, mask), None

    candidates = []
    for current, (path, image, mask, features, entry) in enumerate(iterate_library(), 1):
        score, global_components = global_similarity_with_components(
            q_global, features
        )
        candidates.append(
            Candidate(
                path,
                image,
                mask,
                score,
                global_components=global_components,
                cache_entry=entry,
            )
        )
        _emit_progress(
            progress_reporter,
            "scan_source_library",
            "running",
            f"正在扫描底图库：{current} / {library_decoded}",
            progress_current=current,
            progress_total=library_decoded,
            progress_percent=100.0 * current / total_measured_work,
            details={
                "coarse_current": current,
                "coarse_total": library_decoded,
                "candidate_file_name": path.name,
                "decoded_library_count": library_decoded,
                "global_score": score,
            },
        )

    # Sorting keys and shortlist sizing are intentionally unchanged from v0.2.
    candidates.sort(key=lambda c: (-c.global_score, str(c.path).lower()))
    shortlist = candidates[:shortlist_target]
    _emit_progress(
        progress_reporter,
        "source_global_ranking",
        "success",
        f"全局粗排完成，选取 {len(shortlist)} 个候选进行局部复核。",
        progress_current=library_decoded,
        progress_total=library_decoded,
        progress_percent=100.0 * library_decoded / total_measured_work,
        details={
            "library_decoded": library_decoded,
            "shortlist_size": len(shortlist),
        },
        force=True,
    )

    matcher = LocalMatcher(
        ratio=LOWE_RATIO, ransac_threshold=RANSAC_THRESHOLD
    )
    query_sift = None
    if feature_cache is not None:
        query_sift = matcher.sift.detectAndCompute(
            cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), q_mask
        )
    for current, candidate in enumerate(shortlist, 1):
        completed_before = current - 1
        _emit_progress(
            progress_reporter,
            "source_geometric_review",
            "running",
            f"正在几何复核：{current} / {len(shortlist)}",
            progress_current=current,
            progress_total=len(shortlist),
            progress_percent=(
                100.0
                * (library_decoded + completed_before)
                / total_measured_work
            ),
            details={
                "review_current": current,
                "review_total": len(shortlist),
                "candidate_file_name": candidate.path.name,
                "lowe_ratio": LOWE_RATIO,
                "ransac_threshold": RANSAC_THRESHOLD,
            },
        )

        def report_local_stage(
            stage: str,
            details: dict[str, Any],
            *,
            candidate_path: Path = candidate.path,
            review_current: int = current,
        ) -> None:
            stage_metadata = {
                "sift_features_extracted": (
                    "source_sift_features",
                    "SIFT关键点与描述子提取完成。",
                ),
                "lowe_ratio_filtered": (
                    "source_lowe_filter",
                    "Lowe比率筛选完成。",
                ),
                "ransac_verified": (
                    "source_ransac_verification",
                    "RANSAC几何验证完成。",
                ),
            }
            event_stage, event_message = stage_metadata[stage]
            event_status = "success"
            if stage == "ransac_verified" and not details.get(
                "ransac_attempted", False
            ):
                event_status = "warning"
                event_message = "有效匹配不足，未执行RANSAC几何估计。"
            _emit_progress(
                progress_reporter,
                event_stage,
                event_status,
                event_message,
                progress_current=review_current,
                progress_total=len(shortlist),
                progress_percent=None,
                details={
                    "candidate_file_name": candidate_path.name,
                    "review_current": review_current,
                    "review_total": len(shortlist),
                    **details,
                },
                force=True,
            )

        if candidate.cache_entry is not None:
            candidate.local = matcher.match(
                query,
                q_mask,
                None,
                None,
                progress_callback=report_local_stage,
                features_a=query_sift,
                features_b=feature_cache.sift_features(
                    candidate.cache_entry, matcher.sift
                ),
            )
        else:
            candidate.local = matcher.match(
                query,
                q_mask,
                candidate.image,
                candidate.mask,
                progress_callback=report_local_stage,
            )
        # Frozen v0.2 formula. Do not round before sorting.
        candidate.final_score = (
            GLOBAL_WEIGHT * candidate.global_score
            + LOCAL_WEIGHT * candidate.local.score
        )
        _emit_progress(
            progress_reporter,
            "source_score_fusion",
            "success",
            "候选全局分与局部分融合完成。",
            progress_current=current,
            progress_total=len(shortlist),
            progress_percent=(
                100.0 * (library_decoded + current) / total_measured_work
            ),
            details={
                "candidate_file_name": candidate.path.name,
                "global_score": candidate.global_score,
                "local_score": candidate.local.score,
                "global_weight": GLOBAL_WEIGHT,
                "local_weight": LOCAL_WEIGHT,
                "global_contribution": (
                    GLOBAL_WEIGHT * candidate.global_score
                ),
                "local_contribution": LOCAL_WEIGHT * candidate.local.score,
                "final_score": candidate.final_score,
            },
            force=True,
        )

    # Sorting keys are intentionally unchanged from v0.2.
    shortlist.sort(
        key=lambda c: (-c.final_score, -c.global_score, str(c.path).lower())
    )
    _emit_progress(
        progress_reporter,
        "source_final_ranking",
        "success",
        "局部复核后的Top-K最终排序完成。",
        progress_current=len(shortlist),
        progress_total=len(shortlist),
        progress_percent=(
            100.0
            * (library_decoded + len(shortlist))
            / total_measured_work
        ),
        details={
            "ranked_candidate_count": len(shortlist),
            "requested_top_k": top_k,
        },
        force=True,
    )
    results = []
    top1_local_evidence: dict[str, Any] = {}
    _emit_progress(
        progress_reporter,
        "generate_source_visual_evidence",
        "running",
        "正在生成Top-K候选与局部匹配证据图。",
        progress_percent=None,
        details={"candidate_count": min(top_k, len(shortlist))},
        force=True,
    )
    for rank, candidate in enumerate(
        shortlist[: min(top_k, len(shortlist))], 1
    ):
        if candidate.image is None and candidate.cache_entry is not None:
            candidate.image = SourceFeatureCache.read_image(candidate.cache_entry)
        vis_path = (
            visualization_dir / f"{visualization_prefix}match_{rank:03d}.jpg"
        )
        candidate_image_path = visualization_dir / (
            f"{visualization_prefix}candidate_{rank:03d}.jpg"
        )
        local = candidate.local
        label = (
            f"Rank {rank} | global={candidate.global_score:.4f} | "
            f"local={local.score:.4f} | final={candidate.final_score:.4f} | "
            f"inliers={local.inlier_count}/{local.good_count}"
        )
        write_image(
            vis_path,
            make_visualization(query, candidate.image, local, label),
        )
        candidate_image_artifact_path = _write_optional_visual(
            candidate_image_path,
            lambda image=candidate.image: image,
            evidence_warnings,
            f"第{rank}候选缩略证据",
        )
        candidate_artifacts = {
            "candidate_image": (
                _task_relative(candidate_image_artifact_path, output_dir)
                if candidate_image_artifact_path is not None
                else None
            ),
            "matches_inliers": _task_relative(vis_path, output_dir),
        }
        artifact_manifest.extend(
            [
                _artifact_entry(
                    artifact_id=f"source_candidate_{rank:03d}_image",
                    kind="source_candidate_image",
                    path=candidate_image_artifact_path,
                    task_dir=output_dir,
                    description=(
                        f"最终第{rank}候选的实际底图库图像："
                        f"{candidate.path.name}。"
                    ),
                ),
                _artifact_entry(
                    artifact_id=(
                        f"source_candidate_{rank:03d}_matches_inliers"
                    ),
                    kind="source_ransac_inlier_matches",
                    path=vis_path,
                    task_dir=output_dir,
                    description=(
                        f"最终第{rank}候选经RANSAC保留的内点匹配。"
                    ),
                ),
            ]
        )

        if rank == 1:
            query_keypoints_path = visualization_dir / (
                f"{visualization_prefix}top1_query_keypoints.jpg"
            )
            candidate_keypoints_path = visualization_dir / (
                f"{visualization_prefix}top1_candidate_keypoints.jpg"
            )
            matches_all_path = visualization_dir / (
                f"{visualization_prefix}top1_matches_all_lowe.jpg"
            )
            matches_classified_path = visualization_dir / (
                f"{visualization_prefix}top1_matches_inliers_outliers.jpg"
            )
            homography_path = visualization_dir / (
                f"{visualization_prefix}top1_homography_projection.jpg"
            )
            query_keypoints_artifact_path = _write_optional_visual(
                query_keypoints_path,
                lambda: make_keypoints_visualization(
                    query, local.keypoints_a, "Query"
                ),
                evidence_warnings,
                "Top1查询图SIFT关键点证据",
            )
            candidate_keypoints_artifact_path = _write_optional_visual(
                candidate_keypoints_path,
                lambda: make_keypoints_visualization(
                    candidate.image,
                    local.keypoints_b,
                    f"Candidate {candidate.path.name}",
                ),
                evidence_warnings,
                "Top1候选图SIFT关键点证据",
            )
            matches_all_artifact_path = _write_optional_visual(
                matches_all_path,
                lambda: make_all_matches_visualization(
                    query, candidate.image, local, label
                ),
                evidence_warnings,
                "Top1全部Lowe匹配证据",
            )
            matches_classified_artifact_path = _write_optional_visual(
                matches_classified_path,
                lambda: make_inlier_outlier_visualization(
                    query, candidate.image, local, label
                ),
                evidence_warnings,
                "Top1 RANSAC内点外点证据",
            )
            homography_artifact_path = _write_optional_visual(
                homography_path,
                lambda: make_homography_overlay(
                    query, candidate.image, local, label
                ),
                evidence_warnings,
                "Top1单应投影证据",
            )
            detail_entries = [
                _artifact_entry(
                    artifact_id="source_top1_query_keypoints",
                    kind="source_sift_query_keypoints",
                    path=query_keypoints_artifact_path,
                    task_dir=output_dir,
                    description="第一候选复核所使用的查询图SIFT关键点。",
                ),
                _artifact_entry(
                    artifact_id="source_top1_candidate_keypoints",
                    kind="source_sift_candidate_keypoints",
                    path=candidate_keypoints_artifact_path,
                    task_dir=output_dir,
                    description="第一候选图上实际提取的SIFT关键点。",
                ),
                _artifact_entry(
                    artifact_id="source_top1_matches_all_lowe",
                    kind="source_lowe_matches_all",
                    path=matches_all_artifact_path,
                    task_dir=output_dir,
                    description="通过Lowe比率检验的全部有效匹配线。",
                ),
                _artifact_entry(
                    artifact_id="source_top1_matches_inliers_outliers",
                    kind="source_ransac_inliers_outliers",
                    path=matches_classified_artifact_path,
                    task_dir=output_dir,
                    description="RANSAC内点（绿）与被排除外点（红）。",
                ),
                _artifact_entry(
                    artifact_id="source_top1_homography_projection",
                    kind="source_homography_projection",
                    path=homography_artifact_path,
                    task_dir=output_dir,
                    description=(
                        "可稳定估计单应矩阵时，将查询图边界投影到候选图。"
                    ),
                ),
            ]
            artifact_manifest.extend(detail_entries)
            top1_local_evidence = {
                "candidate_rank": 1,
                "candidate_file_name": candidate.path.name,
                "query_keypoints": (
                    _task_relative(
                        query_keypoints_artifact_path, output_dir
                    )
                    if query_keypoints_artifact_path is not None
                    else None
                ),
                "candidate_keypoints": (
                    _task_relative(
                        candidate_keypoints_artifact_path, output_dir
                    )
                    if candidate_keypoints_artifact_path is not None
                    else None
                ),
                "matches_all": (
                    _task_relative(matches_all_artifact_path, output_dir)
                    if matches_all_artifact_path is not None
                    else None
                ),
                "matches_inliers": _task_relative(vis_path, output_dir),
                "matches_inliers_outliers": (
                    _task_relative(
                        matches_classified_artifact_path, output_dir
                    )
                    if matches_classified_artifact_path is not None
                    else None
                ),
                "homography_overlay": (
                    _task_relative(homography_artifact_path, output_dir)
                    if homography_artifact_path is not None
                    else None
                ),
                "homography_available": local.homography is not None,
                "homography_matrix": (
                    local.homography.tolist()
                    if local.homography is not None
                    else None
                ),
                "query_keypoint_count": len(local.keypoints_a),
                "candidate_keypoint_count": len(local.keypoints_b),
                "raw_knn_pair_count": local.raw_knn_pair_count,
                "lowe_good_match_count": local.good_count,
                "ransac_inlier_count": local.inlier_count,
                "ransac_outlier_count": (
                    local.good_count - local.inlier_count
                ),
            }
        inlier_ratio = (
            local.inlier_count / local.good_count if local.good_count else 0.0
        )
        results.append(
            {
                # Frozen v2 fields and rounding.
                "rank": rank,
                "file_path": str(candidate.path),
                "global_score": round(candidate.global_score, 6),
                "local_score": round(local.score, 6),
                "final_score": round(candidate.final_score, 6),
                "visualization_path": str(vis_path),
                "visualization_relative_path": _task_relative(
                    vis_path, output_dir
                ),
                "candidate_image_artifact": _task_relative(
                    candidate_image_artifact_path, output_dir
                ) if candidate_image_artifact_path is not None else None,
                "evidence_artifacts": candidate_artifacts,
                # v3 independently auditable diagnostics.
                "query_keypoint_count": len(local.keypoints_a),
                "candidate_keypoint_count": len(local.keypoints_b),
                "raw_knn_pair_count": local.raw_knn_pair_count,
                "good_match_count": local.good_count,
                "ransac_inlier_count": local.inlier_count,
                "ransac_inlier_ratio": inlier_ratio,
                "lowe_ratio": LOWE_RATIO,
                "ransac_threshold": RANSAC_THRESHOLD,
                "ransac_attempted": local.ransac_attempted,
                "ransac_applied": local.ransac_applied,
                "local_score_components": local.score_components,
                "global_score_components": candidate.global_components,
                "global_weight": GLOBAL_WEIGHT,
                "local_weight": LOCAL_WEIGHT,
                "score_fusion": {
                    "global_score": candidate.global_score,
                    "local_score": local.score,
                    "global_weight": GLOBAL_WEIGHT,
                    "local_weight": LOCAL_WEIGHT,
                    "global_contribution": (
                        GLOBAL_WEIGHT * candidate.global_score
                    ),
                    "local_contribution": LOCAL_WEIGHT * local.score,
                    "final_score": candidate.final_score,
                    "formula": (
                        "global_weight * global_score + "
                        "local_weight * local_score"
                    ),
                },
            }
        )

    _emit_progress(
        progress_reporter,
        "generate_source_visual_evidence",
        "success",
        "Top-K候选与局部匹配证据图生成完成。",
        progress_percent=None,
        details={
            "artifact_count": len(artifact_manifest),
            "top1_homography_generated": bool(
                top1_local_evidence.get("homography_overlay")
            ),
        },
        force=True,
    )

    top1_top2_margin = (
        round(
            results[0]["final_score"] - results[1]["final_score"], 6
        )
        if len(results) >= 2
        else None
    )
    warnings: list[str] = list(evidence_warnings)
    if q_box is None:
        warnings.append(
            "查询图未检测到可屏蔽的人脸区域，本次按完整图像提取底图特征。"
        )
    if skipped:
        warnings.append(
            f"底图库中有 {len(skipped)} 张图片无法解码，已跳过。"
        )

    explanations: list[str] = []
    if results:
        first = results[0]
        explanations.append(
            "系统先在屏蔽查询图主要人脸区域后执行全局粗检索，再使用 "
            "SIFT、Lowe 比率检验和 RANSAC 几何验证进行局部复核。"
        )
        explanations.append(
            f"当前库内第一候选的全局分为 {first['global_score']:.6f}，"
            f"局部分为 {first['local_score']:.6f}；最终分按 "
            f"{GLOBAL_WEIGHT} × {first['global_score']:.6f} + "
            f"{LOCAL_WEIGHT} × {first['local_score']:.6f} 计算并用于库内排序。"
        )

    runtime_seconds = round(time.perf_counter() - started, 6)
    metrics = {
        "query_face_detected": q_box is not None,
        "query_face_bbox": list(q_box) if q_box is not None else None,
        "face_mask_applied": q_box is not None,
        "masked_query_path": str(masked_query_path),
        "masked_query_relative_path": _task_relative(
            masked_query_path, output_dir
        ),
        "face_detection_relative_path": (
            _task_relative(face_detection_artifact_path, output_dir)
            if face_detection_artifact_path is not None
            else None
        ),
        "masked_pixel_count": int((q_mask == 0).sum()),
        "library_file_count": library_file_count,
        "library_decoded": library_decoded,
        "library_skipped_count": len(skipped),
        "library_skipped_paths": skipped,
        "global_compared_count": len(candidates),
        "shortlist_size": len(shortlist),
        "returned_candidate_count": len(results),
        "lowe_ratio": LOWE_RATIO,
        "ransac_threshold": RANSAC_THRESHOLD,
        "global_weight": GLOBAL_WEIGHT,
        "local_weight": LOCAL_WEIGHT,
        "top1_top2_margin": top1_top2_margin,
        "artifact_count": len(artifact_manifest),
        "feature_cache": (
            feature_cache.stats.as_dict()
            if feature_cache is not None
            else {"enabled": False}
        ),
    }
    method = {
        "name": (
            "face-masked global retrieval plus SIFT geometric verification"
        ),
        "global_features": ["HSV", "grayscale", "edge", "border"],
        "global_score_formula": (
            "0.30 * hsv_histogram_similarity + "
            "0.25 * grayscale_thumbnail_similarity + "
            "0.15 * edge_map_similarity + "
            "0.30 * border_region_similarity"
        ),
        "global_component_weights": {
            "hsv_histogram_similarity": 0.30,
            "grayscale_thumbnail_similarity": 0.25,
            "edge_map_similarity": 0.15,
            "border_region_similarity": 0.30,
        },
        "coarse_ranking": "global_similarity",
        "local_features": "SIFT (nfeatures=2500)",
        "matcher": "BFMatcher L2",
        "lowe_ratio": LOWE_RATIO,
        "ransac_threshold": RANSAC_THRESHOLD,
        "local_score_formula": (
            "0.6 * ransac_inlier_ratio + "
            "0.4 * min(ransac_inlier_count / 40, 1)"
        ),
        "insufficient_match_formula": (
            "min(good_match_count / 40, 0.15)"
        ),
        "final_score_formula": (
            "0.45 * global_score + 0.55 * local_score"
        ),
        "global_weight": GLOBAL_WEIGHT,
        "local_weight": LOCAL_WEIGHT,
    }
    visual_evidence = {
        "source": {
            "schema_version": "1.0",
            "query": {
                "original_image": _optional_task_relative(
                    query_path, output_dir
                ),
                "face_detection_image": (
                    _task_relative(
                        face_detection_artifact_path, output_dir
                    )
                    if face_detection_artifact_path is not None
                    else None
                ),
                "face_masked_image": _task_relative(
                    masked_query_path, output_dir
                ),
                "face_detected": q_box is not None,
                "face_bbox": list(q_box) if q_box is not None else None,
                "face_mask_applied": q_box is not None,
                "masked_pixel_count": int((q_mask == 0).sum()),
                "mask_purpose": (
                    "降低换脸区域对原始场景底图检索的干扰；未检测到人脸时使用完整图像。"
                ),
            },
            "global_stage": {
                "library_file_count": library_file_count,
                "library_decoded": library_decoded,
                "library_skipped_count": len(skipped),
                "library_skipped_paths": skipped,
                "global_compared_count": len(candidates),
                "shortlist_size": len(shortlist),
                "returned_top_k": len(results),
            },
            "global_scoring": {
                "formula": method["global_score_formula"],
                "weights": method["global_component_weights"],
                "candidates": [
                    {
                        "rank": item["rank"],
                        "file_name": Path(item["file_path"]).name,
                        "global_score": item["global_score"],
                        "components": item["global_score_components"],
                    }
                    for item in results
                ],
            },
            "top_candidates": [
                {
                    "rank": item["rank"],
                    "file_name": Path(item["file_path"]).name,
                    "candidate_image": item["candidate_image_artifact"],
                    "global_score": item["global_score"],
                    "local_score": item["local_score"],
                    "final_score": item["final_score"],
                    "good_match_count": item["good_match_count"],
                    "ransac_inlier_count": item["ransac_inlier_count"],
                    "ransac_inlier_ratio": item["ransac_inlier_ratio"],
                    "matches_inliers": item[
                        "visualization_relative_path"
                    ],
                    "gap_from_top1": round(
                        results[0]["final_score"] - item["final_score"], 6
                    ),
                }
                for item in results
            ],
            "local_matching": top1_local_evidence,
            "fusion": (
                {
                    **results[0]["score_fusion"],
                    "displayed_final_score": results[0]["final_score"],
                    "second_candidate_final_score": (
                        results[1]["final_score"]
                        if len(results) >= 2
                        else None
                    ),
                    "top1_top2_margin": top1_top2_margin,
                }
                if results
                else {}
            ),
            "artifact_ids": [item["id"] for item in artifact_manifest],
        }
    }
    diagnostics = {
        # Preserve v2 diagnostics keys.
        "runtime_seconds": runtime_seconds,
        "library_decoded": library_decoded,
        "library_skipped": skipped,
        "query_face_detected": q_box is not None,
        "shortlist_size": len(shortlist),
        # v3 diagnostics and explanatory blocks.
        **metrics,
        "method": method,
        "metrics": metrics,
        "explanation": explanations,
        "warnings": warnings,
        "limitations": [
            "结果是当前底图库内的候选排序，不构成原始底图确认。",
            "分数是检索相似度，不是来源概率。",
        ],
        "runtime": {"seconds": runtime_seconds},
        "visual_evidence": visual_evidence,
        "artifact_manifest": artifact_manifest,
        "provenance": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "query_path": str(query_path),
            "library_path": str(library_path),
        },
    }
    _emit_progress(
        progress_reporter,
        "source_search_complete",
        "success",
        "底图候选重排和匹配可视化已完成。",
        progress_percent=100.0,
        details={
            "library_decoded": library_decoded,
            "shortlist_size": len(shortlist),
            "returned_candidate_count": len(results),
            "top1_top2_margin": top1_top2_margin,
        },
        force=True,
    )
    return results, diagnostics
