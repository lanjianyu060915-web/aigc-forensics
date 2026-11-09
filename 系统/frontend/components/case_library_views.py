"""案件库选择、检查和建索引入口。"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from case_libraries import build_case_libraries, check_case_libraries
from orchestrator import resolve_path


def _reference_manifest_view(config: Mapping[str, Any], config_path: Path,
                             source: str, identity: str) -> dict[str, Any] | None:
    """只读本地出处清单；不读取图片，也不将原件映射加入模型输入。"""
    policy = config.get("test_only_policy", {})
    if not isinstance(policy, Mapping) or not policy.get("reference_manifest"):
        return None
    path = resolve_path(str(policy["reference_manifest"]), config_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, Mapping) or manifest.get("usage") != "test_only":
        raise ValueError("参考出处清单缺少 test_only 标记。")
    libraries = manifest.get("libraries", {})
    if not isinstance(libraries, Mapping) or any(not isinstance(libraries.get(kind), str) or not libraries[kind].strip()
                                               for kind in ("source", "identity")):
        raise ValueError("参考出处清单的库目录无效。")
    rows = manifest.get("references", [])
    if not isinstance(rows, list):
        raise ValueError("参考出处清单的文件列表无效。")
    counts = {"source": 0, "identity": 0}
    anonymous_count = 0
    origins = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("kind") not in counts:
            raise ValueError("参考出处清单含未知角色。")
        kind = row["kind"]
        counts[kind] += 1
        anonymous_count += row.get("role") == "anonymous_distractor_reference"
        row_origins = row.get("origins", [])
        if not isinstance(row_origins, list) or any(not isinstance(item, Mapping) for item in row_origins):
            raise ValueError("参考出处清单的来源映射无效。")
        for origin in row_origins:
            origins.append({"库角色": "素材参考" if kind == "source" else "身份参考",
                            "库内文件": row.get("relative_path", ""),
                            "原件出处": origin.get("original_path", ""),
                            "压缩包成员": origin.get("archive_member") or "",
                            "原登记压缩包": origin.get("registered_archive_original_path") or "",
                            "隔离快照": origin.get("snapshot_path", ""),
                            "SHA-256": row.get("sha256", "")})
    matches = all(
        value.strip() and resolve_path(value.strip(), config_path) ==
        resolve_path(str(libraries.get(kind, "")), config_path)
        for kind, value in (("source", source), ("identity", identity))
    )
    return {"name": policy.get("reference_library_name") or manifest.get("name", "功能测试参考库"),
            "original_data_root": policy.get("original_data_root") or manifest.get("original_data_root", ""),
            "manifest_path": str(path), "counts": counts, "origins": origins,
            "anonymous_count": anonymous_count,
            "matches_selection": bool(matches)}


def _show_reference_manifest(config: Mapping[str, Any], config_path: Path,
                             source: str, identity: str) -> None:
    try:
        view = _reference_manifest_view(config, config_path, source, identity)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"参考出处清单未能读取：{exc}")
        return
    if view is None:
        return
    st.caption(f"登记参考库：{view['name']} · 原始资料目录：{view['original_data_root']}")
    st.caption("这些资料及派生缓存仅用于功能测试，禁止训练、校准和调参；出处映射只在本机界面显示。")
    if not view["matches_selection"]:
        st.warning("当前选择目录与登记参考库不一致，以下清单不代表本次实际检索范围。")
    st.write(f"登记素材参考 {view['counts']['source']} 张；身份参考 {view['counts']['identity']} 张。"
             "有效数量以实际检查或建索引结果为准，登记数量不代表已可用。")
    if view["anonymous_count"]:
        st.info(f"其中 {view['anonymous_count']} 张为单图匿名参考；编号只代表图片参考项，"
                "不代表已确认人物，也不将不同图片自动合并为同一人。本库仅适用于本轮 B 查询。")
    with st.expander("查看原件出处与参考角色"):
        st.caption(f"出处清单：{view['manifest_path']}")
        st.dataframe(view["origins"], hide_index=True, width="stretch")


def _show_summary(result: Mapping[str, Any]) -> None:
    rows = []
    issues = []
    for kind, label in (("source", "素材库"), ("identity", "身份库")):
        item = result.get(kind, {})
        rows.append({"库": label, "图片数": item.get("file_count", "未记录"),
                     "无法解码": item.get("invalid_count", "未记录"),
                     "重复文件": item.get("duplicate_count", "未记录"),
                     "无人脸": item.get("no_face_count", "未记录"),
                     "多人脸参考": item.get("multiple_faces_count", "未记录"),
                     "可用图片": item.get("usable_count", "未记录")})
        for record in item.get("files", []):
            if record.get("issues"):
                issues.append({"库": label, "文件": record["path"],
                               "检查结果": "；".join(record["issues"])})
        for record in item.get("duplicates", []):
            issues.append({"库": label, "文件": record["path"],
                           "检查结果": f"内容重复：{record['duplicate_of']}"})
    st.dataframe(rows, hide_index=True, width="stretch")
    if issues:
        st.warning("部分照片需要检查。无法解码或不符合身份照片要求的文件不会形成有效身份参考。")
        st.dataframe(issues, hide_index=True, width="stretch")
    if result.get("status") == "ready":
        st.success("案件库索引已准备好，可继续研判。")
        st.caption(f"库名称：{result.get('name')} · 建库时间：{result.get('built_at')}")
        st.caption(f"实际素材索引 {result.get('source_indexed_count', '未记录')} 项；"
                   f"实际身份索引 {result.get('identity_indexed_count', '未记录')} 个参考分组（不等于确认人数）。")
    st.caption("建库阶段在本机读取文件；报告会记录本次比对库的名称、文件数及内容清单摘要。")


def render_case_library_settings(default_identity: str, default_source: str,
                                 config: Mapping[str, Any], config_path: Path) -> tuple[str, str]:
    with st.expander("案件库：选择目录、检查和建立索引"):
        policy = config.get("test_only_policy", {})
        default_name = policy.get("reference_library_name", "案件库") if isinstance(policy, Mapping) else "案件库"
        name = st.text_input("案件库名称", value=default_name, key="case_library_name")
        identity = st.text_input("本次任务身份库", value=default_identity, key="live_identity_root")
        source = st.text_input("本次任务素材库", value=default_source, key="live_source_root")
        st.caption("身份库每个参考分组一个子目录；素材库放候选原图。可选择本机任意目录，库内链接不能指向库外。")
        _show_reference_manifest(config, config_path, source, identity)
        controls = st.columns(2)
        check_clicked = controls[0].button("检查案件库", key="case_library_check", width="stretch")
        build_clicked = controls[1].button("检查并建立索引", key="case_library_build", width="stretch")
        selection = (source.strip(), identity.strip(), name.strip())
        previous = st.session_state.get("case_library_check_selection")
        if previous != selection:
            st.session_state.pop("case_library_check_result", None)
        if check_clicked or build_clicked:
            bar = st.progress(0.0, text="正在检查案件库")

            def progress(event: dict[str, Any]) -> None:
                total = event["total"]
                fraction = min(1.0, event["current"] / total) if total else 0.0
                bar.progress(fraction, text=event["message"])

            try:
                if not source.strip() or not identity.strip():
                    raise ValueError("请先选择素材库和身份库目录。")
                kwargs = {
                    "source_root": resolve_path(source.strip(), config_path),
                    "identity_root": resolve_path(identity.strip(), config_path),
                    "product_root": Path(__file__).resolve().parents[2],
                    "model_root": resolve_path(config["identity"]["model_root"], config_path),
                    "identity_scale": int(config["identity"].get("det_size_chain", [1280])[0]),
                    "identity_scales": tuple(config["identity"].get("det_size_chain", [1280, 640, 320])),
                    "progress": progress,
                }
                with st.spinner("正在处理本地案件库，请保留此页面"):
                    if build_clicked:
                        result = build_case_libraries(
                            **kwargs, name=name,
                            prewarm_sift=policy.get("prewarm_source_sift", True) if isinstance(policy, Mapping) else True,
                            cache_root=resolve_path("{DATA_ROOT}/runtime/cache", config_path),
                            source_cache=resolve_path(config["source"].get("feature_cache", "{DATA_ROOT}/runtime/cache/source"), config_path),
                            identity_cache=resolve_path(config["runtime"].get("identity_cache", "{DATA_ROOT}/runtime/cache/identity"), config_path))
                    else:
                        result = check_case_libraries(**kwargs)
                st.session_state["case_library_check_selection"] = selection
                st.session_state["case_library_check_result"] = result
                bar.progress(1.0, text="案件库检查完成")
            except Exception as exc:
                st.session_state.pop("case_library_check_result", None)
                st.error(f"案件库准备未完成：{exc}")
        result = st.session_state.get("case_library_check_result")
        if isinstance(result, Mapping):
            _show_summary(result)
    return identity, source
