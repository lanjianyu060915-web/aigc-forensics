"""大模型用户设置与实际上传范围提示。"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import streamlit as st

from orchestrator import load_config, resolve_path
from reporting.llm_client import complete_json, load_llm_settings, write_json


def llm_context(config_path: str | Path):
    config, resolved = load_config(config_path)
    profile = resolve_path("{DATA_ROOT}/profile/llm.json", resolved)
    return config, profile, load_llm_settings(config, profile_path=profile)


def upload_notice(settings) -> str:
    if not settings.configured:
        return "大模型接口未完整配置，本次使用本地规则和模板报告。"
    if settings.options["upload_scope"] == "text_only":
        return "只上传去除文件名等信息后的结构化结果，不上传图片；报告图件在本地嵌入。"
    return "启用大模型研判或报告撰写时，系统会把本次材料的图片、分析过程图件和全部结构化结果上传到所配置的接口。本机路径不上传。"


def save_llm_profile(path: Path, *, base_url: str, model: str, api_key: str,
                     upload_scope: str, decision_mode: str, judge_send_images: bool,
                     config: dict | None = None) -> None:
    """由用户表单保存；空密钥保持原值，页面从不回填密钥。"""
    if path.is_symlink() or any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in path.parents):
        raise ValueError("用户配置目录不能是链接")
    current = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    if not isinstance(current, dict):
        raise ValueError("用户配置不是有效对象")
    current.update(base_url=base_url.strip(), model=model.strip(), upload_scope=upload_scope,
                   decision_mode=decision_mode, judge_send_images=judge_send_images)
    if api_key.strip():
        current["api_key"] = api_key.strip()
    candidate = load_llm_settings({**(config or {}), "llm": current}, environ={})
    if candidate.configuration_error:
        raise ValueError(candidate.configuration_error)
    current.update(upload_scope=candidate.options["upload_scope"], judge_send_images=candidate.options["judge_send_images"])
    write_json(path, current)


def render_llm_status(config_path: str | Path) -> None:
    try:
        config, _, settings = llm_context(config_path)
        state = settings.public_status()
        face = config.get("general_aigc", {}).get("face_branch", {})
        st.markdown("#### 大模型与人脸分支")
        st.table([{"项目": name, "当前值": value} for name, value in [
            ("大模型配置", "已配置" if state["configured"] else "未完整配置"),
            ("接口主机", state["host"] or "未设置"), ("模型", state["model"] or "未设置"),
            ("研判方式", "大模型研判" if state["decision_mode"] == "llm" else "规则研判"),
            ("报告撰写", "云端辅助（已配置，按任务调用）" if state["configured"] else "本地模板"),
            ("上传范围", "图件与结构化结果" if state["upload_scope"] == "full" else "脱敏结构化结果"),
            ("研判带图", "是" if state["judge_send_images"] and state["upload_scope"] == "full" else "否"),
            ("人脸后端", str(face.get("backend", "cf384"))),
            ("人脸定档口径", str(face.get("band_aggregation", "largest"))) ]])
        st.caption(upload_notice(settings))
    except (OSError, ValueError) as error:
        st.warning(f"大模型设置不可读取：{type(error).__name__}")


def render_llm_settings(config_path: str | Path) -> None:
    st.subheader("大模型设置")
    try:
        config, profile, settings = llm_context(config_path)
    except (OSError, ValueError) as error:
        st.error(f"无法读取设置：{type(error).__name__}")
        return
    if st.session_state.pop("llm_profile_saved", False):
        st.success("设置已保存。密钥不会在页面回显。")
    st.caption("环境变量优先于此页保存的接口、模型和密钥。检测模型在本机运行，大模型用于研判与报告撰写。")
    policy = config.get("test_only_policy", {})
    policy = policy if isinstance(policy, dict) else {}
    text_only_locked = policy.get("cloud_text_only") is True
    endpoint_locked = "allowed_cloud_base_url" in policy
    if text_only_locked:
        st.info("测试资料保护已启用：仅发送当前任务的去标识结构化结果，图片与参考库原件留在本机；此页不能改为带图上传。")
    if endpoint_locked:
        st.caption("云端地址由测试隔离策略限定；不符合策略时停止调用。服务端数据保留按供应商条款执行。")
    st.write("密钥：" + ("已设置" if settings.api_key else "未设置"))
    active_env = [label for key, label in [("AIGC_LLM_BASE_URL", "接口地址"), ("AIGC_LLM_MODEL", "模型"), ("AIGC_LLM_API_KEY", "密钥")] if os.environ.get(key)]
    if active_env:
        st.caption("当前由环境变量覆盖：" + "、".join(active_env))
    if settings.configuration_error:
        st.warning("当前大模型配置无效。接口地址未回填，请检查环境变量或重新填写设置。")
    with st.form("llm_settings_form", clear_on_submit=True):
        # 被拒绝的URL仍可能含userinfo或查询参数；不能把它回填到明文输入框。
        base_url = st.text_input("兼容接口地址", value=settings.base_url if not settings.configuration_error else "", disabled=endpoint_locked)
        model = st.text_input("模型名", value=settings.model)
        api_key = st.text_input("新密钥（留空保留原值）", type="password", value="")
        scope_options = ["text_only"] if text_only_locked else ["full", "text_only"]
        scope_index = scope_options.index(settings.options["upload_scope"]) if settings.options["upload_scope"] in scope_options else 0
        scope = st.selectbox("上传范围", scope_options, index=scope_index, disabled=text_only_locked,
                             format_func=lambda x: "图件与结构化结果" if x == "full" else "仅脱敏结构化结果")
        mode = st.selectbox("研判方式", ["rule", "llm"], index=0 if settings.options["decision_mode"] == "rule" else 1,
                            format_func=lambda x: "规则研判" if x == "rule" else "大模型研判")
        images = st.checkbox("研判时附带图件", value=settings.options["judge_send_images"],
                             disabled=text_only_locked, help="仅文本模式优先；全量模式下报告撰写始终附带图件。")
        saved = st.form_submit_button("保存设置")
    if saved:
        try:
            save_llm_profile(profile, base_url=base_url, model=model, api_key=api_key,
                             upload_scope=scope, decision_mode=mode, judge_send_images=images, config=config)
            st.session_state["llm_profile_saved"] = True
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"设置未保存：{type(error).__name__}，请检查配置内容和目录。")
    st.info(upload_notice(settings))
    if st.button("测试连接", disabled=not settings.configured):
        try:
            options = {**settings.options, "timeout_seconds": min(30, settings.options["timeout_seconds"]), "writer_enable_thinking": False}
            with st.spinner("正在发送不含案件信息的测试请求"):
                result = complete_json(replace(settings, options=options), system_prompt='请仅返回 JSON 对象 {"ok":true}。',
                                       user_payload={"purpose": "连接测试，无案件信息"}, stage="writer", bypass_cache=True)
                content = result["response"]["choices"][0]["message"]["content"]
                valid = json.loads(content).get("ok") is True
            if valid:
                st.success("连接成功，接口返回有效 JSON。")
            else:
                st.warning("接口已返回，测试内容未通过校验。")
        except Exception as error:
            st.error(f"连接测试未通过（{getattr(error, 'code', type(error).__name__)}）。请核对已配置地址和模型。")
