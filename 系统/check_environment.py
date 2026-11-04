from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system_status import inspect_system_status


PRODUCT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PRODUCT_ROOT / "config.json"
BUFFALO_L_MODELS = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


@dataclass(frozen=True)
class CheckResult:
    level: str
    item: str
    detail: str

    def render(self) -> str:
        return f"{self.level} | {self.item} | {self.detail}"


def _result(ok: bool, item: str, success: str, failure: str) -> CheckResult:
    return CheckResult("PASS" if ok else "FAIL", item, success if ok else failure)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置顶层不是 JSON 对象")
    return payload


def _resolve(value: str | Path, config_path: Path) -> Path:
    raw_value = os.path.expandvars(str(value))
    configured_data_root = os.environ.get("FSTS_DATA_ROOT")
    local_app_data = os.environ.get("LOCALAPPDATA")
    data_root = (
        Path(configured_data_root).expanduser()
        if configured_data_root
        else Path(local_app_data) / "FaceSourceTrace"
        if local_app_data
        else Path.home() / "AppData" / "Local" / "FaceSourceTrace"
    )
    for token, replacement in {
        "{APP_ROOT}": str(config_path.parent),
        "{DATA_ROOT}": str(data_root),
        "{CURRENT_PYTHON}": sys.executable,
    }.items():
        raw_value = raw_value.replace(token, replacement)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=False)


def _configured_path(
    config: dict[str, Any],
    section_name: str,
    keys: tuple[str, ...],
    config_path: Path,
) -> Path:
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise KeyError(f"缺少配置对象 {section_name}")
    for key in keys:
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            return _resolve(value, config_path)
    raise KeyError(f"{section_name} 缺少配置项 {'/'.join(keys)}")


def _check_file(
    config: dict[str, Any],
    config_path: Path,
    section: str,
    key: str,
    item: str,
) -> CheckResult:
    try:
        path = _configured_path(config, section, (key,), config_path)
    except (KeyError, TypeError, ValueError) as exc:
        return CheckResult("FAIL", item, str(exc))
    return _result(
        path.is_file(),
        item,
        f"存在：{path}",
        f"不存在或不是文件：{path}",
    )


def _check_any_file(
    config: dict[str, Any],
    config_path: Path,
    section: str,
    keys: tuple[str, ...],
    item: str,
) -> CheckResult:
    try:
        path = _configured_path(config, section, keys, config_path)
    except (KeyError, TypeError, ValueError) as exc:
        return CheckResult("FAIL", item, str(exc))
    return _result(
        path.is_file(),
        item,
        f"存在：{path}",
        f"不存在或不是文件：{path}",
    )


def _check_library(
    config: dict[str, Any],
    config_path: Path,
    section: str,
    item: str,
) -> CheckResult:
    try:
        path = _configured_path(
            config,
            section,
            ("library", "default_library"),
            config_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return CheckResult("FAIL", item, str(exc))
    return _result(
        path.is_dir(),
        item,
        f"存在：{path}",
        f"不存在或不是目录：{path}",
    )


def _check_models(config: dict[str, Any], config_path: Path) -> list[CheckResult]:
    try:
        model_root = _configured_path(
            config,
            "identity",
            ("model_root",),
            config_path,
        )
        model_dir = (
            model_root
            if model_root.name.casefold() == "buffalo_l"
            else model_root / "models" / "buffalo_l"
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [CheckResult("FAIL", "buffalo_l 五模型", str(exc))]

    results: list[CheckResult] = []
    for filename in BUFFALO_L_MODELS:
        path = model_dir / filename
        results.append(
            _result(
                path.is_file(),
                f"buffalo_l/{filename}",
                f"存在：{path}",
                f"不存在或不是文件：{path}",
            )
        )
    return results


def _check_frontend(config: dict[str, Any], config_path: Path) -> CheckResult:
    item = "frontend 依赖"
    try:
        python_path = _configured_path(
            config,
            "frontend",
            ("python",),
            config_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return CheckResult("FAIL", item, str(exc))
    if not python_path.is_file():
        return CheckResult("FAIL", item, f"frontend Python 不存在：{python_path}")

    try:
        completed = subprocess.run(
            [str(python_path), "-B", "-c", "import streamlit"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("FAIL", item, f"无法验证 streamlit：{exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"退出码 {completed.returncode}"
        return CheckResult("FAIL", item, f"streamlit 不可导入：{reason}")
    return CheckResult("PASS", item, f"streamlit 可由 {python_path} 导入")


def _runtime_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        values = [
            runtime.get("tasks_root"),
            runtime.get("inbox_root"),
            runtime.get("identity_cache"),
        ]
        paths = [
            _resolve(value, config_path)
            for value in values
            if isinstance(value, str) and value.strip()
        ]
        if paths:
            return paths

    value = config.get("runtime_root")
    if isinstance(value, str) and value.strip():
        return [_resolve(value, config_path)]
    raise KeyError("缺少 runtime 目录配置")


def _check_runtime(config: dict[str, Any], config_path: Path) -> CheckResult:
    item = "runtime 目录可写"
    try:
        paths = _runtime_paths(config, config_path)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".environment_probe_",
                suffix=".tmp",
                dir=path,
                delete=True,
            ) as probe:
                probe.write("ok")
                probe.flush()
    except (KeyError, OSError) as exc:
        return CheckResult("FAIL", item, str(exc))
    return CheckResult(
        "PASS",
        item,
        "可写：" + ", ".join(str(path) for path in paths),
    )


def run_checks(config_path: str | Path = DEFAULT_CONFIG) -> list[CheckResult]:
    resolved_config = Path(config_path).expanduser().resolve(strict=False)
    try:
        config = _load_config(resolved_config)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return [CheckResult("FAIL", "config", f"无法读取 {resolved_config}：{exc}")]

    checks: list[CheckResult] = [
        _check_file(config, resolved_config, "deepfake", "python", "AIGC Python"),
        _check_file(config, resolved_config, "deepfake", "runner", "AIGC adapter"),
        _check_any_file(
            config,
            resolved_config,
            "identity",
            ("worker_executable", "executable", "python"),
            "身份运行器",
        ),
        _check_any_file(
            config,
            resolved_config,
            "source",
            ("executable", "python"),
            "底图运行器",
        ),
        _check_library(config, resolved_config, "identity", "identity_library"),
        _check_library(config, resolved_config, "source", "source_library"),
    ]
    checks.extend(_check_models(config, resolved_config))
    system = inspect_system_status(resolved_config)
    for module_name, item in (
        ("deepfake", "AIGC EXE/模型/config/SHA"),
        ("identity", "身份库/embedding/index"),
        ("source", "底图库/实时扫描状态"),
    ):
        module = system[module_name]
        checks.append(
            CheckResult(
                "PASS" if module.get("available") else "FAIL",
                item,
                str(module.get("message") or module.get("status")),
            )
        )
    checks.append(
        CheckResult(
            "PASS",
            "运算设备",
            str(system.get("device", {}).get("message", "设备信息不可用"))
            + f"；AIGC 实际设备={system.get('device', {}).get('deepfake_effective', 'unknown')}",
        )
    )
    checks.append(_check_frontend(config, resolved_config))
    checks.append(_check_runtime(config, resolved_config))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查三模块产品的必要运行环境。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run_checks(args.config)
    except Exception as exc:  # 保证环境检查自身也以受控状态结束。
        results = [CheckResult("FAIL", "environment checker", str(exc))]
    for result in results:
        print(result.render())
    return 1 if any(result.level == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
