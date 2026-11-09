"""Explicit independent-development configuration and pre-load admission hooks.

The policy authority comes from the launcher/coordinator, never from input data.
Configuration audit does not decode data or load models. Gate snapshots must be
consumed as bytes; reopening a path afterwards would discard that guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import uuid

from modules.data_admission import AdmittedDataset, PolicyAuthority

PROFILE_SCHEMA = "execution_profile.independent.v1"
_SECRET = re.compile(r"(?i)(api.?key|access.?key|token|password|secret|credential|authorization)")
_PATH_KEYS = frozenset({"python", "runner", "persistent_worker", "oneshot_runner", "install_root", "model_root",
                        "model_package", "model_dir", "rule_path", "feature_cache", "cache_dir", "library",
                        "tasks_root", "inbox_root", "identity_cache", "ffprobe", "executable", "manifest_path", "receipt_path", "path", "app_root",
                        "resources_root", "model", "calibration", "routing", "persistent_writable_root", "open_set_thresholds"})
_MODEL_KEYS = frozenset({"install_root", "model_root", "model_package", "model_dir", "app_root", "resources_root", "model", "calibration", "routing", "open_set_thresholds"})
_CODE_KEYS = frozenset({"runner", "persistent_worker", "oneshot_runner", "rule_path"})
_SECTIONS = frozenset({"schema_version", "product_name", "product_short_name", "product_version", "display_version",
                       "runtime", "video", "deepfake", "general_aigc", "provenance", "identity", "source",
                       "pair_compare", "case_libraries", "decision", "frontend", "llm"})


class ExecutionProfileError(ValueError):
    pass


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ExecutionProfileError("asset_changed_during_digest")
    return digest.hexdigest()


def _absolute(value) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or not Path(value).is_absolute():
        raise ExecutionProfileError("explicit_absolute_path_required")
    text = str(value)
    if "{" in text or "}" in text or "%" in text or "$" in text or text.startswith("~"):
        raise ExecutionProfileError("unresolved_path_or_environment_token")
    return Path(value).absolute()


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _safe_path(value, context, *, required_root: Path | None = None) -> Path:
    path = _absolute(value)
    resolved = path.resolve(strict=False)
    if context["core"].within(path, context["roots"]) or context["core"].within(resolved, context["roots"]):
        raise ExecutionProfileError("protected_path_or_resolved_target")
    if required_root is not None and (not _within(path, required_root) or not _within(resolved, required_root.resolve())):
        raise ExecutionProfileError("path_outside_independent_root")
    return path


def _walk_values(value, prefix=()):
    yield prefix, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_values(child, (*prefix, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_values(child, (*prefix, str(index)))


def is_independent_profile(config: Mapping) -> bool:
    if "execution_profile" not in config:
        return False
    if not isinstance(config["execution_profile"], Mapping) or config["execution_profile"].get("schema_version") != PROFILE_SCHEMA:
        raise ExecutionProfileError("unsupported_execution_profile_schema")
    return True


def trusted_policy_authority(environ: Mapping[str, str] | None = None) -> PolicyAuthority:
    """Trust only the launcher-owned authority setting, never a data/profile field."""
    source = os.environ if environ is None else environ
    value = source.get("FSTS_EXCLUSION_POLICY_PATH")
    if not value:
        raise ExecutionProfileError("launcher_exclusion_policy_path_required")
    return PolicyAuthority(_absolute(value))


def isolated_environment(config: Mapping, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment; never log its values."""
    if not is_independent_profile(config):
        raise ExecutionProfileError("independent_profile_required")
    source = os.environ if environ is None else environ
    result = {str(k): str(v) for k, v in source.items()
              if not _SECRET.search(str(k)) and not str(k).upper().startswith("AIGC_LLM_")
              and str(k).upper() not in {"FSTS_CONFIG", "FSTS_CONFIG_PATH", "FSTS_DATA_ROOT", "FSTS_OPEN_SET_THRESHOLDS", "AIGC_MEDIA_TOOLS", "PYTHONPATH", "PYTHONHOME"}
              and not ("://" in str(v) and "@" in str(v).split("://", 1)[1].split("/", 1)[0])}
    result["FSTS_DATA_ROOT"] = str(_absolute(config["execution_profile"]["independent_root"]))
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    cache_root = Path(result["FSTS_DATA_ROOT"]) / "cache" / "model_runtime"
    for key in list(result):
        if key.upper() in {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"}:
            del result[key]
    result.update(HF_HOME=str(cache_root / "huggingface"), HF_HUB_CACHE=str(cache_root / "huggingface" / "hub"),
                  TORCH_HOME=str(cache_root / "torch"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    thresholds = config.get("decision", {}).get("open_set_thresholds")
    if thresholds:
        result["FSTS_OPEN_SET_THRESHOLDS"] = str(_absolute(thresholds))
    if config.get("general_aigc", {}).get("enabled") is True:
        result["AIGC_MEDIA_TOOLS"] = str(_absolute(config["execution_profile"]["asset_root"]) / "ffmpeg")
    return result


def llm_load_overrides(config: Mapping) -> dict[str, Any]:
    """Apply inside load_llm_settings before reading any external profile."""
    if not is_independent_profile(config):
        return {}
    return {"profile_path": None, "environ": {}, "credentials": {"base_url": "", "model": "", "api_key": ""}}


def build_independent_profile(base: Mapping, *, product_root: str | Path, asset_root: str | Path,
                              independent_root: str | Path, interpreter: str | Path) -> dict:
    """Project approved defaults into explicit paths, with empty libraries.

    Source code is referenced in product_root, existing weight/runtime assets in
    asset_root. Nothing is copied or hashed here and no asset is marked verified.
    """
    product, assets, independent, python = map(_absolute, (product_root, asset_root, independent_root, interpreter))
    result = {key: deepcopy(value) for key, value in base.items() if key in _SECTIONS}
    def project(value, key="", ancestors=()):
        if isinstance(value, dict):
            return {k: project(v, k, (*ancestors, key)) for k, v in value.items()
                    if not _SECRET.search(k) and k not in {"persistent_environment", "profile_path", "llm_profile", "evidence"}}
        if isinstance(value, list):
            return [project(v, key, ancestors) for v in value]
        if isinstance(value, str) and key in _PATH_KEYS and value and "known_sha256" not in ancestors:
            target_root = assets if key in _MODEL_KEYS or key in {"python", "ffprobe"} else product
            text = value.replace("{CURRENT_PYTHON}", str(python)).replace("{APP_ROOT}", str(target_root)).replace("{CONFIG_ROOT}", str(product)).replace("{DATA_ROOT}", str(independent))
            path = Path(text)
            return str(path if path.is_absolute() else target_root / path)
        return value
    result = project(result)
    runtime = result.setdefault("runtime", {})
    runtime.update(tasks_root=str(independent / "outputs" / "tasks"), inbox_root=str(independent / "inbox"),
                   identity_cache=str(independent / "cache" / "identity"), persistent_workers_enabled=False)
    result.setdefault("identity", {}).update(library=str(independent / "libraries" / "identity"))
    result.setdefault("source", {}).update(library=str(independent / "libraries" / "source"), feature_cache=str(independent / "cache" / "source"))
    result.setdefault("pair_compare", {}).update(cache_dir=str(independent / "cache" / "pair"))
    if "deepfake_detection_v2" in str(result.get("deepfake", {}).get("persistent_worker", "")):
        result["deepfake"]["app_root"] = str(assets)
    result.setdefault("case_libraries", {})["enabled"] = False
    result.setdefault("frontend", {}).update(address="127.0.0.1", port=8844)
    result.setdefault("llm", {}).update(enabled=False, decision_mode="rule", upload_scope="text_only", judge_send_images=False,
                                         base_url="", model="", api_key="")
    for name in ("video", "deepfake", "general_aigc", "identity", "source", "pair_compare"):
        if isinstance(result.get(name), dict):
            result[name]["enabled"] = False
    result["execution_profile"] = {"schema_version": PROFILE_SCHEMA, "mode": "engineering_only",
        "independent_root": str(independent), "product_root": str(product), "asset_root": str(assets),
        "interpreter": str(python), "external_profiles_allowed": False, "environment_credentials_allowed": False,
        "cloud_allowed": False, "query_admission": None,
        "reference_admission": {"identity": None, "source": None}, "model_bindings": [],
        "asset_verification": "not_performed", "base_config_digest": _json_digest(base),
        "scope": "independent_engineering_and_explicitly_admitted_inputs_only"}
    return result


def _verify_bindings(bindings, context, allowed_roots) -> list[dict]:
    if not isinstance(bindings, list):
        raise ExecutionProfileError("explicit_frozen_model_bindings_required")
    records = []
    seen = set()
    for binding in bindings:
        if not isinstance(binding, dict) or binding.get("verification") != "verified":
            raise ExecutionProfileError("model_binding_not_verified")
        path = _safe_path(binding.get("path"), context)
        if not any(_within(path.resolve(), root.resolve()) for root in allowed_roots):
            raise ExecutionProfileError("model_binding_outside_declared_asset_or_product_root")
        if path.resolve() in seen:
            raise ExecutionProfileError("duplicate_model_asset_binding")
        seen.add(path.resolve())
        expected = binding.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ExecutionProfileError("model_digest_missing")
        actual = _file_digest(path)
        if actual != expected.lower() or actual in context["hashes"]:
            raise ExecutionProfileError("model_digest_mismatch_or_protected")
        records.append({"role": binding.get("role", "unrecorded"), "path": str(path), "sha256": actual, "verification": "verified"})
    return records


_BUFFALO_FILES = ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")


def _required_assets(config: Mapping, read_metadata) -> list[dict]:
    """Enumerate only known entry-point assets, never discover by directory scan."""
    result = []
    def add(module, role, path, expected=None):
        result.append({"module": module, "role": role, "path": str(_absolute(path)), "declared_sha256": expected})
    def buffalo(module, root):
        for name in _BUFFALO_FILES:
            add(module, "face_detection_and_recognition_weights", _absolute(root) / "models" / "buffalo_l" / name)
    def package(module, root, *, gend=False):
        root = _absolute(root)
        manifest_path = root / "manifest.json"
        add(module, "model_manifest", manifest_path)
        manifest = read_metadata(manifest_path)
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ExecutionProfileError("model_manifest_checkpoint_missing")
        checkpoint_sha = manifest.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha):
            raise ExecutionProfileError("model_manifest_checkpoint_digest_missing")
        filenames = {checkpoint: checkpoint_sha}
        if gend:
            files = manifest.get("files_sha256")
            if not isinstance(files, dict):
                raise ExecutionProfileError("gend_file_manifest_required")
            filenames.update(files)
            for name in ("config.json", "clip_config/config.json", "clip_config/preprocessor_config.json", "clip_config/vocab.json", "clip_config/merges.txt"):
                filenames.setdefault(name, None)
        for name, expected in filenames.items():
            path = root / name
            if Path(name).is_absolute() or not _within(path.resolve(), root.resolve()):
                raise ExecutionProfileError("model_manifest_path_escape")
            add(module, "weights_or_preprocessing_configuration", path, expected)
        add(module, "calibration", root / "calibration.json")
    general = config.get("general_aigc", {})
    if general.get("enabled") is True:
        package("general_aigc", general.get("model_package"))
        crops = general.get("face_crops")
        if isinstance(crops, Mapping) and crops.get("enabled", True):
            buffalo("face_crops", crops.get("model_root"))
            branch = general.get("face_branch", {})
            backend = branch.get("backend", "cf384")
            if backend == "gend":
                package("face_branch", branch.get("model_package"), gend=True)
            elif backend != "cf384":
                raise ExecutionProfileError("unsupported_frozen_face_backend")
    deepfake = config.get("deepfake", {})
    if deepfake.get("enabled") is True:
        worker = str(deepfake.get("persistent_worker", ""))
        runner = str(deepfake.get("oneshot_runner", deepfake.get("runner", "")))
        if "deepfake_detection_v2" in worker or "deepfake_detection_v2" in runner:
            app = _absolute(deepfake.get("app_root"))
            folder = app / "packaging_v0.4/resources/detectors_v2"
            for role, name in (("weights", "communityforensics_vit/model.onnx"), ("calibration_and_preprocessing", "communityforensics_vit/calibration.json"), ("routing", "routing.json")):
                add("deepfake_v2", role, folder / name)
        elif "deepfake_detection" not in worker and "deepfake_detection" not in runner:
            raise ExecutionProfileError("unregistered_deepfake_entry_point")
        frozen = _absolute(deepfake.get("install_root"))
        for name in ("深度伪造识别系统.exe", "models/spatial_mobilenet_finetuned_gpu/spatial_mobilenet_finetuned_gpu.pt",
                     "models/spatial_mobilenet_finetuned_gpu/metadata.json", "models/spatial_mobilenet_finetuned_gpu/model_file_sha256.json",
                     "models/frequency_stat_formal/frequency_stat_detector.joblib", "models/frequency_stat_formal/metadata.json",
                     "models/fusion_formal/fusion_config.json", "config/runtime_pipeline.json"):
            add("deepfake_frozen", "frozen_runtime_weights_or_configuration", frozen / name)
    identity = config.get("identity", {})
    if identity.get("enabled") is True:
        buffalo("identity", identity.get("model_root"))
    video = config.get("video", {})
    if video.get("enabled") is True and video.get("face_detector", "insightface") == "insightface":
        buffalo("video_frame_selection", video.get("model_root"))
    pair = config.get("pair_compare", {})
    if pair.get("enabled") is True:
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            add("pair_compare", "weights", _absolute(pair.get("model_dir")) / name)
        add("pair_compare", "rule_configuration", pair.get("rule_path"))
    thresholds = config.get("decision", {}).get("open_set_thresholds")
    if thresholds:
        add("open_set_decision", "frozen_thresholds", thresholds)
    return result


def collect_asset_bindings(config: Mapping, *, authority: PolicyAuthority) -> list[dict]:
    """Observe hashes for the explicit known asset closure without loading models.

    This establishes file identity, not model quality, license or code-release
    approval. The caller must still freeze the code release before inference.
    """
    context = authority.read()
    observed = {}
    def observe(path):
        path = _safe_path(path, context)
        resolved = path.resolve(strict=True)
        digest = _file_digest(path)
        if digest in context["hashes"]:
            raise ExecutionProfileError("protected_model_asset_digest")
        observed[resolved] = {"path": str(path), "sha256": digest, "verification": "verified", "role": "required_runtime_asset"}
        return path
    def metadata(path):
        checked = observe(path)
        data = checked.read_bytes()
        if hashlib.sha256(data).hexdigest() != observed[checked.resolve()]["sha256"]:
            raise ExecutionProfileError("model_manifest_changed_during_read")
        return json.loads(data.decode("utf-8-sig"))
    expected = _required_assets(config, metadata)
    for item in expected:
        path = observe(item["path"])
        if item["declared_sha256"] and observed[path.resolve()]["sha256"] != item["declared_sha256"].lower():
            raise ExecutionProfileError("model_manifest_declared_digest_mismatch")
    return list(observed.values())


def _verify_asset_coverage(config, records, context):
    by_path = {Path(item["path"]).resolve(): item for item in records}
    def metadata(path):
        path = _safe_path(path, context)
        binding = by_path.get(path.resolve())
        if binding is None:
            raise ExecutionProfileError("required_model_asset_binding_missing:model_manifest")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise ExecutionProfileError("model_manifest_changed_during_read")
        return json.loads(data.decode("utf-8-sig"))
    expected = _required_assets(config, metadata)
    for item in expected:
        binding = by_path.get(_safe_path(item["path"], context).resolve())
        if binding is None:
            raise ExecutionProfileError("required_model_asset_binding_missing:" + item["module"] + ":" + item["role"])
        if item["declared_sha256"] and binding["sha256"] != item["declared_sha256"].lower():
            raise ExecutionProfileError("model_manifest_declared_digest_mismatch")
    return expected


def _required_runtime_code(config: Mapping) -> list[dict]:
    records = []
    def section_paths(name, section):
        for key in ("python", "runner", "persistent_worker", "oneshot_runner"):
            if section.get(key):
                records.append({"module": name, "role": "interpreter" if key == "python" else "code_entrypoint", "path": str(_absolute(section[key]))})
    for name in ("video", "deepfake", "general_aigc", "identity", "source", "pair_compare"):
        section = config.get(name, {})
        if section.get("enabled") is True:
            section_paths(name, section)
    general = config.get("general_aigc", {})
    crops = general.get("face_crops")
    if general.get("enabled") is True and isinstance(crops, Mapping) and crops.get("enabled", True):
        section_paths("face_crops", crops)
    deepfake = config.get("deepfake", {})
    if deepfake.get("enabled") is True and any("deepfake_detection_v2" in str(deepfake.get(key, "")) for key in ("persistent_worker", "oneshot_runner")):
        app = _absolute(deepfake.get("app_root"))
        for name in ("persistent_worker.py", "run_deepfake_detection.py", "pyinstaller_pyz_runtime.py"):
            records.append({"module": "deepfake_frozen_proxy", "role": "external_frozen_adapter_code", "path": str(app / "modules/deepfake_detection" / name)})
        records.append({"module": "deepfake_frozen_proxy", "role": "interpreter", "path": str(app / "packaging_v0.3/resources/portable_python/python.exe")})
    provenance = config.get("provenance", {})
    if provenance.get("enabled") is True and provenance.get("ffprobe"):
        records.append({"module": "provenance", "role": "metadata_executable", "path": str(_absolute(provenance["ffprobe"]))})
    if general.get("enabled") is True and any("general_aigc" in str(general.get(key, "")) for key in ("runner", "persistent_worker")):
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            records.append({"module": "general_aigc", "role": "media_decoder_executable", "path": str(_absolute(config["execution_profile"]["asset_root"]) / "ffmpeg" / name)})
    return records


def collect_runtime_code_bindings(config: Mapping, *, authority: PolicyAuthority) -> list[dict]:
    """Hash explicit entrypoints, interpreter executables and frozen proxy code.

    This is not a claim that installed third-party packages or an entire Python
    environment are reproducibly frozen. The source release is bound separately.
    """
    context = authority.read()
    records = {}
    for item in _required_runtime_code(config):
        path = _safe_path(item["path"], context)
        digest = _file_digest(path)
        if digest in context["hashes"]:
            raise ExecutionProfileError("protected_runtime_code_digest")
        records[path.resolve()] = {**item, "sha256": digest, "verification": "verified"}
    return list(records.values())


def _verify_runtime_code(config, records):
    by_path = {Path(item["path"]).resolve(): item for item in records}
    expected = _required_runtime_code(config)
    for item in expected:
        if Path(item["path"]).resolve() not in by_path:
            raise ExecutionProfileError("required_runtime_code_binding_missing:" + item["module"] + ":" + item["role"])
    return expected


def audit_execution_profile(config: Mapping, *, authority: PolicyAuthority,
                            config_path: str | Path | None = None, for_inference: bool = False) -> dict:
    if not is_independent_profile(config):
        raise ExecutionProfileError("independent_profile_required")
    context = authority.read()
    profile = config["execution_profile"]
    def checked(value, field, *, required_root=None):
        try:
            return _safe_path(value, context, required_root=required_root)
        except ExecutionProfileError as exc:
            raise ExecutionProfileError(str(exc) + ":field=" + field) from exc
        except OSError as exc:
            raise ExecutionProfileError("path_resolution_failed:field=" + field + ":error_type=" + type(exc).__name__) from exc
    independent = checked(profile.get("independent_root"), "execution_profile.independent_root")
    for key in ("product_root", "asset_root", "interpreter"):
        checked(profile.get(key), "execution_profile." + key)
    if config_path is not None:
        checked(config_path, "config_path", required_root=independent)
    if profile.get("mode") not in {"engineering_only", "independent_inference"}:
        raise ExecutionProfileError("unsupported_independent_mode")
    for key in ("external_profiles_allowed", "environment_credentials_allowed", "cloud_allowed"):
        if profile.get(key) is not False:
            raise ExecutionProfileError("external_profile_or_cloud_inheritance_forbidden")
    for keys, value in _walk_values(config):
        key = keys[-1] if keys else ""
        if _SECRET.search(key) and value not in (None, "", False):
            raise ExecutionProfileError("credential_field_forbidden:" + ".".join(keys))
        if key in {"persistent_environment", "profile_path", "llm_profile"} and value:
            raise ExecutionProfileError("external_profile_override_forbidden")
        if key in {"persistent_worker_args", "extra_args", "worker_executable_args", "worker_executable"} and value:
            raise ExecutionProfileError("unregistered_worker_argument_override")
        if key == "python_args" and (not isinstance(value, list) or any(arg not in {"-B", "-E", "-S", "-s", "-u"} for arg in value)):
            raise ExecutionProfileError("unregistered_python_argument_override")
        if key in _PATH_KEYS and value and "known_sha256" not in keys:
            path = checked(value, ".".join(keys))
            if key in {"persistent_writable_root", "feature_cache", "cache_dir", "tasks_root", "inbox_root", "identity_cache"}:
                checked(value, ".".join(keys), required_root=independent)
            if key in _CODE_KEYS and not _within(path.resolve(), _absolute(profile["product_root"]).resolve()):
                raise ExecutionProfileError("code_path_outside_product_root:field=" + ".".join(keys))
            if key in _MODEL_KEYS and not _within(path.resolve(), _absolute(profile["asset_root"]).resolve()):
                raise ExecutionProfileError("asset_path_outside_declared_asset_root:field=" + ".".join(keys))
    for section, key in (("runtime", "tasks_root"), ("runtime", "inbox_root"), ("runtime", "identity_cache"),
                         ("source", "feature_cache"), ("pair_compare", "cache_dir")):
        checked(config.get(section, {}).get(key), section + "." + key, required_root=independent)
    llm = config.get("llm", {})
    if llm.get("enabled") is not False or llm.get("decision_mode") != "rule" or llm.get("upload_scope") != "text_only" or llm.get("judge_send_images") is not False:
        raise ExecutionProfileError("independent_cloud_must_be_disabled")
    if any(llm.get(key) for key in ("api_key", "base_url", "model")):
        raise ExecutionProfileError("independent_cloud_credentials_must_be_empty")
    if config.get("case_libraries", {}).get("enabled") is not False:
        raise ExecutionProfileError("unregistered_case_libraries_forbidden")
    refs = profile.get("reference_admission", {})
    if not isinstance(refs, Mapping) or any(refs.get(kind) is not None and not isinstance(refs.get(kind), Mapping) for kind in ("identity", "source")):
        raise ExecutionProfileError("invalid_reference_admission_binding")
    for kind in ("identity", "source"):
        library = checked(config.get(kind, {}).get("library"), kind + ".library", required_root=independent)
        if not refs.get(kind) and library.exists() and any(library.iterdir()):
            raise ExecutionProfileError("nonempty_library_requires_explicit_admission:" + kind)
    mode = profile["mode"]
    if mode == "engineering_only":
        if config.get("runtime", {}).get("persistent_workers_enabled") is not False:
            raise ExecutionProfileError("engineering_profile_cannot_autostart_models")
        if any(config.get(name, {}).get("enabled") is not False for name in ("video", "deepfake", "general_aigc", "identity", "source", "pair_compare")):
            raise ExecutionProfileError("engineering_profile_cannot_enable_inference")
    model_records = []
    coverage = []
    runtime_code = []
    runtime_coverage = []
    if for_inference:
        if mode != "independent_inference" or not isinstance(profile.get("query_admission"), dict):
            raise ExecutionProfileError("explicit_inference_profile_and_query_admission_required")
        allowed_roots = [_absolute(profile["asset_root"]), _absolute(profile["product_root"])]
        model_records = _verify_bindings(profile.get("model_bindings"), context, allowed_roots)
        coverage = _verify_asset_coverage(config, model_records, context)
        runtime_code = _verify_bindings(profile.get("runtime_code_bindings", []), context, allowed_roots)
        runtime_coverage = _verify_runtime_code(config, runtime_code)
    return {"schema_version": "execution_profile.audit.v1", "status": "passed", "config_digest": _json_digest(config),
            "policy_binding": context["bindings"], "mode": mode,
            "model_bindings": model_records, "model_verification": "verified_required_asset_closure" if coverage else "no_enabled_model_assets" if for_inference else "not_performed",
            "expected_asset_paths": coverage,
            "runtime_code_bindings": runtime_code, "expected_runtime_code_paths": runtime_coverage,
            "runtime_code_binding_scope": "explicit_entrypoints_interpreter_executables_and_frozen_adapter_only_not_full_dependency_environment",
            "media_decoded": False, "models_loaded": False, "directory_dataset_discovery": False}


def assert_reference_selection(config: Mapping, *, identity_library=None, source_library=None) -> None:
    for kind, value in (("identity", identity_library), ("source", source_library)):
        if value is not None and _absolute(value).resolve() != _absolute(config[kind]["library"]).resolve():
            raise ExecutionProfileError("unregistered_reference_override:" + kind)


def open_admitted_manifest(binding: Mapping, *, authority: PolicyAuthority, expected_frozen_release=None):
    """Read explicit metadata across a second gate check; reject selection races."""
    kwargs = {"expected_purpose": binding["purpose"], "expected_frozen_release": expected_frozen_release}
    AdmittedDataset(authority, binding["manifest_path"], binding["receipt_path"], **kwargs)
    raw = Path(binding["manifest_path"]).read_bytes()
    dataset = AdmittedDataset(authority, binding["manifest_path"], binding["receipt_path"], **kwargs)
    if Path(binding["manifest_path"]).read_bytes() != raw:
        raise ExecutionProfileError("manifest_changed_during_metadata_selection")
    return dataset, json.loads(raw.decode("utf-8-sig"))


def admit_query_snapshot(config: Mapping, query_path: str | Path, *, authority: PolicyAuthority,
                         expected_frozen_release: dict | None = None, maximum_bytes=64 * 1024 * 1024):
    """Called before copy/decoder/model/cloud; return immutable admitted bytes."""
    audit_execution_profile(config, authority=authority, for_inference=True)
    binding = config["execution_profile"]["query_admission"]
    dataset, manifest = open_admitted_manifest(binding, authority=authority, expected_frozen_release=expected_frozen_release)
    selected = _absolute(query_path).resolve(strict=True)
    matches = [(i, item) for i, item in enumerate(manifest["items"]) if _absolute(item["path"]).resolve(strict=True) == selected]
    if len(matches) != 1:
        raise ExecutionProfileError("query_must_match_one_explicit_manifest_item")
    index, item = matches[0]
    snapshot = dataset.read_snapshot(index, maximum_bytes=maximum_bytes)
    if snapshot.sha256 != item["sha256"].lower():
        raise ExecutionProfileError("query_manifest_changed_during_selection")
    return snapshot


def admit_reference_library(config: Mapping, kind: str, *, authority: PolicyAuthority, expected_frozen_release=None) -> dict:
    """Verify the registered reference closure; enumeration only checks equality.

    Candidate inputs come exclusively from the admitted explicit manifest. The
    bounded configured-library walk rejects extra entries and all links; it does
    not discover or add candidate data and does not decode any file.
    """
    if kind not in {"identity", "source"}:
        raise ExecutionProfileError("invalid_reference_kind")
    audit_execution_profile(config, authority=authority)
    raw_root = _absolute(config[kind]["library"])
    if raw_root.is_symlink() or (hasattr(raw_root, "is_junction") and raw_root.is_junction()):
        raise ExecutionProfileError("reference_links_forbidden")
    root = raw_root.resolve(strict=True)
    binding = config["execution_profile"]["reference_admission"].get(kind)
    if not binding:
        if any(root.iterdir()):
            raise ExecutionProfileError("unregistered_reference_files")
        return {"status": "empty", "count": 0}
    dataset, manifest = open_admitted_manifest(binding, authority=authority, expected_frozen_release=expected_frozen_release)
    declared = {_absolute(item["path"]).resolve(strict=True) for item in manifest["items"]}
    if any(not _within(path, root) for path in declared) or len(declared) != len(manifest["items"]):
        raise ExecutionProfileError("reference_manifest_path_closure_invalid")
    actual = set()
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(folder) / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ExecutionProfileError("reference_links_forbidden")
        actual.update((Path(folder) / name).resolve(strict=True) for name in files)
    if actual != declared:
        raise ExecutionProfileError("reference_library_has_unregistered_or_missing_files")
    # Revalidate after closure inspection without decoding or retaining bytes.
    AdmittedDataset(authority, binding["manifest_path"], binding["receipt_path"],
                    expected_purpose=binding["purpose"], expected_frozen_release=expected_frozen_release)
    return {"status": "admitted", "count": len(declared), "purpose": binding["purpose"],
            "scope": "manifest_and_path_closure_check_only", "immutable_loader_integration_required": True}


def materialize_reference_snapshot(config: Mapping, kind: str, destination: str | Path, *,
                                   authority: PolicyAuthority, expected_frozen_release=None,
                                   exclude_sha256=()) -> dict:
    """Copy only gate-returned immutable bytes into a new attempt directory.

    Files are marked read-only to reduce accidental mutation, not to claim a
    security boundary against another process with the same account rights.
    No old index/cache is copied. Partial attempts are retained on failure and
    the same destination can never be reused or overwritten.
    """
    from artifact_io import atomic_write_bytes
    closure = admit_reference_library(config, kind, authority=authority, expected_frozen_release=expected_frozen_release)
    context = authority.read()
    independent = _safe_path(config["execution_profile"]["independent_root"], context)
    target = _safe_path(destination, context, required_root=independent)
    original = _absolute(config[kind]["library"]).resolve(strict=True)
    if _within(target.resolve(), original):
        raise ExecutionProfileError("snapshot_cannot_be_inside_original_library")
    if target.exists():
        raise ExecutionProfileError("snapshot_destination_must_be_new")
    rejected = {str(value).lower() for value in exclude_sha256}
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in rejected):
        raise ExecutionProfileError("invalid_excluded_query_digest")
    binding = config["execution_profile"]["reference_admission"].get(kind)
    manifest_digest = None
    records = []
    target.mkdir(parents=True, exist_ok=False)
    if binding:
        dataset, manifest = open_admitted_manifest(binding, authority=authority, expected_frozen_release=expected_frozen_release)
        manifest_bytes = Path(binding["manifest_path"]).read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if json.loads(manifest_bytes.decode("utf-8-sig")) != manifest:
            raise ExecutionProfileError("reference_manifest_changed_during_snapshot")
        for index, item in enumerate(manifest["items"]):
            source = _absolute(item["path"]).resolve(strict=True)
            if not _within(source, original):
                raise ExecutionProfileError("reference_manifest_path_closure_invalid")
            relative = source.relative_to(original)
            snapshot = dataset.read_snapshot(index)
            if snapshot.sha256 != item["sha256"].lower() or hashlib.sha256(snapshot.data).hexdigest() != snapshot.sha256:
                raise ExecutionProfileError("reference_snapshot_binding_changed")
            if snapshot.sha256 in rejected:
                raise ExecutionProfileError("reference_matches_excluded_query_digest")
            output = _safe_path(target / relative, context, required_root=target)
            atomic_write_bytes(output, snapshot.data, role="admitted_reference_snapshot", overwrite=False)
            output.chmod(stat.S_IREAD)
            records.append({"path": str(output), "relative_path": relative.as_posix(), "sha256": snapshot.sha256,
                            "source_dataset_id": getattr(snapshot, "source_dataset_id", item.get("source_dataset_id")),
                            "purpose": binding["purpose"], "size_bytes": len(snapshot.data)})
        if hashlib.sha256(Path(binding["manifest_path"]).read_bytes()).hexdigest() != manifest_digest:
            raise ExecutionProfileError("reference_manifest_changed_during_snapshot")
        # Re-read the current maintained policy after the final byte snapshot.
        AdmittedDataset(authority, binding["manifest_path"], binding["receipt_path"],
                        expected_purpose=binding["purpose"], expected_frozen_release=expected_frozen_release)
    latest = authority.read()
    if latest["bindings"] != context["bindings"]:
        raise ExecutionProfileError("policy_changed_during_reference_snapshot")
    return {"schema_version": "reference_snapshot.v1", "status": "admitted" if records else "empty", "kind": kind,
            "library_path": str(target), "count": len(records), "files": records,
            "source_manifest_sha256": manifest_digest, "policy_binding": latest["bindings"],
            "closure": closure, "excluded_query_sha256": sorted(rejected), "read_only_files": True,
            "old_cache_copied": False, "external_bypass_prevention_claimed": False}


def execution_binding(config: Mapping, *, input_digest: str | None, code_bindings=None,
                       model_bindings=None, attempt_id: str | None = None) -> dict:
    if input_digest is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", input_digest):
        raise ExecutionProfileError("invalid_input_digest")
    return {"schema_version": "execution_binding.v1", "attempt_id": attempt_id or uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(), "config_digest": _json_digest(config),
            "input_digest": input_digest, "input_verification": "bound_digest" if input_digest else "not_provided",
            "code_bindings": list(code_bindings or []), "model_bindings": list(model_bindings or []),
            "model_verification": "see_individual_bindings" if model_bindings else "not_performed"}
