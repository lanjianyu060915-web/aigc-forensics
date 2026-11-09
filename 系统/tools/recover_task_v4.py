"""Explicit receipt/writer/new-task recovery; no batch resumption.

Cloud writer retry additionally requires the original frozen prompt SHA and a
separate cloud authorization flag. Independent profiles can never call cloud.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("receipt", "writer", "new-task"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorize-action", action="store_true", help="Current explicit authorization for this single action")
    parser.add_argument("--authorize-cloud", action="store_true", help="Current explicit authorization for the original text-only cloud endpoint")
    parser.add_argument("--writer-prompt-sha256", help="Writer prompt digest from the frozen release/authorization")
    parser.add_argument("--llm-profile", type=Path, help="Explicit original authorized LLM settings profile, if used")
    parser.add_argument("--stop-path", type=Path, action="append", default=[])
    parser.add_argument("--query", type=Path, help="Explicit current admitted input for one new task")
    args = parser.parse_args(argv)
    if not args.authorize_action:
        parser.error("--authorize-action is required; prior batch authorization is insufficient")
    from reporting.recovery_actions_v4 import write_recovery_receipt, retry_report_writer, rerun_as_new_task
    shared = {"authorized": True, "stop_paths": args.stop_path}
    if args.action == "new-task":
        if args.query is None:
            parser.error("new-task requires --query")
        result = rerun_as_new_task(args.task_dir, args.query, config_path=args.config, **shared)
    else:
        from reporting.evidence_figures import local_file
        from reporting.llm_client import load_llm_settings
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        source_path = local_file(args.task_dir.resolve(strict=True), "analysis_source_report.json")
        if source_path is None:
            parser.error("verified task source report is unavailable")
        report = json.loads(source_path.read_text(encoding="utf-8-sig"))
        settings = load_llm_settings(config, profile_path=args.llm_profile)
        if args.action == "receipt":
            result = write_recovery_receipt(args.task_dir, report, config=config, settings=settings, **shared)
        else:
            if not args.authorize_cloud or not args.writer_prompt_sha256:
                parser.error("writer requires --authorize-cloud and --writer-prompt-sha256")
            result = retry_report_writer(args.task_dir, report, config=config, settings=settings,
                cloud_authorized=True, expected_writer_prompt_sha256=args.writer_prompt_sha256, **shared)
    # Only action/status/new attempt paths, never payloads or credentials.
    summary = result.get("result", result)
    print(json.dumps({"action": args.action, "status": summary.get("status"), "task_dir": result.get("task_dir"),
                      "attempt_id": summary.get("attempt_id"), "cloud_calls": summary.get("cloud_calls")}, ensure_ascii=False))
    return 0 if summary.get("status") not in {"failed", "error", "stopped"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # No raw exception payloads: vendor/client errors may contain secrets.
        from reporting.recovery_actions_v4 import RecoveryActionError
        message = str(exc) if isinstance(exc, RecoveryActionError) else type(exc).__name__
        print("Recovery rejected: " + message, file=sys.stderr)
        raise SystemExit(2)
