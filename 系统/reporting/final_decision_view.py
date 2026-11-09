"""Render a saved canonical decision without deriving another decision."""
from collections.abc import Mapping

from contracts.task_contract_v4 import VALID_BANDS

DEFAULT_LABELS = {"none": "未见明显 AIGC 痕迹", "review": "建议人工复核",
                  "obvious": "明显 AIGC 痕迹", "undetermined": "无法完成判断"}


def saved_decision_view(report: Mapping) -> dict | None:
    analysis = report.get("analysis")
    if not isinstance(analysis, Mapping):
        return None
    final = analysis.get("final_decision")
    if not isinstance(final, Mapping) or final.get("final_band") not in VALID_BANDS:
        return None
    configured = analysis.get("band_labels") or {}
    labels = {**DEFAULT_LABELS, **(configured if isinstance(configured, Mapping) else {})}
    return {"band": final["final_band"], "label": labels[final["final_band"]],
            "reason": final.get("reason", ""), "score": final.get("score"), "source": final.get("source", "unrecorded"),
            "rule_version": final.get("rule_version", "unrecorded"), "evidence_digest": final.get("evidence_digest", "unrecorded")}
