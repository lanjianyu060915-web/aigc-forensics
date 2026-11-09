"""逐脸身份候选的共用展示数据；不参与检测档位和证据矩阵。"""

from collections.abc import Mapping
from typing import Any

FACE_NOTICE = "系统不判断哪张脸被改动，每张脸的候选都只是候选。"
FACE_LIMITATION = "逐脸候选沿用现有开放集阈值，该阈值尚未在逐脸场景专门验证。"


def face_rows(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    faces = identity.get("faces")
    if not isinstance(faces, list) or len(faces) <= 1:
        return []
    return sorted((dict(face) for face in faces if isinstance(face, Mapping)),
                  key=lambda face: (not face.get("is_main_face"), face.get("area_rank", 999)))


def face_table_rows(identity: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for face in face_rows(identity):
        label = f"脸 {face.get('face_index')}" + ("（主脸）" if face.get("is_main_face") else "")
        decision = face.get("open_set") if isinstance(face.get("open_set"), Mapping) else {}
        candidates = face.get("candidates") or []
        for candidate in candidates[:3] or [{}]:
            rows.append([label, candidate.get("rank"), candidate.get("identity_name") or candidate.get("identity_id"),
                         candidate.get("score"), decision.get("state_label") or "未完成"])
    return rows


def face_notes(identity: Mapping[str, Any]) -> list[str]:
    search = identity.get("face_search") if isinstance(identity.get("face_search"), Mapping) else {}
    return list(dict.fromkeys([FACE_NOTICE, FACE_LIMITATION, *search.get("limitations", [])]))
