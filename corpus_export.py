"""
코퍼스 주석 템플릿 exporter.
GroundLens 세션(sessions.db)의 각 턴 맥락을 gold 주석용 JSON 템플릿으로 내보낸다.
주석자는 시스템 출력과 무관하게 각 턴의 gold(Common Ground / LLM Ground)를 독립적으로 라벨한다.

사용법:
  python corpus_export.py <session_id> [<session_id> ...]
  python corpus_export.py --all [min_turns=3]
출력: GL/corpus/annotation_<sid8>.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "sessions.db"
OUT = ROOT / "corpus"
OUT.mkdir(exist_ok=True)

SUBTYPES = "구조적 결정 | 암묵적 전제 | 용어 해석 | 기타"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def export_session(c: sqlite3.Connection, sid: str) -> dict:
    arts = c.execute(
        "SELECT turn, C_t, D_t, A_t, B_t FROM turn_artifacts WHERE session_id=? ORDER BY turn",
        (sid,),
    ).fetchall()
    turns = []
    for a in arts:
        try:
            at = json.loads(a["A_t"]) if a["A_t"] else {}
        except Exception:
            at = {"_raw": a["A_t"]}
        bt = a["B_t"] or ""
        turns.append({
            "turn": a["turn"],
            "user_C_t": a["C_t"],
            "ai_D_t": a["D_t"],
            "shadow_probe_A_t": at,
            "extended_thinking_excerpt": bt[:300],
            "gold": {
                "common_ground": [{"content": "", "provenance_turn": None}],
                "llm_ground": [
                    {"content": "", "subtype": f"<{SUBTYPES}>",
                     "provenance_turn": None, "reused_later": False}
                ],
            },
        })
    return {
        "session_id": sid,
        "domain": "방학 계획",
        "annotation_guide": {
            "aspect1_supported": "전제가 응답 또는 추론 산출물에 의해 지지되는가",
            "aspect2_positive_evidence": "대응하는 사용자 positive evidence가 존재하는가 (있으면 CG, 없으면 LG)",
            "aspect3_bin": "Common Ground 또는 LLM Ground로 분류",
            "aspect4_subtype": f"LG일 경우 세부유형: {SUBTYPES}",
            "aspect5_provenance": "전제가 처음 등장한 턴 번호",
            "how_to": "각 턴 gold의 빈 슬롯을 채우고, 전제 수만큼 항목을 늘리세요. 시스템 출력과 독립적으로 라벨하세요.",
        },
        "turns": turns,
    }


def main() -> None:
    c = _conn()
    args = sys.argv[1:]
    if args and args[0] == "--all":
        min_turn = int(args[1]) if len(args) > 1 else 3
        rows = c.execute(
            "SELECT session_id, MAX(turn) m FROM groundlens_states GROUP BY session_id HAVING m>=?",
            (min_turn,),
        ).fetchall()
        sids = [r["session_id"] for r in rows]
    else:
        sids = args
    if not sids:
        print("usage: corpus_export.py <session_id ...> | --all [min_turns]")
        return
    for sid in sids:
        data = export_session(c, sid)
        if not data["turns"]:
            print(f"skip {sid}: no turns")
            continue
        p = OUT / f"annotation_{sid[:8]}.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p.name}  ({len(data['turns'])} turns)")


if __name__ == "__main__":
    main()
