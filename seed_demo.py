"""
Seed 코퍼스 생성기: 실제 파이프라인으로 다양한 방학 계획 대화를 돌려
Common Ground / LLM Ground를 추적하고, 결과를 sessions.db에 저장한다.
(REAL LLM, 주입 없음 — 자연 발생 LG 관찰. 저장되므로 corpus_export.py로 바로 주석 템플릿화 가능.)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv
load_dotenv(HERE / ".env", override=True)

from app.config import load_config
from app.database import init_db, get_connection, insert_state, now_iso
from app.graph import initial_state, run_message_graph

SCENARIOS = [
    {
        "session_id": "seed_job_01",
        "participant_id": "seed",
        "user_intent": "취업 준비 중심 방학 계획",
        "utterances": [
            "이번 방학엔 취업 준비에 집중하려고. 코딩테스트랑 자소서 준비를 하고 싶어.",
            "개발 직무로 갈 거고, 코테는 파이썬으로. 자소서는 아직 기업을 안 정했어.",
            "인턴도 몇 군데 지원해볼까 해. 근데 방학이 짧아서 다 하긴 빠듯할 것 같아.",
        ],
    },
    {
        "session_id": "seed_rest_01",
        "participant_id": "seed",
        "user_intent": "휴식 중심 방학 계획",
        "utterances": [
            "이번 방학은 좀 쉬고 싶어. 여행도 다니고 취미도 하면서.",
            "취미는 그림이랑 요리 정도. 여행은 부담 없이 국내로.",
            "공부는 최소한만 하려고. 그래도 영어는 감 안 잃을 정도로.",
        ],
    },
]

out: list[str] = []
def emit(s: str = "") -> None:
    print(s)
    out.append(s)


def persist_turn(conn, sid, turn, state):
    conn.execute(
        """INSERT OR REPLACE INTO turn_artifacts(session_id, turn, C_t, D_t, B_t, A_t, system_prompt_used, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (sid, turn, state.get("C_t", ""), state.get("D_t", ""), state.get("B_t", ""),
         json.dumps(state.get("A_t", {}), ensure_ascii=False),
         state.get("system_prompt_used", ""), now_iso()),
    )
    insert_state(conn, sid, turn, state)
    conn.execute("UPDATE sessions SET current_turn=? WHERE session_id=?", (turn, sid))
    conn.commit()


def run_scenario(cfg, conn, sc) -> None:
    sid = sc["session_id"]
    conn.execute(
        """INSERT OR REPLACE INTO sessions(session_id, participant_id, user_intent, started_at, config_snapshot, session_type)
           VALUES (?, ?, ?, ?, ?, 'gl')""",
        (sid, sc["participant_id"], sc["user_intent"], now_iso(),
         json.dumps({"model": cfg.get("model"), "seed": True}, ensure_ascii=False)),
    )
    conn.commit()
    state = initial_state(sid, sc["user_intent"])
    emit(f"\n########## SCENARIO {sid} : {sc['user_intent']} ##########")
    for u in sc["utterances"]:
        turn = int(state.get("current_turn", 0)) + 1
        state["current_turn"] = turn
        state["C_t"] = u
        state.setdefault("conversation_history", []).append(
            {"role": "user", "content": u, "turn": turn,
             "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        state = run_message_graph(state, cfg)
        persist_turn(conn, sid, turn, state)
        emit(f"\n===== TURN {turn} =====")
        emit(f"USER: {u}")
        emit(f"CG ({len(state.get('common_ground', []))}): "
             + " | ".join(i.get("content", "")[:50] for i in state.get("common_ground", [])))
        emit(f"LG ({len(state.get('llm_ground', []))}):")
        for i in state.get("llm_ground", []):
            emit(f"  - [{i.get('type', '?')}] {i.get('content', '')}")


def main():
    cfg = load_config()
    init_db()
    conn = get_connection()
    emit(f"model={cfg.get('model')}  (seed corpus 생성, DB 저장)")
    try:
        for sc in SCENARIOS:
            run_scenario(cfg, conn, sc)
    finally:
        conn.close()
    (HERE / "corpus" / "seed_corpus_output.txt").write_text("\n".join(out), encoding="utf-8")
    emit("\n[saved corpus/seed_corpus_output.txt + sessions.db]")


if __name__ == "__main__":
    main()
