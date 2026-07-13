from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import load_config
from app.database import (
    dump_schema,
    get_connection,
    init_db,
    insert_state,
    load_latest_state,
    now_iso,
)
from app.graph import initial_state, run_correction_graph, run_deletion_graph, run_message_graph
from app.ws import build_update_payload, manager, router as ws_router, set_event_loop

load_dotenv(override=False)

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

app = FastAPI(
    title="GroundLens v2",
    description="LangGraph-style GroundLens chatbot with correction-aware state.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(ws_router)


@app.on_event("startup")
async def startup() -> None:
    init_db()
    set_event_loop(asyncio.get_event_loop())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dev/schema")
def schema():
    return {"tables": dump_schema()}


@app.get("/api/config")
def config_public():
    cfg = load_config()
    return {
        "session_minutes": cfg.get("session_minutes", 20),
        "task_context": cfg.get("task_context", ""),
    }


class StartSessionRequest(BaseModel):
    participant_id: str
    user_intent: str
    paper_id: str | None = None
    session_type: str = "gl"  # 'gl' | 'baseline'


class StartSessionResponse(BaseModel):
    session_id: str


@app.post("/api/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    participant_id = req.participant_id.strip()
    user_intent = req.user_intent.strip()
    if not participant_id:
        raise HTTPException(status_code=400, detail="participant_id is required")
    if not user_intent:
        raise HTTPException(status_code=400, detail="user_intent is required")

    cfg = load_config()
    session_id = uuid.uuid4().hex
    state = initial_state(session_id, user_intent)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO sessions(
                session_id, participant_id, user_intent, paper_id,
                current_turn, current_stage, turn_in_stage, revision_count,
                session_type, started_at, config_snapshot
            )
            VALUES (?, ?, ?, ?, 0, 'S1', 0, 0, ?, ?, ?)
            """,
            (
                session_id,
                participant_id,
                user_intent,
                req.paper_id or "",
                req.session_type,
                now_iso(),
                json.dumps(cfg, ensure_ascii=False),
            ),
        )
        insert_state(conn, session_id, 0, state)
        conn.commit()
    finally:
        conn.close()
    return StartSessionResponse(session_id=session_id)


class MessageRequest(BaseModel):
    session_id: str
    user_message: str


class MessageResponse(BaseModel):
    assistant_message: str
    current_stage: str
    stage_gate_ready: bool


@app.post("/api/message", response_model=MessageResponse)
def post_message(req: MessageRequest):
    user_message = req.user_message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message is required")

    cfg = load_config()
    conn = get_connection()
    try:
        session = _get_session(conn, req.session_id)
        previous = load_latest_state(conn, req.session_id)
        if previous is None:
            previous = initial_state(req.session_id, session["user_intent"])

        state = dict(previous)
        turn = int(state.get("current_turn", 0)) + 1
        user_entry = {
            "role": "user",
            "content": user_message,
            "turn": turn,
            "timestamp": now_iso(),
        }
        state["current_turn"] = turn
        state["C_t"] = user_message
        state.setdefault("conversation_history", []).append(user_entry)

        next_state = run_message_graph(state, cfg)

        conn.execute(
            """
            INSERT INTO messages(session_id, turn, role, content, timestamp)
            VALUES (?, ?, 'user', ?, ?)
            """,
            (req.session_id, turn, user_message, user_entry["timestamp"]),
        )
        conn.execute(
            """
            INSERT INTO messages(
                session_id, turn, role, content, timestamp,
                system_prompt_used, extended_thinking
            )
            VALUES (?, ?, 'assistant', ?, ?, ?, ?)
            """,
            (
                req.session_id,
                turn,
                next_state.get("D_t", ""),
                now_iso(),
                next_state.get("system_prompt_used", ""),
                next_state.get("B_t", ""),
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO turn_artifacts(
                session_id, turn, C_t, D_t, B_t, A_t, system_prompt_used, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.session_id,
                turn,
                user_message,
                next_state.get("D_t", ""),
                next_state.get("B_t", ""),
                json.dumps(next_state.get("A_t", {}), ensure_ascii=False),
                next_state.get("system_prompt_used", ""),
                now_iso(),
            ),
        )
        conn.execute(
            """
            UPDATE sessions
            SET current_turn = ?, current_stage = ?, turn_in_stage = ?, revision_count = ?
            WHERE session_id = ?
            """,
            (
                turn,
                next_state.get("current_stage", "S1"),
                next_state.get("turn_in_stage", 0),
                next_state.get("revision_count", 0),
                req.session_id,
            ),
        )
        insert_state(conn, req.session_id, turn, next_state)
        conn.commit()
    finally:
        conn.close()

    manager.push(req.session_id, build_update_payload(previous, next_state))
    return MessageResponse(
        assistant_message=next_state.get("D_t", ""),
        current_stage=next_state.get("current_stage", "S1"),
        stage_gate_ready=bool(next_state.get("stage_gate_ready", False)),
    )


class CorrectionRequest(BaseModel):
    session_id: str
    target_item_id: str
    correction_text: str


@app.post("/api/ground/correct")
def correct_ground(req: CorrectionRequest):
    correction_text = req.correction_text.strip()
    if not correction_text:
        raise HTTPException(status_code=400, detail="correction_text is required")

    conn = get_connection()
    try:
        _get_session(conn, req.session_id)
        previous = load_latest_state(conn, req.session_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="GroundLens state not found")
        state = dict(previous)
        state["target_item_id"] = req.target_item_id
        state["correction_text"] = correction_text
        next_state = run_correction_graph(state)

        source_turn = next(
            (
                int(item.get("source_turn", next_state.get("current_turn", 0)))
                for item in previous.get("llm_ground", [])
                if item.get("id") == req.target_item_id
            ),
            int(next_state.get("current_turn", 0)),
        )
        conn.execute(
            """
            INSERT INTO corrections(
                session_id, target_item_id, correction_text,
                source_turn, corrected_at_turn, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                req.session_id,
                req.target_item_id,
                correction_text,
                source_turn,
                int(next_state.get("current_turn", 0)),
                now_iso(),
            ),
        )
        insert_state(conn, req.session_id, int(next_state.get("current_turn", 0)), next_state)
        conn.commit()
    finally:
        conn.close()

    manager.push(req.session_id, build_update_payload(previous, next_state))
    return _public_state(next_state)


class DeleteRequest(BaseModel):
    session_id: str
    item_id: str


@app.post("/api/ground/delete")
def delete_ground(req: DeleteRequest):
    conn = get_connection()
    try:
        _get_session(conn, req.session_id)
        previous = load_latest_state(conn, req.session_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="GroundLens state not found")
        state = dict(previous)
        state["target_item_id"] = req.item_id
        next_state = run_deletion_graph(state)
        insert_state(conn, req.session_id, int(next_state.get("current_turn", 0)), next_state)
        conn.commit()
    finally:
        conn.close()

    manager.push(req.session_id, build_update_payload(previous, next_state))
    return _public_state(next_state)


@app.get("/api/turn/{session_id}/{turn}")
def get_turn(session_id: str, turn: int):
    conn = get_connection()
    try:
        _get_session(conn, session_id)
        row = conn.execute(
            """
            SELECT C_t, D_t FROM turn_artifacts
            WHERE session_id = ? AND turn = ?
            """,
            (session_id, turn),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Turn not found")
        return {"turn": turn, "user_message": row["C_t"], "assistant_message": row["D_t"]}
    finally:
        conn.close()


@app.get("/api/export/{session_id}")
def export_session(session_id: str):
    conn = get_connection()
    try:
        session = _get_session(conn, session_id)
        messages = _select_all(conn, "SELECT * FROM messages WHERE session_id = ? ORDER BY id", session_id)
        artifacts = _select_all(conn, "SELECT * FROM turn_artifacts WHERE session_id = ? ORDER BY turn", session_id)
        states = _select_all(conn, "SELECT * FROM groundlens_states WHERE session_id = ? ORDER BY turn", session_id)
        corrections = _select_all(conn, "SELECT * FROM corrections WHERE session_id = ? ORDER BY id", session_id)
        return {
            "session": dict(session),
            "messages": messages,
            "turn_artifacts": artifacts,
            "groundlens_states": states,
            "corrections": corrections,
        }
    finally:
        conn.close()


class ProfileSubmitRequest(BaseModel):
    session_id: str
    profile: dict


@app.post("/api/profile/submit")
def submit_profile(req: ProfileSubmitRequest):
    import uuid as _uuid
    conn = get_connection()
    try:
        session = _get_session(conn, req.session_id)
        p = req.profile
        profile_id = _uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO constraint_profiles(
                profile_id, session_id, participant_id,
                priority_1, priority_2, priority_3,
                time_preference, work_hours_per_week, work_days,
                monthly_budget, fixed_events, forbidden_activities,
                health_constraints, study_style, commute_minutes,
                has_car, must_complete, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                profile_id,
                req.session_id,
                session["participant_id"],
                p.get("priority_1", ""),
                p.get("priority_2", ""),
                p.get("priority_3", ""),
                p.get("time_preference", ""),
                float(p.get("work_hours_per_week", 0)),
                json.dumps(p.get("work_days", []), ensure_ascii=False),
                int(p.get("monthly_budget", 0)),
                json.dumps(p.get("fixed_events", []), ensure_ascii=False),
                p.get("forbidden_activities", ""),
                p.get("health_constraints", ""),
                p.get("study_style", ""),
                int(p.get("commute_minutes", 0)),
                1 if p.get("has_car") else 0,
                p.get("must_complete", ""),
                now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"profile_id": profile_id}


class StageAdvanceRequest(BaseModel):
    session_id: str
    to_stage: str  # 'S2' | 'S3' | 'done'


@app.post("/api/stage/advance")
def advance_stage(req: StageAdvanceRequest):
    if req.to_stage not in {"S2", "S3", "done"}:
        raise HTTPException(status_code=400, detail="to_stage must be 'S2', 'S3', or 'done'")
    conn = get_connection()
    try:
        _get_session(conn, req.session_id)
        previous = load_latest_state(conn, req.session_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="State not found")

        next_state = dict(previous)
        next_state["current_stage"] = req.to_stage
        next_state["turn_in_stage"] = 0
        next_state["revision_count"] = 0
        next_state["stage_gate_ready"] = (req.to_stage == "S3")

        conn.execute(
            "UPDATE sessions SET current_stage = ?, turn_in_stage = 0, revision_count = 0 WHERE session_id = ?",
            (req.to_stage, req.session_id),
        )
        insert_state(conn, req.session_id, int(next_state.get("current_turn", 0)), next_state)
        conn.commit()
    finally:
        conn.close()

    manager.push(req.session_id, build_update_payload(previous, next_state))
    stage_labels = {"S2": "2단계(계획 초안)로 전환되었습니다.", "S3": "3단계(최종 점검)로 전환되었습니다.", "done": "세션이 완료되었습니다."}
    return {"stage": req.to_stage, "message": stage_labels.get(req.to_stage, "단계 전환")}


@app.get("/api/debug/{session_id}")
def debug_session(session_id: str):
    conn = get_connection()
    try:
        _get_session(conn, session_id)
        st = load_latest_state(conn, session_id)
        if st is None:
            raise HTTPException(status_code=404, detail="State not found")
        return {
            "session_id": session_id,
            "current_turn": st.get("current_turn", 0),
            "current_stage": st.get("current_stage", "S1"),
            "turn_in_stage": st.get("turn_in_stage", 0),
            "revision_count": st.get("revision_count", 0),
            "stage_gate_ready": st.get("stage_gate_ready", False),
            "F": st.get("F", []),
            "common_ground": st.get("common_ground", []),
            "llm_ground": st.get("llm_ground", []),
            "grounding_corrections": st.get("grounding_corrections", []),
            "llm_ground_deletions": st.get("llm_ground_deletions", []),
        }
    finally:
        conn.close()


def _get_session(conn, session_id: str):
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


def _select_all(conn, sql: str, session_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, (session_id,)).fetchall()]


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": state.get("current_turn", 0),
        "common_ground": state.get("common_ground", []),
        "llm_ground": state.get("llm_ground", []),
    }
