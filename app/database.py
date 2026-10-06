from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "sessions.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    journal_path = DB_PATH.with_name(DB_PATH.name + "-journal")
    if DB_PATH.exists() and DB_PATH.stat().st_size == 0 and journal_path.exists():
        journal_path.unlink(missing_ok=True)
        DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                user_intent TEXT NOT NULL,
                paper_id TEXT NOT NULL DEFAULT '',
                current_turn INTEGER NOT NULL DEFAULT 0,
                current_stage TEXT NOT NULL DEFAULT 'S1',
                turn_in_stage INTEGER NOT NULL DEFAULT 0,
                revision_count INTEGER NOT NULL DEFAULT 0,
                session_type TEXT NOT NULL DEFAULT 'gl',
                started_at TEXT NOT NULL,
                ended_at TEXT,
                config_snapshot TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user','assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                system_prompt_used TEXT,
                extended_thinking TEXT
            );

            CREATE TABLE IF NOT EXISTS turn_artifacts (
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn INTEGER NOT NULL,
                C_t TEXT NOT NULL,
                D_t TEXT NOT NULL,
                B_t TEXT NOT NULL,
                A_t TEXT NOT NULL,
                system_prompt_used TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, turn)
            );

            CREATE TABLE IF NOT EXISTS groundlens_states (
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, turn)
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                target_item_id TEXT NOT NULL,
                correction_text TEXT NOT NULL,
                source_turn INTEGER NOT NULL,
                corrected_at_turn INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS constraint_profiles (
                profile_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                participant_id TEXT NOT NULL,
                priority_1 TEXT,
                priority_2 TEXT,
                priority_3 TEXT,
                time_preference TEXT,
                work_hours_per_week REAL,
                work_days TEXT,
                monthly_budget INTEGER,
                fixed_events TEXT,
                forbidden_activities TEXT,
                health_constraints TEXT,
                study_style TEXT,
                commute_minutes INTEGER,
                has_car INTEGER,
                must_complete TEXT,
                created_at TEXT NOT NULL
            );

            -- 요구사항 변경 (2단계 과업의 조작, 논문 6.5절)
            -- 주입과 달리 세션당 1건이며 제시 시점이 결정적이다.
            CREATE TABLE IF NOT EXISTS requirement_changes (
                change_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                change_type TEXT NOT NULL,
                label TEXT NOT NULL,
                text TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                profile_field TEXT,
                assigned_by TEXT,
                delivered_at_turn INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS injections (
                injection_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                injection_type TEXT NOT NULL,
                target_stage TEXT NOT NULL,
                target_turn_min INTEGER,
                target_turn_max INTEGER,
                template_filled TEXT NOT NULL,
                latency_mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                manifest_turn INTEGER,
                reintroduced INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS injection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                injection_id TEXT NOT NULL REFERENCES injections(injection_id),
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                checker_type TEXT NOT NULL,
                manifested INTEGER NOT NULL,
                checker_confidence REAL,
                checker_note TEXT,
                reintroduced INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """
        )
        # Migrate existing sessions table for new columns (idempotent)
        for col, definition in [
            ("current_stage",  "TEXT NOT NULL DEFAULT 'S1'"),
            ("turn_in_stage",  "INTEGER NOT NULL DEFAULT 0"),
            ("revision_count", "INTEGER NOT NULL DEFAULT 0"),
            ("session_type",   "TEXT NOT NULL DEFAULT 'gl'"),
            # 과업 단계: 1 = 계획 수립, 2 = 요구사항 변경 후 개정 (논문 6.3절)
            ("task_phase",     "INTEGER NOT NULL DEFAULT 1"),
        ]:
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {definition}")
            except Exception:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()


def insert_state(conn: sqlite3.Connection, session_id: str, turn: int, state: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO groundlens_states(session_id, turn, state_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, turn, json.dumps(state, ensure_ascii=False), now_iso()),
    )


def load_latest_state(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT state_json FROM groundlens_states
        WHERE session_id = ?
        ORDER BY turn DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return json.loads(row["state_json"]) if row else None


def dump_schema() -> list[dict[str, str]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [{"name": row["name"], "sql": row["sql"]} for row in rows]
    finally:
        conn.close()
