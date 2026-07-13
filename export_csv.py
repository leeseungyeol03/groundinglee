from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from app.database import get_connection

ROOT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ROOT_DIR / "exports"


def export_session(session_id: str) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        tables = ["sessions", "messages", "turn_artifacts", "groundlens_states", "corrections"]
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table} WHERE session_id = ?", (session_id,)).fetchall()
            path = EXPORT_DIR / f"{table}_{session_id}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                if not rows:
                    continue
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([dict(row) for row in rows])
        state_rows = conn.execute(
            "SELECT turn, state_json FROM groundlens_states WHERE session_id = ? ORDER BY turn",
            (session_id,),
        ).fetchall()
        with (EXPORT_DIR / f"states_{session_id}.json").open("w", encoding="utf-8") as f:
            json.dump([{"turn": row["turn"], "state": json.loads(row["state_json"])} for row in state_rows], f, ensure_ascii=False, indent=2)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python export_csv.py <session_id>")
    export_session(sys.argv[1])
