from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
_loop: Optional[asyncio.AbstractEventLoop] = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


class GroundLensWS:
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self.connections.pop(session_id, None)

    async def send(self, session_id: str, payload: dict) -> None:
        ws = self.connections.get(session_id)
        if ws:
            await ws.send_json(payload)

    def push(self, session_id: str, payload: dict) -> None:
        if _loop and _loop.is_running():
            asyncio.run_coroutine_threadsafe(self.send(session_id, payload), _loop)


manager = GroundLensWS()


@router.websocket("/ws/groundlens/{session_id}")
async def groundlens_ws(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)


def build_update_payload(previous: dict | None, current: dict) -> dict:
    previous = previous or {"common_ground": [], "llm_ground": []}
    prev_cg = {item.get("id") for item in previous.get("common_ground", [])}
    prev_llm = {item.get("id") for item in previous.get("llm_ground", [])}
    common_ground = current.get("common_ground", [])
    llm_ground = current.get("llm_ground", [])
    return {
        "event": "e_updated",
        "turn": current.get("current_turn", 0),
        "common_ground": common_ground,
        "llm_ground": llm_ground,
        "common_ground_delta": [item for item in common_ground if item.get("id") not in prev_cg],
        "llm_ground_delta": [item for item in llm_ground if item.get("id") not in prev_llm],
        "current_stage": current.get("current_stage", "S1"),
        "stage_gate_ready": current.get("stage_gate_ready", False),
    }
