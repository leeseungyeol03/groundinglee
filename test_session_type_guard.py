"""session_type 조건 가드 회귀 테스트 (논문 6.3절).

GL-view의 조건 정의는 "GL-full과 동일한 가시화, 개입 채널만 없음"이다.
따라서 노드 조작 엔드포인트는 서버에서 막혀야 한다. 프론트엔드 비활성화만으로는
직접 요청을 차단하지 못해 조건 간 차이가 오염될 수 있다.

실행:
  python test_session_type_guard.py      # 종료코드 0 = 전부 통과
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app.database as database  # noqa: E402

# 실제 DB를 건드리지 않도록 임시 경로로 교체
_tmp = tempfile.TemporaryDirectory()
database.DB_PATH = Path(_tmp.name) / "test.db"
database.init_db()

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402

# main이 load_dotenv(override=True)로 .env를 적용하므로 import 이후에 설정해야 한다.
# mock 판정은 호출 시점에 os.environ을 읽으므로 여기서 덮어쓰면 된다.
os.environ["MOCK_LLM"] = "true"

client = TestClient(main.app)

results: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    results.append((passed, label))
    mark = "PASS" if passed else "**FAIL**"
    print(f"{mark}  {label}" + (f"   ({detail})" if detail else ""))


def start(session_type: str) -> str | None:
    res = client.post("/api/session/start", json={
        "participant_id": f"P_{session_type}",
        "user_intent": "방학 계획을 세우고 싶다",
        "session_type": session_type,
    })
    if res.status_code != 200:
        return None
    return res.json()["session_id"]


def first_llm_item_id(session_id: str) -> str:
    """턴을 한 번 돌려 실제 llm_ground 항목 id를 얻는다. 없으면 더미."""
    res = client.post("/api/message", json={
        "session_id": session_id,
        "user_message": "이번 방학엔 코딩테스트 준비를 하고 싶어",
    })
    if res.status_code == 200:
        items = res.json().get("llm_ground", [])
        if items:
            return items[0].get("id", "dummy_id")
    return "dummy_id"


print("=" * 78)
print("session_type 가드 — 조건별 노드 조작 허용 여부")
print("=" * 78)

# ── 1. 유효하지 않은 session_type 거부 ──
res = client.post("/api/session/start", json={
    "participant_id": "P0", "user_intent": "x", "session_type": "gl_full",
})
check(res.status_code == 400, "잘못된 session_type은 400으로 거부", f"got {res.status_code}")

# ── 2. 세 조건 모두 세션 생성 가능 ──
sids: dict[str, str] = {}
for st in ("gl", "gl_view", "baseline"):
    sid = start(st)
    check(sid is not None, f"session_type='{st}' 세션 생성")
    if sid:
        sids[st] = sid

# ── 3. GL-full은 조작 허용 ──
if "gl" in sids:
    sid = sids["gl"]
    item_id = first_llm_item_id(sid)
    res = client.post("/api/ground/correct", json={
        "session_id": sid, "target_item_id": item_id, "correction_text": "수정한 전제",
    })
    check(res.status_code == 200, "gl: /api/ground/correct 허용", f"got {res.status_code}")

    res = client.post("/api/ground/delete", json={"session_id": sid, "item_id": item_id})
    check(res.status_code == 200, "gl: /api/ground/delete 허용", f"got {res.status_code}")

# ── 4. GL-view는 조작 차단 (핵심) ──
if "gl_view" in sids:
    sid = sids["gl_view"]
    item_id = first_llm_item_id(sid)
    res = client.post("/api/ground/correct", json={
        "session_id": sid, "target_item_id": item_id, "correction_text": "수정 시도",
    })
    check(res.status_code == 403, "gl_view: /api/ground/correct 403 차단", f"got {res.status_code}")

    res = client.post("/api/ground/delete", json={"session_id": sid, "item_id": item_id})
    check(res.status_code == 403, "gl_view: /api/ground/delete 403 차단", f"got {res.status_code}")

    # 대화 채널은 열려 있어야 한다 — 교정 경로 자체를 봉쇄하면 조건이 아니라 결손이 된다
    res = client.post("/api/message", json={
        "session_id": sid, "user_message": "그건 내 의도와 달라. 다시 해줘",
    })
    check(res.status_code == 200, "gl_view: 대화 채널(/api/message)은 정상 동작", f"got {res.status_code}")

# ── 5. baseline도 조작 차단 ──
if "baseline" in sids:
    sid = sids["baseline"]
    res = client.post("/api/ground/delete", json={"session_id": sid, "item_id": "dummy_id"})
    check(res.status_code == 403, "baseline: /api/ground/delete 403 차단", f"got {res.status_code}")

print("\n" + "=" * 78)
ok = sum(1 for p, _ in results if p)
fail = len(results) - ok
print(f"총계: {ok} PASS / {fail} FAIL")
print("=" * 78)

_tmp.cleanup()
raise SystemExit(1 if fail else 0)
