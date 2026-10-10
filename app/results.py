"""실험 결과 저장 — 덮어쓰기 방지.

왜 필요한가:
E2 파일럿 8에피소드를 돌린 뒤 계측기 테스트로 2에피소드를 돌렸더니 결과 파일이
덮어써졌다. 집계 수치는 문서에 적어 뒀지만 에피소드별 상세는 날아갔다.
825초와 API 비용을 버린 것이다.

규칙
    1. 파일명에 타임스탬프를 붙인다. 같은 실험을 다시 돌려도 안 겹친다
    2. latest 심링크 대신 고정 이름 사본을 하나 더 둔다(읽기 편의)
    3. 실행 메타(모델, 시드, 커밋)를 함께 적는다. 수치만 있으면 재현이 안 된다
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

RUNTIME = Path(__file__).resolve().parent.parent / ".runtime"


def _git_commit() -> str:
    git = RUNTIME / "mingit" / "cmd" / "git.exe"
    exe = str(git) if git.exists() else "git"
    try:
        r = subprocess.run([exe, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=20,
                           cwd=str(Path(__file__).resolve().parent.parent))
        return (r.stdout or "").strip() or "?"
    except Exception:
        return "?"


def run_meta(**extra: Any) -> dict[str, Any]:
    """실행 메타. 수치만 저장하면 나중에 재현할 수 없다."""
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "local_model": os.environ.get(
            "GL_LOCAL_MODEL", "C:/seungyeol/vscode/GL/.models/Qwen3-8B"),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    meta.update(extra)
    return meta


def save(name: str, payload: dict[str, Any], **meta_extra: Any) -> Path:
    """타임스탬프 파일과 고정 이름 사본을 모두 쓴다.

    반환: 타임스탬프 파일 경로
    """
    RUNTIME.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    body = {"meta": run_meta(**meta_extra), **payload}
    text = json.dumps(body, ensure_ascii=False, indent=1)

    stamped = RUNTIME / f"{name}_{stamp}.json"
    stamped.write_text(text, encoding="utf-8")
    (RUNTIME / f"{name}_latest.json").write_text(text, encoding="utf-8")
    return stamped


def load_latest(name: str) -> dict[str, Any] | None:
    p = RUNTIME / f"{name}_latest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_runs(name: str) -> list[Path]:
    return sorted(RUNTIME.glob(f"{name}_20*.json"))
