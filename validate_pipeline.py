"""
주입 파이프라인 오프라인 검증 하네스 (AC4b).

논문 §6.2.3의 세 지표를 저장된 세션 데이터에서 산출한다:
  1) 발현율        — 주입된 전제가 파트너 응답에 발현된 비율
  2) checker 정확도 — 규칙 기반 발현 확인기 판정 vs 사람 라벨(수동)
  3) 노드 추출율    — 발현된 주입 전제가 LLM Ground 노드로 추출된 비율(휴리스틱)

이 스크립트는 "측정 도구"이며 실험 동작을 바꾸지 않는다.
[실행 시 결정 / 연구자] 세 지표의 통과 기준값(--manifest-bar 등)은 파일럿
데이터·선행 문헌을 보고 확정한다. 기준을 주지 않으면 보고만 하고 PASS/FAIL을
판정하지 않는다. checker 정확도는 사람 라벨이 필요하므로, 먼저
--export-label-template 로 CSV를 뽑아 사람이 채운 뒤 --labels-csv 로 넣는다.

사용 예:
  python GL/validate_pipeline.py --json
  python GL/validate_pipeline.py --export-label-template labels.csv
  python GL/validate_pipeline.py --labels-csv labels.csv \
      --manifest-bar 0.7 --checker-bar 0.8 --extraction-bar 0.7
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "sessions.db"

# 노드 추출 휴리스틱 매칭 파라미터
_STOPWORDS = {"목표", "계획", "포함", "수준", "이상", "기준", "배치", "활동", "구성", "제시"}
_MIN_TOKEN_LEN = 2
_EXTRACTION_OVERLAP_THRESHOLD = 0.30  # 주입 키워드 중 이 비율 이상이 노드에 등장하면 매칭


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _tokens(text: str) -> set[str]:
    """한글/숫자/영문 토큰 추출 후 불용어·짧은 토큰 제거."""
    raw = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    return {t for t in raw if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS}


# --------------------------------------------------------------------------- #
# 지표 1: 발현율
# --------------------------------------------------------------------------- #
def manifestation_rate(conn: sqlite3.Connection) -> dict[str, Any]:
    injections = conn.execute("SELECT injection_id, status FROM injections").fetchall()
    total = len(injections)
    manifested_ids = {
        r["injection_id"]
        for r in conn.execute(
            "SELECT DISTINCT injection_id FROM injection_events WHERE manifested = 1"
        ).fetchall()
    }
    manifested = sum(
        1
        for r in injections
        if r["status"] == "manifested" or r["injection_id"] in manifested_ids
    )
    return {
        "total_injections": total,
        "manifested": manifested,
        "rate": (manifested / total) if total else None,
    }


# --------------------------------------------------------------------------- #
# 지표 3: 노드 추출율 (휴리스틱 — 사람 검증 권장)
# --------------------------------------------------------------------------- #
def _session_lg_contents(conn: sqlite3.Connection, session_id: str) -> list[str]:
    """세션의 모든 턴 state에서 등장한 llm_ground 노드 content 수집."""
    contents: list[str] = []
    for row in conn.execute(
        "SELECT state_json FROM groundlens_states WHERE session_id = ? ORDER BY turn",
        (session_id,),
    ).fetchall():
        try:
            state = json.loads(row["state_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        for node in state.get("llm_ground", []) or []:
            if isinstance(node, dict) and node.get("content"):
                contents.append(str(node["content"]))
    return contents


def _premise_matches_node(premise: str, node_contents: list[str]) -> bool:
    p_tokens = _tokens(premise)
    if not p_tokens:
        return False
    for content in node_contents:
        c_tokens = _tokens(content)
        if not c_tokens:
            continue
        overlap = len(p_tokens & c_tokens) / len(p_tokens)
        if overlap >= _EXTRACTION_OVERLAP_THRESHOLD:
            return True
    return False


def node_extraction_rate(conn: sqlite3.Connection) -> dict[str, Any]:
    """발현된 주입 전제 중 LLM Ground 노드로 추출된 비율(휴리스틱 키워드 매칭)."""
    manifested_ids = {
        r["injection_id"]
        for r in conn.execute(
            "SELECT DISTINCT injection_id FROM injection_events WHERE manifested = 1"
        ).fetchall()
    }
    rows = conn.execute(
        "SELECT injection_id, session_id, template_filled, status FROM injections"
    ).fetchall()
    considered = [
        r for r in rows if r["status"] == "manifested" or r["injection_id"] in manifested_ids
    ]
    matched = 0
    detail: list[dict[str, Any]] = []
    lg_cache: dict[str, list[str]] = {}
    for r in considered:
        sid = r["session_id"]
        if sid not in lg_cache:
            lg_cache[sid] = _session_lg_contents(conn, sid)
        is_match = _premise_matches_node(r["template_filled"], lg_cache[sid])
        matched += int(is_match)
        detail.append({"injection_id": r["injection_id"], "extracted": is_match})
    n = len(considered)
    return {
        "manifested_considered": n,
        "extracted": matched,
        "rate": (matched / n) if n else None,
        "method": "heuristic-keyword-overlap",
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# 지표 2: checker 정확도 (사람 라벨 필요)
# --------------------------------------------------------------------------- #
def export_label_template(conn: sqlite3.Connection, out_path: Path) -> int:
    """checker 판정 대비용 사람 라벨 CSV 템플릿 출력. human_manifested 열은 사람이 채운다."""
    rows = conn.execute(
        """
        SELECT e.injection_id, e.session_id, e.turn, e.manifested AS checker_manifested,
               COALESCE(t.D_t, '') AS d_t
        FROM injection_events e
        LEFT JOIN turn_artifacts t
          ON t.session_id = e.session_id AND t.turn = e.turn
        ORDER BY e.session_id, e.turn
        """
    ).fetchall()
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["injection_id", "session_id", "turn", "checker_manifested",
             "human_manifested", "d_t_excerpt"]
        )
        for r in rows:
            w.writerow([
                r["injection_id"], r["session_id"], r["turn"],
                int(r["checker_manifested"]), "",  # human_manifested: 사람이 0/1 기입
                (r["d_t"] or "")[:200].replace("\n", " "),
            ])
    return len(rows)


def checker_accuracy(labels_csv: Path) -> dict[str, Any]:
    """사람이 채운 라벨 CSV로 규칙 checker 정확도·혼동행렬 산출."""
    tp = tn = fp = fn = skipped = 0
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            human = (row.get("human_manifested") or "").strip()
            if human not in {"0", "1"}:
                skipped += 1
                continue
            checker = (row.get("checker_manifested") or "").strip()
            h, c = human == "1", checker == "1"
            if h and c:
                tp += 1
            elif not h and not c:
                tn += 1
            elif not h and c:
                fp += 1
            else:
                fn += 1
    labeled = tp + tn + fp + fn
    return {
        "labeled": labeled,
        "skipped_unlabeled": skipped,
        "accuracy": ((tp + tn) / labeled) if labeled else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _judge(value: float | None, bar: float | None) -> str:
    if value is None or bar is None:
        return "n/a"
    return "PASS" if value >= bar else "FAIL"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="주입 파이프라인 오프라인 검증 하네스 (AC4b)")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--export-label-template", type=Path, default=None,
                    help="checker 정확도용 사람 라벨 CSV 템플릿 경로")
    ap.add_argument("--labels-csv", type=Path, default=None,
                    help="사람이 채운 라벨 CSV (checker 정확도 산출)")
    ap.add_argument("--manifest-bar", type=float, default=0.70, help="발현율 통과 기준 (추천 기본 0.70; 연구자 조정 가능)")
    ap.add_argument("--checker-bar", type=float, default=0.85, help="checker 정확도 통과 기준 (추천 기본 0.85; 연구자 조정 가능)")
    ap.add_argument("--extraction-bar", type=float, default=0.70, help="노드 추출율 통과 기준 (추천 기본 0.70; 연구자 조정 가능)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB 없음: {args.db}")

    conn = _connect(args.db)
    try:
        if args.export_label_template:
            n = export_label_template(conn, args.export_label_template)
            print(f"라벨 템플릿 {n}행 작성: {args.export_label_template} "
                  f"(human_manifested 열을 0/1로 채운 뒤 --labels-csv 로 전달)")
            return

        report: dict[str, Any] = {
            "manifestation": manifestation_rate(conn),
            "node_extraction": node_extraction_rate(conn),
            "checker_accuracy": (
                checker_accuracy(args.labels_csv) if args.labels_csv else
                {"accuracy": None, "note": "사람 라벨 CSV 미제공 — --export-label-template 후 --labels-csv"}
            ),
        }
        report["gate"] = {
            "manifest": _judge(report["manifestation"]["rate"], args.manifest_bar),
            "checker": _judge(report["checker_accuracy"].get("accuracy"), args.checker_bar),
            "extraction": _judge(report["node_extraction"]["rate"], args.extraction_bar),
            "bars": {"manifest": args.manifest_bar, "checker": args.checker_bar,
                     "extraction": args.extraction_bar},
        }
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        m, nx, ca, g = (report["manifestation"], report["node_extraction"],
                        report["checker_accuracy"], report["gate"])
        print("=== 주입 파이프라인 오프라인 검증 ===")
        print(f"① 발현율      : {m['rate']}  ({m['manifested']}/{m['total_injections']})  [{g['manifest']}]")
        print(f"② checker 정확도: {ca.get('accuracy')}  [{g['checker']}]  {ca.get('note','')}")
        print(f"③ 노드 추출율  : {nx['rate']}  ({nx['extracted']}/{nx['manifested_considered']}, {nx['method']})  [{g['extraction']}]")


if __name__ == "__main__":
    main()
