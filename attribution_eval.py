"""귀속(provenance) 정확도 평가 — 템플릿 생성 + 채점.

배경:
계획 필드 → 전제 귀속은 A축 설계의 핵심이지만 LLM 판정이 들어가는 지점이다.
CG/LG 분류에서 근거 없는 승격(false common ground)이 대량 발생했던 전례가 있으므로,
귀속도 정확도를 재기 전에는 신뢰하지 않는다.

CG/LG 골드 주석과 달리 귀속 라벨링은 싸다:
  - 필드와 전제가 둘 다 화면에 보이는 텍스트다
  - 유한 집합 간 이분 매칭이라 "무엇이 전제인가"를 새로 판단할 필요가 없다
  - 표본 추출로 일부만 라벨해도 과잉 귀속 여부는 드러난다

사용법:
  # 1) 라벨 템플릿 생성 (필드를 표본 추출)
  python attribution_eval.py export <session_id> [--sample N] [--turn T]

  # 2) 사람이 corpus/attribution_<sid8>.json의 premise_ids_gold를 채운 뒤
  python attribution_eval.py score [corpus/attribution_*.json ...]

채점 지표:
  precision  시스템이 건 귀속 중 옳은 비율      (낮으면 = 과잉 귀속)
  recall     사람이 건 귀속 중 시스템이 잡은 비율 (낮으면 = 누락)
  exact      필드 단위로 귀속 집합이 완전히 일치한 비율
  기타: 미귀속 일치율(둘 다 빈 배열), 환각 id 수
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "sessions.db"
OUT = ROOT / "corpus"
OUT.mkdir(exist_ok=True)

SAMPLE_SEED = 20260920


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _premises_at(conn: sqlite3.Connection, sid: str, turn: int) -> tuple[list[dict], list[dict]]:
    row = conn.execute(
        "SELECT state_json FROM groundlens_states WHERE session_id=? AND turn=?",
        (sid, turn),
    ).fetchone()
    if not row:
        return [], []
    st = json.loads(row["state_json"])
    return st.get("common_ground", []), st.get("llm_ground", [])


def export(sid: str, sample: int | None, only_turn: int | None) -> None:
    from app.plan_parser import has_plan, parse_plan

    conn = _conn()
    arts = conn.execute(
        "SELECT turn, D_t FROM turn_artifacts WHERE session_id=? ORDER BY turn", (sid,)
    ).fetchall()

    rng = random.Random(SAMPLE_SEED)
    turns_out = []
    for a in arts:
        if only_turn is not None and a["turn"] != only_turn:
            continue
        if not has_plan(a["D_t"] or ""):
            continue
        fields = parse_plan(a["D_t"], turn=a["turn"])
        if not fields:
            continue
        cg, lg = _premises_at(conn, sid, a["turn"])
        premises = (
            [{"id": i["id"], "content": i.get("content", ""), "state": "검증됨(CG)"}
             for i in cg if i.get("id")]
            + [{"id": i["id"], "content": i.get("content", ""), "state": "미검증(LG)"}
               for i in lg if i.get("id")]
        )
        if not premises:
            continue

        picked = fields
        if sample and sample < len(fields):
            picked = sorted(rng.sample(fields, sample), key=lambda f: fields.index(f))

        turns_out.append({
            "turn": a["turn"],
            "premises": premises,
            "fields": [
                {
                    "address": f["address"],
                    "section": f["section"],
                    "label": f.get("label", ""),
                    "content": f["content"],
                    "premise_ids_gold": [],       # ← 주석자가 채운다
                }
                for f in picked
            ],
        })
    conn.close()

    if not turns_out:
        print(f"skip {sid}: 계획 응답 없음")
        return

    data = {
        "session_id": sid,
        "task": "계획 필드 → 전제 귀속",
        "annotation_guide": {
            "질문": "이 계획 필드는 위 전제 중 무엇 때문에 지금 내용이 되었는가?",
            "기준": "P가 없었다면 F가 지금 내용이 될 이유가 없다 → 귀속",
            "귀속하지_않는_경우": [
                "단어만 겹침",
                "막연한 관련성",
                "계획 전반에 두루 적용되는 일반 원칙(특정 필드의 근거가 아님)",
            ],
            "상한": "한 필드당 최대 3개. 가장 직접적인 것부터.",
            "빈_배열": "근거를 못 찾으면 빈 배열로 두세요. 이는 정상이며 유의미한 신호입니다.",
            "how_to": "각 field의 premise_ids_gold에 전제 id를 넣으세요. 시스템 출력은 제공하지 않습니다.",
        },
        "turns": turns_out,
    }
    p = OUT / f"attribution_{sid[:8]}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n_fields = sum(len(t["fields"]) for t in turns_out)
    print(f"wrote {p.name}  ({len(turns_out)} turns, {n_fields} fields)")


def score(paths: list[Path]) -> int:
    os.environ.setdefault("MOCK_LLM", "true")
    from app.plan_parser import parse_plan
    from classification.module import step3_attribute_plan_fields

    conn = _conn()
    tp = fp = fn = 0
    exact = total = 0
    both_empty = gold_empty = sys_empty = 0
    halluc = 0
    labeled_files = 0

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data["session_id"]
        has_label = any(
            f.get("premise_ids_gold") for t in data["turns"] for f in t["fields"]
        )
        if not has_label:
            continue
        labeled_files += 1

        for t in data["turns"]:
            row = conn.execute(
                "SELECT D_t FROM turn_artifacts WHERE session_id=? AND turn=?",
                (sid, t["turn"]),
            ).fetchone()
            if not row:
                continue
            parsed = {f["address"]: f for f in parse_plan(row["D_t"], turn=t["turn"])}
            targets = [parsed[f["address"]] for f in t["fields"] if f["address"] in parsed]
            if not targets:
                continue

            cg, lg = _premises_at(conn, sid, t["turn"])
            predicted = step3_attribute_plan_fields(targets, cg, lg, [], "eval")
            valid_ids = {p["id"] for p in t["premises"]}

            for f in t["fields"]:
                gold = set(f.get("premise_ids_gold") or [])
                pred = set(predicted.get(f["address"], []))
                halluc += len(pred - valid_ids)
                total += 1
                if gold == pred:
                    exact += 1
                if not gold and not pred:
                    both_empty += 1
                if not gold:
                    gold_empty += 1
                if not pred:
                    sys_empty += 1
                tp += len(gold & pred)
                fp += len(pred - gold)
                fn += len(gold - pred)
    conn.close()

    if labeled_files == 0:
        print("라벨된 파일이 없습니다. premise_ids_gold를 채운 뒤 다시 실행하세요.")
        print("  템플릿 생성: python attribution_eval.py export <session_id> --sample 12")
        return 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("=" * 70)
    print(f"귀속 정확도  (라벨 파일 {labeled_files}개, 필드 {total}개)")
    print("=" * 70)
    print(f"  precision = {prec:.3f}   (시스템 귀속 {tp + fp}건 중 {tp}건 적중)")
    print(f"  recall    = {rec:.3f}   (사람 귀속 {tp + fn}건 중 {tp}건 포착)")
    print(f"  F1        = {f1:.3f}")
    print(f"  exact     = {exact / total:.3f}   (필드 단위 완전일치 {exact}/{total})")
    print()
    print(f"  미귀속 일치   : {both_empty}   (둘 다 빈 배열)")
    print(f"  사람 미귀속   : {gold_empty}/{total}")
    print(f"  시스템 미귀속 : {sys_empty}/{total}")
    print(f"  환각 id       : {halluc}")
    print()
    if prec < 0.6:
        print("  ⚠ precision이 낮다 = 과잉 귀속. 무관한 필드까지 무효화되어")
        print("    교정이 계획을 과도하게 망가뜨린다. 프롬프트 기준을 좁힐 것.")
    if rec < 0.6:
        print("  ⚠ recall이 낮다 = 귀속 누락. 교정해도 살아남는 계획 부분이 생겨")
        print("    누적이 해소되지 않는다.")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    cmd, rest = args[0], args[1:]
    if cmd == "export":
        sample = None
        only_turn = None
        sids = []
        i = 0
        while i < len(rest):
            if rest[i] == "--sample" and i + 1 < len(rest):
                sample = int(rest[i + 1]); i += 2
            elif rest[i] == "--turn" and i + 1 < len(rest):
                only_turn = int(rest[i + 1]); i += 2
            else:
                sids.append(rest[i]); i += 1
        if not sids:
            print("usage: attribution_eval.py export <session_id> [--sample N] [--turn T]")
            return 1
        for sid in sids:
            export(sid, sample, only_turn)
        return 0

    if cmd == "score":
        paths = [Path(p) for p in rest] or sorted(OUT.glob("attribution_*.json"))
        paths = [p for p in paths if p.exists()]
        if not paths:
            print("attribution_*.json 없음. 먼저 export 하세요.")
            return 1
        return score(paths)

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
