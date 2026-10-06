"""
코퍼스 평가 메트릭.
gold 주석(corpus/annotation_*.json, 사람이 채운 것) vs 시스템 출력(sessions.db)을 대조해
§6.2.2 파이프라인 지표 + 현상 통계를 산출한다.

매칭: 전제 텍스트는 gold(사람)와 system(LLM)이 표현이 달라, 토큰 containment로 퍼지 매칭.

분류 지표는 각 대화의 '최종 턴 상태'(누적 CG/LG)에서 계산하고,
현상 통계(LG 빈도·유형분포·재사용)는 gold의 전체 턴에서 계산한다.

사용법:
  python corpus_metrics.py                       # corpus/annotation_*.json 전부
  python corpus_metrics.py annotation_6a1b9d79.json ...
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "sessions.db"
CORPUS = ROOT / "corpus"

_STOP = {"목표", "계획", "포함", "수준", "이상", "기준", "배치", "활동", "구성", "제시", "사용자", "전제"}
_MATCH_THRESH = 0.5   # 토큰 containment 임계
_SUBTYPES = ["구조적 결정", "암묵적 전제", "용어 해석", "기타"]


def _tokens(t: str) -> set[str]:
    return {w for w in re.findall(r"[가-힣A-Za-z0-9]+", t or "") if len(w) >= 2 and w not in _STOP}


def _score(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _best(premise: str, cands: list[str]) -> tuple[int, float]:
    best, bi = 0.0, -1
    for i, c in enumerate(cands):
        s = _score(premise, c)
        if s > best:
            best, bi = s, i
    return (bi, best) if best >= _MATCH_THRESH else (-1, best)


def _system_final(conn: sqlite3.Connection, sid: str) -> dict:
    row = conn.execute(
        "SELECT state_json FROM groundlens_states WHERE session_id=? ORDER BY turn DESC LIMIT 1",
        (sid,),
    ).fetchone()
    if not row:
        return {"cg": [], "lg": []}
    st = json.loads(row[0])
    cg = [i.get("content", "") for i in st.get("common_ground", [])]
    lg = [{"content": i.get("content", ""),
           "type": i.get("type", ""),
           "src": i.get("source_turn", i.get("first_turn"))} for i in st.get("llm_ground", [])]
    return {"cg": cg, "lg": lg}


def _gold_final_and_all(gold: dict) -> tuple[list, list, list]:
    """반환: (final_cg[str], final_lg[{content,subtype,provenance}], all_lg_across_turns[{...,reused}])"""
    turns = gold.get("turns", [])
    all_lg = []
    for t in turns:
        for it in t["gold"].get("llm_ground", []):
            if it.get("content", "").strip():
                all_lg.append(it)
    last = turns[-1]["gold"] if turns else {"common_ground": [], "llm_ground": []}
    final_cg = [i["content"] for i in last.get("common_ground", []) if i.get("content", "").strip()]
    final_lg = [i for i in last.get("llm_ground", []) if i.get("content", "").strip()]
    return final_cg, final_lg, all_lg


def evaluate(gold_files: list[Path]) -> None:
    conn = sqlite3.connect(str(DB))

    ext_tp = ext_gold = ext_sys = 0          # 추출(매칭) 집계
    conf = Counter()                          # (gold_bin, sys_bin)
    prov_hit = prov_tot = 0
    sub_gold, sub_pred = [], []               # 세부유형 (매칭된 gold-LG↔sys-LG)
    ph_lg_counts, ph_reused, ph_lg_total = [], 0, 0
    ph_subtypes = Counter()
    n_sessions = 0
    n_labeled = 0

    for gf in gold_files:
        gold = json.loads(gf.read_text(encoding="utf-8"))
        sid = gold["session_id"]
        fcg, flg, all_lg = _gold_final_and_all(gold)
        if not (fcg or flg):
            continue  # 아직 라벨 안 됨
        n_labeled += 1
        sysf = _system_final(conn, sid)
        n_sessions += 1

        # --- 현상 통계 (gold 전체 턴) ---
        ph_lg_counts.append(len(all_lg))
        ph_lg_total += len(all_lg)
        for it in all_lg:
            st = it.get("subtype", "")
            st = next((s for s in _SUBTYPES if s in str(st)), "미상")
            ph_subtypes[st] += 1
            if it.get("reused_later"):
                ph_reused += 1

        # --- 분류 지표 (최종 턴) ---
        gold_items = [("CG", c) for c in fcg] + [("LG", l["content"]) for l in flg]
        sys_cg = sysf["cg"]
        sys_lg_c = [l["content"] for l in sysf["lg"]]
        sys_pool = sys_cg + sys_lg_c
        sys_bins = ["CG"] * len(sys_cg) + ["LG"] * len(sys_lg_c)

        ext_gold += len(gold_items)
        ext_sys += len(sys_pool)
        for gbin, gtext in gold_items:
            bi, _ = _best(gtext, sys_pool)
            if bi >= 0:
                ext_tp += 1
                conf[(gbin, sys_bins[bi])] += 1
        # provenance & subtype: gold-LG를 sys-LG에 매칭
        for l in flg:
            bi, _ = _best(l["content"], sys_lg_c)
            if bi >= 0:
                # provenance
                gp, sp = l.get("provenance_turn"), sysf["lg"][bi]["src"]
                if gp is not None and sp is not None:
                    prov_tot += 1
                    prov_hit += int(int(gp) == int(sp))
                # subtype
                gsub = next((s for s in _SUBTYPES if s in str(l.get("subtype", ""))), None)
                ssub = next((s for s in _SUBTYPES if s in str(sysf["lg"][bi]["type"])), None)
                if gsub and ssub:
                    sub_gold.append(gsub)
                    sub_pred.append(ssub)

    # ---- 출력 ----
    print(f"라벨된 세션: {n_labeled}/{len(gold_files)}")
    if n_sessions == 0:
        print("→ gold 주석이 아직 없습니다. 템플릿을 채운 뒤 다시 실행하세요.")
        return

    rec = ext_tp / ext_gold if ext_gold else 0
    prec = ext_tp / ext_sys if ext_sys else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print("\n[전제 추출]")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  (gold={ext_gold}, system={ext_sys}, matched={ext_tp})")

    print("\n[CG/LG 분류 confusion  (gold→system)]")
    for gb in ("CG", "LG"):
        print(f"  gold {gb}: CG={conf[(gb,'CG')]}  LG={conf[(gb,'LG')]}")
    lg_matched = conf[("LG", "CG")] + conf[("LG", "LG")]
    false_cg = conf[("LG", "CG")] / lg_matched if lg_matched else 0
    print(f"  ★ false common ground rate = {false_cg:.3f}  (gold-LG를 system이 CG로 오분류)")

    if sub_gold:
        # macro-F1 over subtypes
        f1s = []
        for s in _SUBTYPES:
            tp = sum(1 for g, p in zip(sub_gold, sub_pred) if g == s and p == s)
            fp = sum(1 for g, p in zip(sub_gold, sub_pred) if g != s and p == s)
            fn = sum(1 for g, p in zip(sub_gold, sub_pred) if g == s and p != s)
            pr = tp / (tp + fp) if (tp + fp) else 0
            rc = tp / (tp + fn) if (tp + fn) else 0
            f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0)
        print(f"\n[LG 세부유형]  macro-F1={sum(f1s)/len(f1s):.3f}  (n={len(sub_gold)})")
    print(f"[provenance]  accuracy={prov_hit/prov_tot:.3f}  (n={prov_tot})" if prov_tot else "[provenance]  n/a")

    print("\n[현상 통계 - gold 전체 턴]")
    print(f"  대화당 평균 LG 수: {ph_lg_total/n_sessions:.1f}  (총 {ph_lg_total}, 세션 {n_sessions})")
    print(f"  LG 재사용률: {ph_reused/ph_lg_total:.3f}" if ph_lg_total else "  LG 재사용률: n/a")
    print("  세부유형 분포:")
    for s in _SUBTYPES + ["미상"]:
        if ph_subtypes.get(s):
            print(f"    {s}: {ph_subtypes[s]} ({ph_subtypes[s]/ph_lg_total*100:.0f}%)")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    files = [CORPUS / a for a in args] if args else sorted(CORPUS.glob("annotation_*.json"))
    files = [f for f in files if f.exists()]
    if not files:
        print("gold 파일 없음. 먼저 corpus_export.py로 템플릿을 뽑고 주석하세요.")
        return
    evaluate(files)


if __name__ == "__main__":
    main()
