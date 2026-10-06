"""주의 초점 고착(attentional perseveration) 측정 — 실현 가능성 프로브.

문제 정의:
  LLM이 "틀린 말"을 하는 것이 아니라 "지금 할 말이 아닌 말"을 한다.
  사용자가 대화의 한 가지(branch)로 좁혀 들어가려는데, 모델이 이전 줄기(trunk)를
  계속 붙들고 있는 현상. 내용은 참이고 이전에 합의된 것이므로
  - 할루시네이션이 아니다 (사실 오류가 없다)
  - 그라운딩 실패가 아니다 (공유된 전제다)
  그러나 **지금 활성화된 초점이 아니다.**

이론 좌표 (Grosz & Sidner의 담화 구조 3요소):
  intentional structure  — 의도 구조
  attentional state      — 초점 공간 스택. 무엇이 "지금 활성"인가
  linguistic structure   — 언어 구조
GroundLens(현 논문)는 전제의 **검증 상태**만 모델링한다.
이 프로브가 다루는 것은 전제의 **활성 상태**다. 두 축은 직교한다.

  검증됨 × 활성    → 지금 쓰여야 할 전제
  검증됨 × 비활성  → ★ 사용자가 겪는 실패 모드 (참인데 지금 할 말이 아님)
  미검증 × 활성    → 현 논문이 다루는 실패 모드
  미검증 × 비활성  → 죽은 가정

측정 설계:
  대본화된 공통 전사(trunk)를 고정한다. 전사를 매 시행 생성하면 조건 간
  비교가 오염되므로 assistant 턴까지 하드코딩한다.
  이후 branch 발화를 넣고, 응답이 요청된 범위를 지키는지 판정한다.

조건:
  full   — 전체 대화 이력을 그대로 전달 (현재 표준 방식)
  scoped — 활성 초점에 관련된 턴만 전달 (제안 기법의 상한선 근사)
  scoped가 유의하게 낫다면 "낡은 맥락이 고착의 원인"이라는 인과가 지지되고,
  범위 게이팅이 해법 방향임이 확인된다. 사람이 필요 없다.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["MOCK_LLM"] = "false"

import anthropic

from app.llm import get_api_key

CLIENT = anthropic.Anthropic(api_key=get_api_key())
MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # 판정은 더 강한 모델로

# 의도적으로 중립적인 시스템 프롬프트.
# config.yaml의 "5섹션 템플릿 필수"는 그 자체가 과잉 포함을 유발하므로
# 일반 현상을 재려면 배제해야 한다(교락 제거).
SYSTEM = "당신은 사용자의 계획 수립을 돕는 AI 파트너입니다. 사용자의 요청에 맞춰 답하세요."

# ── 대본화된 trunk: 4개 측면이 모두 확립된 상태 ──────────────────
# 각 측면이 뚜렷이 분리되어 있어야 "범위 위반"을 판정할 수 있다.
TRUNK = [
    ("user", "이번 방학에 코딩테스트 준비랑 자소서를 같이 하고 싶어. 7월 한 달이고 주말엔 알바가 있어."),
    ("assistant",
     "7월 한 달간 두 가지를 병행하는 계획을 세워보겠습니다.\n\n"
     "**코딩테스트 준비**: 평일 오전에 알고리즘 유형별 학습, 오후에 문제 풀이를 배치하는 것이 좋겠습니다.\n"
     "**자소서**: 주 2회 정도 저녁 시간에 기업별 초안을 작성하는 방식을 제안합니다.\n"
     "**알바 일정**: 주말이 막혀 있으니 평일 집중도를 높여야 합니다.\n"
     "**예산**: 인터넷 강의나 교재 비용을 고려해야 합니다.\n\n"
     "각 항목의 구체적인 조건을 알려주시면 더 다듬겠습니다."),
    ("user", "알바는 토요일만 8시간이야. 일요일은 비어. 코테는 프로그래머스 중급 수준으로, 자소서는 3개 기업 초안. 예산은 30만원."),
    ("assistant",
     "조건을 반영하겠습니다.\n\n"
     "**코딩테스트**: 프로그래머스 중급 기준으로 평일 오전 유형 학습, 오후 문제 풀이를 유지합니다.\n"
     "**자소서**: 3개 기업이므로 주 2회 저녁에 한 기업씩 순환하며 초안을 작성합니다.\n"
     "**알바**: 토요일 8시간이 고정이고 일요일은 가용하므로, 일요일을 보조 학습일로 씁니다.\n"
     "**예산**: 30만원 내에서 인터넷 강의 하나와 교재 한 권 정도가 적정합니다.\n\n"
     "이 구조로 진행하면 될까요?"),
]

# ── 프로브: 각각 "좁히기" 유형이 다르다 ───────────────────────────
PROBES = [
    {
        "id": "narrow",
        "desc": "하위 측면으로 좁히기",
        "utterance": "자소서랑 예산 얘기는 일단 접어두고, 코테 유형 학습 순서만 정해줘. 어떤 유형부터 할지.",
        "in_scope": "코딩테스트 알고리즘 유형의 학습 순서",
        "out_scope": ["자소서", "예산", "알바 일정", "주간 루틴 전체"],
    },
    {
        "id": "exclude",
        "desc": "명시적 배제",
        "utterance": "알바 얘기는 하지 말고, 자소서 3개를 어떤 순서로 쓸지만 알려줘.",
        "in_scope": "자소서 작성 순서",
        "out_scope": ["알바", "토요일", "코딩테스트", "예산"],
    },
    {
        "id": "shift",
        "desc": "초점 전환 (trunk 유지, 다른 가지로)",
        "utterance": "코테는 그대로 두고, 자소서 문항 구성을 어떻게 잡을지로 넘어가자.",
        "in_scope": "자소서 문항 구성",
        "out_scope": ["코딩테스트 학습 방법", "알바", "예산"],
    },
]


def call(messages: list[dict], system: str = SYSTEM, max_tokens: int = 900) -> str:
    r = CLIENT.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=messages, extra_body={"temperature": 1.0},
    )
    return "".join(b.text for b in r.content if b.type == "text")


def history_full() -> list[dict]:
    return [{"role": r, "content": c} for r, c in TRUNK]


def history_scoped(probe: dict) -> list[dict]:
    """활성 초점에 관련된 턴만 남긴다.

    사용자 발화는 제약 정보를 담으므로 유지하고, assistant의 장문 응답에서
    out_scope 측면을 서술한 단락을 제거한다. 범위 게이팅의 상한선 근사다.
    """
    out = []
    for role, content in TRUNK:
        if role == "user":
            out.append({"role": role, "content": content})
            continue
        kept = [
            para for para in content.split("\n")
            if not any(k.split()[0] in para for k in probe["out_scope"] if k)
        ]
        text = "\n".join(kept).strip() or "네, 반영하겠습니다."
        out.append({"role": role, "content": text})
    return out


JUDGE_PROMPT = """당신은 대화 응답의 범위 준수를 평가합니다.

사용자가 요청한 범위: {in_scope}
사용자가 벗어나길 원한 범위: {out_scope}

사용자 발화:
{utterance}

AI 응답:
{response}

판정 기준:
- 사용자가 특정 범위로 좁혀 요청했을 때, 응답이 그 범위를 다루는 비중을 본다.
- 벗어난 범위를 "다음에 다루겠다" 정도로 한 줄 언급하는 것은 위반이 아니다.
  그 범위의 내용을 실질적으로 다시 제시하는 것이 위반이다.
- 사용자가 배제를 명시했는데 그 내용을 다시 서술하면 명확한 위반이다.

JSON만 출력:
{{
  "scope_adherence": 0.0~1.0 사이 수치. 응답 분량 중 요청 범위를 다룬 비율,
  "perseveration": true/false. 벗어난 범위의 내용을 실질적으로 재제시했는가,
  "violated_aspects": ["재제시된 범위 이름", ...],
  "reason": "한 문장 근거"
}}"""


def judge(probe: dict, response: str) -> dict:
    p = JUDGE_PROMPT.format(
        in_scope=probe["in_scope"],
        out_scope=", ".join(probe["out_scope"]),
        utterance=probe["utterance"],
        response=response,
    )
    r = CLIENT.messages.create(
        model=JUDGE_MODEL, max_tokens=600,
        messages=[{"role": "user", "content": p}],
        extra_body={"temperature": 0},
    )
    raw = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"scope_adherence": None, "perseveration": None, "violated_aspects": [], "reason": "판정 실패"}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"scope_adherence": None, "perseveration": None, "violated_aspects": [], "reason": "JSON 오류"}


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("=" * 76)
    print("주의 초점 고착 프로브 — 낡은 맥락이 응답 범위를 오염시키는가")
    print(f"모델 {MODEL} / 판정 {JUDGE_MODEL} / 프로브 {len(PROBES)}종 x {trials}회 x 2조건")
    print("=" * 76)

    rows = []
    for probe in PROBES:
        print(f"\n-- [{probe['id']}] {probe['desc']}")
        print(f"   발화: {probe['utterance'][:66]}")
        conds = [("full", history_full), ("scoped", lambda p=probe: history_scoped(p))]
        for cond, hist_fn in conds:
            adh, pers = [], []
            for t in range(trials):
                msgs = hist_fn() + [{"role": "user", "content": probe["utterance"]}]
                resp = call(msgs)
                v = judge(probe, resp)
                if v.get("scope_adherence") is not None:
                    adh.append(v["scope_adherence"])
                    pers.append(bool(v.get("perseveration")))
                    rows.append({
                        "probe": probe["id"], "cond": cond, "trial": t,
                        "adherence": v["scope_adherence"],
                        "perseveration": bool(v.get("perseveration")),
                        "violated": v.get("violated_aspects", []),
                        "reason": v.get("reason", "")[:100],
                        "resp_len": len(resp),
                    })
            if adh:
                ma = sum(adh) / len(adh)
                mp = sum(pers) / len(pers)
                print(f"   {cond:7} 범위준수 {ma:.2f} | 고착발생 {mp:.0%} ({sum(pers)}/{len(pers)})")

    if not rows:
        print("\n판정 전부 실패 - 중단")
        return

    print("\n" + "=" * 76)
    print("집계")
    print("=" * 76)
    for cond in ("full", "scoped"):
        sub = [r for r in rows if r["cond"] == cond]
        if not sub:
            continue
        ma = sum(r["adherence"] for r in sub) / len(sub)
        mp = sum(r["perseveration"] for r in sub) / len(sub)
        print(f"  {cond:7} n={len(sub):2d}  범위준수 {ma:.2f}  고착발생 {mp:.0%}")

    f = [r for r in rows if r["cond"] == "full"]
    s = [r for r in rows if r["cond"] == "scoped"]
    if f and s:
        df = sum(r["adherence"] for r in s) / len(s) - sum(r["adherence"] for r in f) / len(f)
        pf = sum(r["perseveration"] for r in f) / len(f)
        ps = sum(r["perseveration"] for r in s) / len(s)
        print()
        print(f"  범위 게이팅 효과: 준수 {df:+.2f} / 고착 {pf:.0%} -> {ps:.0%}")
        print()
        if pf >= 0.4:
            print("  [O] 현상 재현됨 - 전체 이력 조건에서 고착이 빈발한다.")
            print("      '참이지만 지금 할 말이 아닌' 실패 모드가 실재하며 자동 측정 가능하다.")
        else:
            print(f"  [~] 고착 발생률 {pf:.0%} - 이 과업에서는 약하다.")
            print("      더 긴 이력 / 더 많은 측면 / 더 미묘한 가지치기로 재설계 필요.")
        if ps < pf:
            print("  [O] 범위 게이팅이 고착을 줄인다 - 낡은 맥락이 원인이라는 인과 지지.")
        elif ps > pf:
            print("  [X] 게이팅이 악화 - 초점 판별 자체가 어렵다는 신호.")
        else:
            print("  [~] 게이팅 효과 없음 - 원인이 맥락 길이가 아닐 수 있다.")

    out = ROOT / ".runtime" / "focus_probe_result.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
