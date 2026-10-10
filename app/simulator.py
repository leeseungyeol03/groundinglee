"""사용자 시뮬레이터 — E2 벤치마크의 상대역 (feedback2.md E2).

왜 시뮬레이터인가:
E2는 "처리(P1·P2·P3)가 메모리식 편집보다 낫다"를 사람 없이 재현 가능하게
보이는 것이 목적이다. 조건 7개 × 시나리오 수십 개를 사람으로 돌릴 수 없고,
같은 시나리오를 조건만 바꿔 반복해야 비교가 성립한다.

설계의 핵심은 **정보를 한 번에 다 주지 않는 것**이다. 사용자가 프로필을 전부
말해 버리면 LLM이 해석을 만들 이유가 없고, 그러면 연구 문제 자체가 사라진다.
실제 사용자도 처음에는 모호하게 말하고 필요할 때 조금씩 드러낸다.

구성 요소
    숨은 프로필     목표·제약·선호 8~10개. 시뮬레이터만 안다
    사적 의미 사전   모호한 표현을 페르소나 고유 의미로 쓴다
                    ("여유롭게" = 매일 오전만 활동). 용어 해석 범주의 LG를
                    자연스럽게 유발하는 장치다
    공개 일정       질문을 받거나 일정 확률로 프로필 조각을 공개
    주의 예산 k     매 턴 패널 항목이나 응답 세그먼트를 k개까지만 검토.
                    사람이 전부 읽지 않는다는 사실을 반영한다
    조건 변경       T턴에 자원 축소·제약 추가·우선순위 반전 중 하나를 제시

시뮬레이터 모델은 **파트너와 다른 계열**이어야 한다. 같은 모델이면 서로의
표현 습관에 맞춰져 그라운딩 문제가 과소 평가된다.
"""
from __future__ import annotations

import json
from app.usage_meter import METER
import os
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# 조건 변경 유형 (feedback2.md E2)
CHANGE_TYPES = ("resource_cut", "constraint_add", "priority_flip")


@dataclass
class Persona:
    """한 시나리오의 숨은 프로필."""
    id: str
    domain: str
    opening: str                                   # 첫 발화. 모호한 목표만
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    # 모호한 표현 -> 이 사람에게는 이런 뜻
    private_meanings: dict[str, str] = field(default_factory=dict)
    # T턴에 제시할 조건 변경
    change_type: str = "resource_cut"
    change_utterance: str = ""
    change_implies: list[str] = field(default_factory=list)   # 변경으로 참이 되는 것
    change_invalidates: list[str] = field(default_factory=list)  # 변경으로 거짓이 되는 것

    @property
    def checklist(self) -> list[str]:
        """숨은 프로필 충족도 채점에 쓰는 항목 전체."""
        return [*self.goals, *self.constraints, *self.preferences]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 계획 도메인 시나리오
#
# 도메인 3종(계획/글쓰기/요구사항)은 연구자 결정 사항이라 보류 상태다.
# 계획 하나로 파일럿을 돌려 G2를 먼저 본다.
# --------------------------------------------------------------------------- #

PERSONAS: list[Persona] = [
    Persona(
        id="plan_exam",
        domain="plan",
        opening="이번 방학에 코딩테스트 준비를 좀 여유롭게 해보려고 해. 계획 좀 세워줄래?",
        goals=[
            "프로그래머스 중급 수준 문제를 꾸준히 푼다",
            "자료구조와 알고리즘 기본서를 한 권 끝낸다",
            "8월 말 코딩테스트 응시 준비를 마친다",
        ],
        constraints=[
            "토요일은 하루 종일 아르바이트가 있다",
            "전체 예산은 20만원을 넘기면 안 된다",
            "7월 셋째 주는 가족 여행으로 통째로 비운다",
        ],
        preferences=[
            "혼자 공부하는 편이고 스터디는 부담스럽다",
            "인터넷 강의보다 책으로 보는 걸 선호한다",
        ],
        private_meanings={
            "여유롭게": "하루 세 시간을 넘기지 않고 저녁은 비운다",
        },
        change_type="resource_cut",
        change_utterance="갑자기 알바가 주 3회로 늘었어. 평일 저녁 두 번이 더 빠져.",
        change_implies=["평일 저녁 두 번은 쓸 수 없다"],
        change_invalidates=["평일 저녁에 일정을 넣는다"],
    ),
    Persona(
        id="plan_portfolio",
        domain="plan",
        opening="8월에 포트폴리오를 하나 만들려고 하는데 계획을 간단하게 짜줘.",
        goals=[
            "React로 웹앱 하나를 배포까지 끝낸다",
            "기술 블로그 글을 네 편 쓴다",
            "깃허브 README를 정리한다",
        ],
        constraints=[
            "일요일은 동아리 모임이 하루 종일 있다",
            "예산은 15만원이고 유료 서비스는 최소한만 쓴다",
            "노트북 성능이 낮아 무거운 작업은 어렵다",
        ],
        preferences=[
            "완성도보다 끝까지 마무리하는 걸 중요하게 본다",
            "디자인은 기성 템플릿을 쓰고 싶다",
        ],
        private_meanings={
            "간단하게": "주 단위로만 나누고 날짜별로 쪼개지 않는다",
        },
        change_type="constraint_add",
        change_utterance="아 그리고 8월 둘째 주에 집안 행사가 있어서 그 주는 거의 못 해.",
        change_implies=["8월 둘째 주는 작업할 수 없다"],
        change_invalidates=["8월 둘째 주에 일정을 넣는다"],
    ),
    Persona(
        id="plan_lang",
        domain="plan",
        opening="겨울방학에 영어랑 자격증을 같이 준비하고 싶은데 무리 안 가게 계획 짜줘.",
        goals=[
            "토익 850점 이상을 목표로 한다",
            "정보처리기사 필기를 통과한다",
            "영어 회화 스터디에 꾸준히 나간다",
        ],
        constraints=[
            "주중 오전에는 학원 수업이 있다",
            "예산은 30만원이고 교재비가 우선이다",
            "1월 첫째 주는 아르바이트 교육으로 바쁘다",
        ],
        preferences=[
            "아침보다 밤에 집중이 잘 된다",
            "인강은 배속으로 몰아 듣는 편이다",
        ],
        private_meanings={
            "무리 안 가게": "하루 두 과목을 같이 하지 않는다",
        },
        change_type="priority_flip",
        change_utterance="생각해보니 기사 시험이 훨씬 급해. 토익은 다음 학기로 미룰게.",
        change_implies=["정보처리기사가 1순위다", "토익은 이번 방학에 하지 않는다"],
        change_invalidates=["토익 850점 이상을 목표로 한다",
                            "영어 회화 스터디에 꾸준히 나간다"],
    ),
    Persona(
        id="plan_fitness",
        domain="plan",
        opening="운동이랑 공부를 같이 하는 루틴을 만들고 싶어. 빡세지 않게 부탁해.",
        goals=[
            "주 4회 헬스장에 간다",
            "전공 서적 두 권을 읽는다",
            "체중을 5kg 줄인다",
        ],
        constraints=[
            "헬스장은 밤 10시에 닫는다",
            "예산은 25만원이고 헬스장 등록비가 포함이다",
            "수요일은 저녁 수업이 있다",
        ],
        preferences=[
            "유산소보다 근력 운동을 좋아한다",
            "식단 관리는 자신 없어서 최소한만 하고 싶다",
        ],
        private_meanings={
            "빡세지 않게": "하루에 운동과 공부를 합쳐 네 시간을 넘기지 않는다",
        },
        change_type="resource_cut",
        change_utterance="예산이 줄어서 헬스장은 못 끕을 것 같아. 10만원 안에서 해결해야 해.",
        change_implies=["예산은 10만원이다", "헬스장을 등록하지 않는다"],
        # 무효화 목록은 빠짐없어야 한다. 예산 항목을 빼먹었더니 프로필에
        # "예산 25만원"과 "예산 10만원"이 동시에 남아 시뮬레이터가 자기
        # 변경을 되돌렸다(패널에서 "헬스장을 등록한다"로 재수정).
        change_invalidates=["헬스장에 주 4회 간다",
                            "예산은 25만원이고 헬스장 등록비가 포함이다"],
    ),
]


def get_persona(pid: str) -> Persona:
    for p in PERSONAS:
        if p.id == pid:
            return p
    raise KeyError(pid)


# --------------------------------------------------------------------------- #
# 시뮬레이터 엔진
# --------------------------------------------------------------------------- #

_SYS = """당신은 AI와 대화하며 계획을 세우는 사용자를 연기합니다.

규칙
1. 당신에게는 숨은 프로필이 있습니다. 한 번에 다 말하지 마세요.
2. 이번 턴에 공개할 항목만 자연스럽게 흘리세요. 공개 항목이 없으면
   반응·질문·간단한 수정 요청만 하세요.
3. 아직 공개하지 않은 항목은 절대 언급하지 마세요.
4. 사람이 채팅하듯 짧게 쓰세요. 두 문장을 넘기지 마세요.
5. 목록이나 번호를 쓰지 마세요. 평범한 구어체로 쓰세요.
6. AI가 당신이 말한 적 없는 내용을 사실처럼 단정했고 그게 당신 프로필과
   어긋나면, 그 부분만 짧게 바로잡으세요.

출력은 당신의 발화 한 덩어리만. 설명이나 따옴표를 붙이지 마세요."""


def _client():
    from dotenv import load_dotenv
    import anthropic

    load_dotenv(override=True)
    key = os.environ.get("Anthropic_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Anthropic_API_KEY 없음")
    return anthropic.Anthropic(api_key=key)


@dataclass
class SimState:
    """시뮬레이터가 턴 사이에 들고 가는 것."""
    disclosed: list[str] = field(default_factory=list)
    turn: int = 0
    changed: bool = False          # 조건 변경을 이미 제시했는가
    corrections: list[str] = field(default_factory=list)


class UserSimulator:
    """페르소나를 연기하며 정보를 조금씩 공개한다.

    파트너와 다른 계열 모델을 쓴다. 같은 모델이면 서로의 표현 습관에 맞춰져
    그라운딩 문제가 과소 평가된다. 파트너는 로컬 Qwen, 시뮬레이터는 API Claude다.

    attention_budget k: 파트너 응답에서 k개 항목까지만 검토한다. 사람이 긴
    산출물을 전부 읽지 않는다는 사실을 반영한 것이고, k를 바꿔 민감도를 본다.
    """

    def __init__(
        self,
        persona: Persona,
        model: str = "claude-haiku-4-5-20251001",
        attention_budget: int | None = 5,
        disclose_prob: float = 0.5,
        seed: int = 0,
    ):
        self.persona = persona
        self.model = model
        self.attention_budget = attention_budget
        self.disclose_prob = disclose_prob
        self.rng = random.Random(seed)
        self.state = SimState()
        self._client = _client()

    # ------------------------------------------------------------------ #
    @property
    def undisclosed(self) -> list[str]:
        return [x for x in self.persona.checklist if x not in self.state.disclosed]

    def _pick_disclosures(self, partner_asked: bool) -> list[str]:
        """이번 턴에 공개할 항목을 고른다.

        질문을 받으면 1~2개, 아니면 확률적으로 0~1개. 전부 쏟아내지 않는 것이
        설계의 핵심이다. 사용자가 다 말해 버리면 LLM이 해석을 만들 이유가 없고
        연구 문제가 사라진다.
        """
        pool = self.undisclosed
        if not pool:
            return []
        if partner_asked:
            n = min(len(pool), self.rng.choice([1, 2]))
        else:
            n = 1 if self.rng.random() < self.disclose_prob else 0
        return pool[:n]

    @staticmethod
    def _asked_question(text: str) -> bool:
        tail = (text or "")[-400:]
        return "?" in tail or "알려주" in tail or "말씀해" in tail

    def _focus(self, partner_text: str) -> str:
        """주의 예산만큼만 파트너 응답을 본다."""
        if self.attention_budget is None:
            return partner_text
        from app.segmenter import segment

        segs = [s for s in segment(partner_text, turn=self.state.turn)
                if s.kind != "heading"]
        if not segs:
            return partner_text[:600]
        k = min(self.attention_budget, len(segs))
        picked = segs[:k]
        return "\n".join(f"- {s.text}" for s in picked)

    # ------------------------------------------------------------------ #
    def first_turn(self) -> str:
        self.state.turn = 1
        return self.persona.opening

    def change_turn(self) -> str:
        """조건 변경을 제시하고 **자기 프로필을 갱신한다**.

        갱신하지 않으면 시뮬레이터가 자기 변경과 싸운다. 실측 사례:
        "헬스장은 못 끕을 것 같아"라고 말해 놓고, 패널에 "헬스장을 등록하지
        않는다"가 뜨자 옮은 프로필("주 4회 헬스장")과 다르다며 다시 "등록한다"로
        고쳐 버렸다. 그러면 어떤 조건도 추적할 수 없다.

        사람으로 치면 "예산이 줄었다"고 말한 다음부터는 줄어든 예산이 그 사람의
        현재 제약이다. 그걸 반영한다.
        """
        self.state.turn += 1
        self.state.changed = True
        p = self.persona
        # 무효화된 항목을 빼고 새 조건을 넣는다
        for dead in p.change_invalidates:
            key = set(re.findall(r"[가-훣]{2,}", dead))
            for bucket in (p.goals, p.constraints, p.preferences):
                for item in list(bucket):
                    hits = sum(1 for k in key if k in item)
                    if key and hits >= max(1, len(key) // 2):
                        bucket.remove(item)
        for added in p.change_implies:
            if added not in p.constraints:
                p.constraints.append(added)
        # 이미 공개한 목록에서도 사라진 항목을 뺀다
        self.state.disclosed = [x for x in self.state.disclosed if x in p.checklist]
        self.state.disclosed.extend(a for a in p.change_implies
                                    if a not in self.state.disclosed)
        return p.change_utterance

    def next_turn(self, partner_text: str) -> str:
        """파트너 응답을 보고 다음 발화를 만든다."""
        self.state.turn += 1
        asked = self._asked_question(partner_text)
        to_disclose = self._pick_disclosures(asked)
        self.state.disclosed.extend(to_disclose)

        p = self.persona
        meanings = "\n".join(f'- "{k}"는 당신에게 {v}라는 뜻입니다'
                             for k, v in p.private_meanings.items())
        already = "\n".join(f"- {x}" for x in self.state.disclosed) or "- (아직 없음)"
        nowdis = "\n".join(f"- {x}" for x in to_disclose) or "- (이번 턴에는 없음)"
        focus = self._focus(partner_text)

        user_msg = f"""[당신의 사적 의미]
{meanings or '- (없음)'}

[이미 말한 것]
{already}

[이번 턴에 말할 것]
{nowdis}

[AI가 방금 보여준 내용 중 당신이 본 부분]
{focus}

위 규칙에 따라 당신의 다음 발화를 쓰세요."""

        r = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            system=_SYS,
            messages=[{"role": "user", "content": user_msg}],
            extra_body={"temperature": 0.0},
        )
        METER.record("simulator.turn", self.model, r)
        text = (r.content[0].text or "").strip()
        text = re.sub(r'^["\u201c\']|["\u201d\']$', "", text).strip()
        return text or "응, 그렇게 해줘."

    # ------------------------------------------------------------------ #
    def checklist_report(self, final_text: str) -> dict[str, Any]:
        """최종 산출물이 숨은 프로필을 얼마나 충족했는지 LLM 판정자로 채점한다.

        판정자는 파트너·시뮬레이터와 또 다른 호출이다. 항목별 예/아니오만
        받아 로그로 남긴다. 사람 검증은 E2의 kappa 측정에서 따로 한다.
        """
        items = self.persona.checklist
        numbered = "\n".join(f"{i+1}. {x}" for i, x in enumerate(items))
        prompt = f"""아래 계획이 각 요구사항을 충족하는지 판정하세요.

[요구사항]
{numbered}

[계획]
{final_text[:4000]}

각 항목에 대해 충족이면 1, 아니면 0을 매기고 JSON 배열로만 답하세요.
예: [1,0,1,1,0,0,1,0]"""
        r = self._client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"temperature": 0.0},
        )
        METER.record("simulator.checklist", "claude-sonnet-4-5-20250929", r)
        raw = r.content[0].text or ""
        m = re.search(r"\[[^\]]*\]", raw, re.S)
        vals: list[int] = []
        if m:
            try:
                vals = [int(bool(v)) for v in json.loads(m.group())]
            except Exception:
                vals = []
        if len(vals) != len(items):
            vals = (vals + [0] * len(items))[:len(items)]
        return {
            "items": items,
            "scores": vals,
            "satisfied": sum(vals),
            "total": len(items),
            "rate": (sum(vals) / len(items)) if items else 0.0,
        }
