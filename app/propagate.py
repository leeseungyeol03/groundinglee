"""M5. 수정 전파와 반영 검증 — P3이며 원칙 3의 구현.

왜 이게 핵심인가:
제품 메모리는 항목을 고치면 **다음 주입부터** 바뀐다. 고친 내용이 기존 결과물에
어떻게 퍼져 있었는지, 재생성한 결과에 실제로 반영됐는지는 알려주지 않는다.
M5는 그 둘을 한다. 메모리식 기준선(E2의 B1)과 가장 크게 갈리는 지점이다.

원칙 3(정정도 수락의 증거가 필요하다): 정정을 보냈다고 끝나지 않는다. 지금
시스템과 제품 메모리는 자기가 보낸 정정에 대해 추정 그라운딩(presumptive
grounding)을 한다. 고쳐졌다고 가정하고 넘어간다. M5는 고쳐졌는지 **검사**한다.

절차 (사용자가 p를 p'로 수정한 경우, feedback2.md M5)
    1. 영향 찾기    반대 값 = p'로 행동 검사 → p에 기댄 기존 세그먼트를 superseded로
    2. 파생 처리    p에서 파생된 LG 전제에 재검토 표시 (자동 수정 금지, 깊이 2)
    3. 재생성       M4로 만든 컨텍스트에서 새 응답 생성
    4. 반영 재검사  새 응답에 do(p')를 걸어 유의하게 흔들리는 세그먼트를 찾는다.
                   흔들리면 아직 옛 p에 기대고 있다는 뜻이다.
    5. 함께 바뀐 부분  p에 기대지 않았는데 바뀐 세그먼트를 표시 (자동 복구 금지)

4번 논리를 풀어 쓰면: 새 응답의 어떤 문장이 p'를 제대로 반영했다면, p'를 다시
단언해도 그 문장은 흔들리지 않는다. 이미 그렇게 쓰여 있으니까. 반대로 옛 p에
그대로 기대고 있으면 p'를 단언하는 순간 모순이 생겨 확률이 떨어진다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.behavior_test import judge_premise
from app.segmenter import Segment, align
from app.state_store import Evidence, StateStore


# --------------------------------------------------------------------------- #
# 유의성 판정 — 단일 개입용
# --------------------------------------------------------------------------- #

def _loo_moments(values: list[float], k: int) -> tuple[float, float]:
    n = len(values)
    s = sum(values)
    sq = sum(v * v for v in values)
    m = (s - values[k]) / (n - 1)
    var = (sq - values[k] * values[k]) / (n - 1) - m * m
    return m, ((var ** 0.5) if var > 0 else 1e-9)


def significant_segments(
    delta: dict[str, float],
    placebo_deltas: list[dict[str, float]],
) -> tuple[list[str], dict[str, float], float]:
    """개입 하나가 유의하게 흔든 세그먼트를 찾는다.

    판정은 행동 검사와 같은 스튜던트화 max-T다. 반대·단언 쌍이 아니라
    개입 한 개만 보므로 judge_premise를 쓸 수 없어 따로 둔다.

    검정 점을 위약 풀에 넣고 전부 LOO로 채점한다. 그러지 않으면 위약 점들만
    자기 모멘트에 기여해 축소되어 교환가능성이 깨진다(거짓양성 0.8% → 2.8%).

    반환: (유의한 세그먼트, 세그먼트별 z, 임계값)
    """
    seg_ids = sorted(set(delta) & {s for d in placebo_deltas for s in d})
    if not seg_ids or len(placebo_deltas) < 4:
        return [], {}, float("inf")

    pool = [*placebo_deltas, delta]
    z_pool: list[dict[str, float]] = [dict() for _ in pool]
    for sid in seg_ids:
        vals = [p.get(sid) for p in pool]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        for i in range(len(pool)):
            m, sd = _loo_moments(vals, i)
            z_pool[i][sid] = (vals[i] - m) / sd

    null_max = [max(z.values()) for z in z_pool[:-1] if z]
    thr = max(null_max) if null_max else float("inf")
    z_obs = z_pool[-1]
    hits = sorted((sid for sid, z in z_obs.items() if z > thr),
                  key=lambda s: -z_obs[s])
    return hits, z_obs, thr


def reflection_verdict(
    delta_old: dict[str, float],
    delta_new: dict[str, float],
    placebo_deltas: list[dict[str, float]],
) -> tuple[bool, float, list[str]]:
    """정정이 반영됐는지 짝지어 판정한다.

    단일 개입의 절대 기준은 쓰지 않는다. 실제로 재보니 p'를 제대로
    반영한 응답에도 p'를 단언하면 세그먼트 0.7개가 흔렸다. 주제가 같은
    문장을 덧붙이는 행위 자슴이 지역 분포를 바꾸기 때문이다. 위약은
    "무관한 문장 삽입"을 통제하지만 "주제가 같은 문장 삽입"을 통제하지 못한다.

    대슴 옇 값과 새 값을 모두 단언해 보고 어느 쪽이 더 충돌하는지 본다.
        반영된 응답(p'을 담은)   → 옇 값 p가 더 충돌한다
        반영 안 된 응답(p를 담은) → 새 값 p'이 더 충돌한다
    개입 자슴의 일반 교란은 양쪽에 똑같이 작용해 상쇄된다.

    반환: (반영됨, 통계값(양수면 반영), 새 값에 충돌하는 세그먼트)
    """
    stale_new, z_new, _t1 = significant_segments(delta_new, placebo_deltas)
    _s_old, z_old, _t2 = significant_segments(delta_old, placebo_deltas)
    if not z_new or not z_old:
        return False, 0.0, list(stale_new)
    common = set(z_new) & set(z_old)
    if not common:
        return False, 0.0, list(stale_new)
    # 옇 값이 새 값보다 강하게 충돌하면 반영된 것이다
    stat = max(z_old[s] for s in common) - max(z_new[s] for s in common)
    return stat > 0.0, stat, list(stale_new)


# --------------------------------------------------------------------------- #
@dataclass
class PropagationResult:
    """한 번의 수정 전파 결과. 패널과 분석이 같이 쓴다."""
    premise_id: str
    old_text: str
    new_text: str
    # 1단계
    affected: list[str] = field(default_factory=list)        # 옛 p에 기댄 기존 세그먼트
    affected_texts: list[str] = field(default_factory=list)
    # 2단계
    flagged_premises: list[str] = field(default_factory=list)
    # 4단계
    reflected: bool = False
    unreflected: list[str] = field(default_factory=list)     # 아직 옛 p에 기댄 새 세그먼트
    unreflected_texts: list[str] = field(default_factory=list)
    retried: bool = False
    # 5단계
    collateral: list[str] = field(default_factory=list)      # 함께 바뀐 부분
    collateral_texts: list[str] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def status(self) -> str:
        if self.reflected:
            return "반영 확인"
        return "반영 안 됨"

    def to_dict(self) -> dict[str, Any]:
        return {
            "premise_id": self.premise_id,
            "old_text": self.old_text, "new_text": self.new_text,
            "affected": list(self.affected), "affected_texts": list(self.affected_texts),
            "flagged_premises": list(self.flagged_premises),
            "reflected": self.reflected, "status": self.status,
            "unreflected": list(self.unreflected),
            "unreflected_texts": list(self.unreflected_texts),
            "retried": self.retried,
            "collateral": list(self.collateral),
            "collateral_texts": list(self.collateral_texts),
            "threshold": self.threshold,
        }


def find_collateral(
    old_segments: list[Segment],
    new_segments: list[Segment],
    affected_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """p에 기대지 않았는데 바뀐 세그먼트를 찾는다 (5단계).

    M0 정렬로 옛 세그먼트와 새 세그먼트를 맞춘 뒤, 영향 범위 밖인데 내용이
    바뀐 것을 모은다. 자동으로 되돌리지 않는다. 사용자가 판단한다.

    이 지표가 중요한 이유는, 한 전제를 고쳤을 때 무관한 부분까지 흔들리면
    사용자가 결과물을 다시 전부 검토해야 하기 때문이다. E2의 '부수 변경률'이
    이걸 센다.
    """
    aff = set(affected_ids)
    r = align(old_segments, new_segments)
    old_by_id = {s.id: s for s in old_segments}
    new_by_id = {s.id: s for s in new_segments}

    ids: list[str] = []
    texts: list[str] = []
    for pair in r["pairs"]:
        if not pair["changed"]:
            continue
        if pair["prev"] in aff or pair["curr"] in aff:
            continue
        ids.append(pair["curr"])
        o = old_by_id.get(pair["prev"])
        n = new_by_id.get(pair["curr"])
        texts.append(f"{(o.text if o else '')[:40]} → {(n.text if n else '')[:40]}")
    return ids, texts


# --------------------------------------------------------------------------- #
# 전파 절차 전체
# --------------------------------------------------------------------------- #

def propagate_correction(
    tester,
    store: StateStore,
    premise_id: str,
    new_text: str,
    messages: list[dict[str, str]],
    old_response: str,
    old_segments: list[Segment],
    placebo_deltas: list[dict[str, float]],
    placebos: list[str],
    regenerate: Callable[[], tuple[str, list[Segment]]],
    turn: int = 0,
    evidence_type: str = "ui_correction",
    max_retry: int = 1,
) -> PropagationResult:
    """수정 전파와 반영 검증 전체 절차 (feedback2.md M5 1~5단계).

    regenerate 는 M4로 만든 컨텍스트에서 새 응답을 내는 콜백이다. 호출자가
    주입하게 둔 이유는, M5가 컨텍스트 구성 방식(H-full/H-mark/H-state)에
    묶이면 E2에서 변형을 비교할 수 없기 때문이다.

    placebo_deltas 는 **옛 응답**에 대한 위약 Δ 행렬이다. 새 응답은 세그먼트가
    달라지므로 4단계에서 다시 계산해야 한다.
    """
    old = store.premises[premise_id]
    res = PropagationResult(premise_id=premise_id, old_text=old.text, new_text=new_text)

    # ── 1단계. 영향 찾기 ────────────────────────────────────────
    # 반대 값 = p'(새 값), 단언 = p(옛 값). p에 기대던 세그먼트가 잡힌다.
    base = tester.score_segments(messages, old_response, old_segments)
    d_assert = tester.delta(
        base, tester.score_segments(messages, old_response, old_segments,
                                    assumption=old.text))
    d_contra = tester.delta(
        base, tester.score_segments(messages, old_response, old_segments,
                                    assumption=new_text))
    v = judge_premise(d_assert, [d_contra], placebo_deltas)
    res.affected = list(v.segments)
    res.threshold = v.threshold
    by_id = {s.id: s for s in old_segments}
    res.affected_texts = [by_id[s].text for s in res.affected if s in by_id]

    # 상태 기록: 영향받은 세그먼트는 superseded, 전제는 corrected
    store.set_verdict(premise_id, v.verdict, turn, res.affected)
    for sid in res.affected:
        if sid in by_id:
            by_id[sid].state = "superseded"

    ev = Evidence(type=evidence_type, turn=turn, pointer=f"prop_{premise_id}",
                  attribution="designated" if evidence_type.startswith("ui_") else "estimated")
    store.correct(premise_id, new_text, ev)

    # ── 2단계. 파생 전제 재검토 표시 (자동 수정 금지) ──────────
    res.flagged_premises = store.flag_derived(premise_id, depth=2)

    # ── 3~4단계. 재생성과 반영 재검사 ──────────────────────────
    attempt = 0
    while True:
        new_response, new_segments = regenerate()
        new_base = tester.score_segments(messages, new_response, new_segments)
        new_placebos = [
            tester.delta(new_base,
                         tester.score_segments(messages, new_response, new_segments,
                                               assumption=pl))
            for pl in placebos
        ]
        d_new = tester.delta(
            new_base, tester.score_segments(messages, new_response, new_segments,
                                            assumption=new_text))
        # p'를 단언했는데 흔들리면 아직 옛 p에 기대고 있다는 뜻이다
        stale, _z, thr = significant_segments(d_new, new_placebos)
        nb = {s.id: s for s in new_segments}
        res.unreflected = list(stale)
        res.unreflected_texts = [nb[s].text for s in stale if s in nb]
        res.reflected = not stale
        res.threshold = thr

        # 5단계. 함께 바뀐 부분
        res.collateral, res.collateral_texts = find_collateral(
            old_segments, new_segments, res.affected)

        if res.reflected or attempt >= max_retry:
            break
        attempt += 1
        res.retried = True

    return res
