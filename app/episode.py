"""E2 에피소드 실행기 — 조건 B0~B4를 한 경로로 돌린다.

왜 한 경로인가:
B2·B3는 어블레이션이 목적이다. "어느 처리에서 이득이 생기는가"를 가르려면
조건 사이에 처리 외의 차이가 없어야 한다. 조건마다 함수를 따로 쓰면 프롬프트
구성이나 턴 진행이 미묘하게 달라져 비교가 오염된다. 스위치만 둔다.

조건별로 켜지는 것 (feedback2.md E2)
    B0  패널 없음. 교정은 대화로만
    B1  평평한 메모리 목록 + 편집                      <- 제품 메모리
    B2  + P1 검증 상태 구분 (확인/추측/정정/거부)
    B3  + P2 사용 추적 (행동 검사로 쓰인 LG만 표시)
    B4  + P3 파생 재검토 + 반영 재검사와 재시도

흐름
    턴1      페르소나의 모호한 첫 발화 -> 파트너가 계획 초안
    턴2~T-1  시뮬레이터가 프로필을 조금씩 공개 -> 파트너가 개정
    턴T      조건 변경 제시
    턴T+     최대 2턴 개정
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.conditions import (
    LEVEL,
    PLAN_TASK,
    assign_status,
    build_system,
    extract_memory_items,
    review_memory_items,
    review_panel,
    test_dependence,
)
from app.context_builder import build_context
from app.propagate import find_collateral, judge_premise, reflection_verdict
from app.segmenter import Segment, align, segment
from app.simulator import Persona, UserSimulator
from app.state_store import Evidence, StateStore


@dataclass
class EpisodeTrace:
    """에피소드 진행 기록. 지표 계산과 사후 진단에 쓴다."""
    plans: list[str] = field(default_factory=list)
    user_turns: list[str] = field(default_factory=list)
    panel_ops: int = 0
    panel_shown: list[int] = field(default_factory=list)
    status_counts: list[dict] = field(default_factory=list)
    verdict_counts: list[dict] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)
    propagations: list[dict] = field(default_factory=list)
    pre_change_plan: str = ""
    first_after_change: str = ""


class EpisodeRunner:
    """한 에피소드를 돌린다. 조건은 레벨 스위치로만 갈린다."""

    def __init__(self, bt, persona: Persona, condition: str, seed: int = 0,
                 attention: int | None = 5, placebos: list[str] | None = None,
                 build_turns: int = 3, revise_turns: int = 2,
                 score_history_turns: int = 2):
        self.score_history_turns = score_history_turns
        self.bt = bt
        self.persona = persona
        self.condition = condition
        self.lvl = LEVEL[condition]
        self.attention = attention
        self.placebos = placebos or []
        self.build_turns = build_turns
        self.revise_turns = revise_turns

        self.sim = UserSimulator(persona, attention_budget=attention, seed=seed)
        self.client = self.sim._client
        self.store = StateStore() if self.lvl >= 2 else None
        self.history: list[dict[str, str]] = []
        self.memory: list[str] = []            # B1 평평한 목록
        self.superseded: list[tuple[str, str]] = []   # B3+ H-mark 재료
        self.trace = EpisodeTrace()
        self._placebo_cache: list[dict[str, float]] | None = None

    # ------------------------------------------------------------------ #
    def _user_utterances(self) -> list[str]:
        return [m["content"] for m in self.history if m["role"] == "user"]

    def _partner(self, user_text: str) -> str:
        """파트너 응답. B2 이상은 상태에서 결정적으로 컨텍스트를 만든다."""
        if self.lvl >= 2:
            system, msgs = build_context(
                self.store, self.history, user_text, mode="H-mark",
                task_instruction=PLAN_TASK, superseded=self.superseded)
        else:
            system = build_system(self.condition, self.memory)
            msgs = [*self.history, {"role": "user", "content": user_text}]
        resp = self.bt.generate(msgs, system=system, max_new_tokens=900)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": resp})
        self.trace.user_turns.append(user_text)
        self.trace.plans.append(resp)
        return resp

    def _score_msgs(self) -> list[dict[str, str]]:
        """채점용 컨텍스트. 최근 턴만 쓴다.

        전체 기록을 넣으면 턴이 갈수록 채점이 느려진다(실측 0.11s → 0.47s).
        더 중요한 건 행동 검사가 묻는 게 "이 전제가 이 응답에 영향을 주는가"라
        전체 대화가 필요 없다는 점이다. 오히려 긴 기록은 개입 효과를 희석시킨다.
        E1-1·E1-7·M3 검증을 전부 짧은 컨텍스트에서 했으므로 일관성도 맞는다.
        """
        msgs = self.history[:-1] if self.history else []
        if not msgs:
            return [{"role": "user", "content": self.persona.opening}]
        keep = self.score_history_turns * 2
        out = msgs[-keep:] if keep > 0 else msgs[-1:]
        # 첫 메시지가 assistant면 짝이 깨진다
        if out and out[0]["role"] == "assistant":
            out = out[1:]
        return out or [{"role": "user", "content": self.persona.opening}]

    def _placebo_deltas(self, plan: str, segs: list[Segment]) -> list[dict[str, float]]:
        """위약 Δ 행렬. 턴마다 한 번 계산해 모든 전제가 공유한다.

        에피소드 시간의 81%가 여기서 나온다(실측 1565회 채점 중 1400회가 위약).
        검증 실험(E1)은 200개를 써야 거짓양성이 보장되지만, B3·B4의 운용
        용도(표시할 LG 추림, 영향 범위 찾기)는 그보다 느슨해도 된다.
        """
        msgs = self._score_msgs()
        base = self.bt.score_segments(msgs, plan, segs)
        return [self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=pl)) for pl in self.placebos]

    # ------------------------------------------------------------------ #
    def _panel_cycle(self, plan: str, turn: int) -> list[tuple[str, str]]:
        """패널 추출·상태부여·검사·검토. 레벨에 따라 단계가 늘어난다."""
        if self.lvl == 0:
            return []

        items = extract_memory_items(self.client, self.history, plan)

        if self.lvl == 1:
            self.memory, ops = review_memory_items(
                self.client, items, self.persona, self.attention)
            self.trace.panel_ops += ops
            self.trace.panel_shown.append(len(items))
            return []

        # B2 이상: 상태 부여
        counts = assign_status(self.store, items, self._user_utterances(), turn)
        self.trace.status_counts.append(counts)

        # B3 이상: 행동 검사로 쓰인 LG만 추린다
        if self.lvl >= 3 and self.placebos:
            segs = [s for s in segment(plan, turn=turn) if s.start >= 0]
            if len(segs) >= 4:
                self._placebo_cache = self._placebo_deltas(plan, segs)
                msgs = self._score_msgs()
                vc = test_dependence(self.bt, msgs, plan, segs, self.store,
                                     self._placebo_cache, turn)
                self.trace.verdict_counts.append(vc)

        res = review_panel(self.client, self.store, self.persona,
                           self.attention, turn,
                           only_operative=(self.lvl >= 3))
        self.trace.panel_ops += res["ops"]
        self.trace.panel_shown.append(res["shown"])
        return res["corrections"]

    def _apply_corrections(self, corrections: list[tuple[str, str]],
                           plan: str, turn: int) -> None:
        """수정을 상태에 반영한다. B4는 전파와 반영 재검사까지 한다 (M5)."""
        for pid, new_text in corrections:
            old = self.store.premises.get(pid)
            if old is None:
                continue
            self.trace.corrections.append((old.text, new_text))

            if self.lvl < 4:
                self.store.correct(pid, new_text, Evidence(
                    type="ui_correction", turn=turn, pointer=f"c{turn}"))
                continue

            # B4: 영향 찾기 -> 파생 재검토 -> 정정 -> (재생성은 다음 턴에)
            info: dict[str, Any] = {"premise": old.text, "new": new_text}
            segs = [s for s in segment(plan, turn=turn) if s.start >= 0]
            if self._placebo_cache and len(segs) >= 4:
                msgs = self._score_msgs()
                base = self.bt.score_segments(msgs, plan, segs)
                d_a = self.bt.delta(base, self.bt.score_segments(
                    msgs, plan, segs, assumption=old.text))
                d_c = self.bt.delta(base, self.bt.score_segments(
                    msgs, plan, segs, assumption=new_text))
                v = judge_premise(d_a, [d_c], self._placebo_cache)
                by_id = {s.id: s for s in segs}
                info["affected"] = len(v.segments)
                # 영향 세그먼트를 H-mark 재료로 넘긴다. 과거 응답에 남은 옛 값이
                # 상태 블록을 이기는 것을 막는 유일한 수단이다.
                for sid in v.segments:
                    if sid in by_id:
                        self.superseded.append((by_id[sid].text, new_text))
            self.store.correct(pid, new_text, Evidence(
                type="ui_correction", turn=turn, pointer=f"c{turn}"))
            info["flagged"] = len(self.store.flag_derived(pid, depth=2))
            self.trace.propagations.append(info)

    def _handle_condition_change(self, plan: str, turn: int) -> None:
        """조건 변경을 정정으로 취급해 전파를 걸어 준다 (B4, M5).

        문서 M5 마지막 줄: "조건 변경(예: 예산 축소)도 CG 전제의 수정이므로
        같은 절차를 탄다." 처음엔 전파를 패널 편집에서만 발동하게 만들었는데,
        조건 변경은 대화로 들어오므로 전파가 아예 걸리지 않았다.

        실측 사례: "예산이 줄어서 헬스장은 못 끕을 것 같아"라고 했는데 패널에서
        사용자가 고친 건 운동 용품 예산 액수였다. 헬스장 자체는 전제로 등록되지
        않아 전파 대상이 아니었고, 최종 계획이 매일 헬스장을 유지해 잔존율 100%가 됐다.

        어느 전제가 무효화됐는지 고르지 않는다. 필요한 건 **영향 받는 세그먼트**이고,
        그건 변경 발화 자체를 반대 값으로 써서 직접 찾을 수 있다(M5 1단계).
        """
        if self.lvl < 4 or not self.placebos:
            return
        segs = [s for s in segment(plan, turn=turn) if s.start >= 0]
        if len(segs) < 4:
            return
        change = self.persona.change_utterance
        implies = " ".join(self.persona.change_implies) or change

        msgs = self._score_msgs()
        base = self.bt.score_segments(msgs, plan, segs)
        P = self._placebo_cache or [
            self.bt.delta(base, self.bt.score_segments(
                msgs, plan, segs, assumption=pl)) for pl in self.placebos]
        # 반대 = 변경 후 상태, 단언 = 변경 전 상태가 유지된다는 명제
        invalid = " ".join(self.persona.change_invalidates)
        assert_text = invalid or "지금까지의 조건은 그대로다"
        d_a = self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=assert_text))
        d_c = self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=implies))
        v = judge_premise(d_a, [d_c], P)
        by_id = {s.id: s for s in segs}
        for sid in v.segments:
            if sid in by_id:
                self.superseded.append((by_id[sid].text, implies))

        # 변경 내용을 확인된 전제로 올리고, 충돌하는 기존 전제를 정정한다.
        from app.state_store import Premise
        from app.graph import _content_tokens, _support_ratio, _SUPPORT_THRESHOLD

        ft = _content_tokens(change + " " + implies)
        for text in self.persona.change_implies:
            pid = f"chg{turn}_{abs(hash(text)) % 9999:04d}"
            if pid in self.store.premises:
                continue
            self.store.add(Premise(id=pid, text=text, origin="session"))
            self.store.ground(pid, Evidence(type="user_statement", turn=turn,
                                            pointer=f"chg{turn}"))
        # 변경과 맞붙는 기존 전제를 정정 처리
        for p in list(self.store.by_status("grounded", "proposed")):
            if p.id.startswith("chg"):
                continue
            if _support_ratio(p.text, ft) >= _SUPPORT_THRESHOLD:
                self.store.correct(p.id, implies, Evidence(
                    type="user_correction", turn=turn, pointer=f"chg{turn}",
                    attribution="estimated"))
                self.store.flag_derived(p.id, depth=2)
        self.trace.propagations.append(
            {"source": "condition_change", "affected": len(v.segments),
             "change": change[:50]})

    def _recheck_reflection(self, plan: str, turn: int) -> None:
        """재생성 결과가 정정을 반영했는지 검사하고 1회 재시도한다 (B4, M5 4단계)."""
        if self.lvl < 4 or not self.trace.corrections or not self.placebos:
            return
        segs = [s for s in segment(plan, turn=turn) if s.start >= 0]
        if len(segs) < 4:
            return
        old_text, new_text = self.trace.corrections[-1]
        msgs = self._score_msgs()
        base = self.bt.score_segments(msgs, plan, segs)
        P = [self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=pl)) for pl in self.placebos]
        d_new = self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=new_text))
        d_old = self.bt.delta(base, self.bt.score_segments(
            msgs, plan, segs, assumption=old_text))
        ok, stat, stale = reflection_verdict(d_old, d_new, P)
        rec = {"reflected": ok, "stat": stat, "n_stale": len(stale), "retried": False}
        if not ok:
            # 재시도: 반영되지 않은 부분을 명시해 다시 생성한다
            hint = (f"이전 응답이 다음 정정을 반영하지 않았습니다. "
                    f"반드시 반영해서 계획 전체를 다시 작성하세요: {new_text}")
            retry = self._partner(hint)
            rec["retried"] = True
            rec["retry_len"] = len(retry)
        self.trace.propagations.append(rec)

    # ------------------------------------------------------------------ #
    def run(self) -> EpisodeTrace:
        turn = 1
        plan = self._partner(self.sim.first_turn())

        for _ in range(self.build_turns):
            turn += 1
            corr = self._panel_cycle(plan, turn)
            self._apply_corrections(corr, plan, turn)
            plan = self._partner(self.sim.next_turn(plan))

        self.trace.pre_change_plan = plan

        # 조건 변경
        turn += 1
        corr = self._panel_cycle(plan, turn)
        self._apply_corrections(corr, plan, turn)
        # 변경을 파트너에게 넘기기 전에 전파를 걸어 둔다.
        # 전전 응답을 대상으로 영향 범위를 찾아야 옮은 값이 기록에 남은 채로
        # 상태 블록을 이기는 걸 막을 수 있다.
        self._handle_condition_change(plan, turn)
        plan = self._partner(self.sim.change_turn())
        self.trace.first_after_change = plan
        self._recheck_reflection(plan, turn)

        for _ in range(self.revise_turns):
            turn += 1
            corr = self._panel_cycle(plan, turn)
            self._apply_corrections(corr, plan, turn)
            plan = self._partner(self.sim.next_turn(plan))

        return self.trace
