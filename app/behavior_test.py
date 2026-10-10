"""M3. 행동 검사 — 로그확률로 전제 의존을 판정한다.

왜 로그확률인가:
이전 설계(A-V3)는 개입 후 계획을 다시 생성해 변화량을 쟀다. 실측 결과 개입 없이
반복 생성만 해도 잡음 바닥이 0.20~0.32로 커서, 위약과 실제 전제의 분리가 거의
사라졌다. 생성은 비결정적이고 자유도가 너무 크다.

로그확률 방식은 다시 생성하지 않는다. **이미 나온 응답 문장이 조건에 따라 얼마나
덜 그럴듯해지는지**를 잰다. 가중치·dtype·입력이 같으면 결과가 같으므로 재현 가능하다.

측정량:
    Δ(s, X) = [log p(s | 원래 컨텍스트) − log p(s | 원래 컨텍스트 + X)] / 토큰 수

판정(제안서 M3을 수정):
    기댐 = Δ(반대)가 위약들의 **전체 최대 Δ 분포**보다 크고, Δ(단언)은 그 이하

제안서 원안은 세그먼트별 순열 + BH FDR이었으나, 위약 19개면 순열 p의 하한이
1/20 = 0.05라 BH가 요구하는 (1/m)·q에 도달할 수 없다. 모의 결과 세그먼트별
순열은 귀무가설에서 거짓양성이 0.92까지 치솟았다. max-T(Westfall-Young)로 바꾸면
같은 위약 수로 다중비교가 통제된다(거짓양성 0.046).

주의: 이 검사는 "LLM이 이 전제를 쓰고 있다"만 말한다. "사용자와 확인되었다"가
아니므로 **Common Ground 승격 근거로 쓰면 안 된다**(제안서 원칙 2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# 개입 블록. 위치를 고정해야 길이·위치 효과가 위약과 같은 조건이 된다.
ASSUMPTION_BLOCK = "\n\n【작업 가정】\n{text}\n"

_MODEL_DIR = os.environ.get("GL_LOCAL_MODEL", "C:/seungyeol/vscode/GL/.models/Qwen3-8B")


@dataclass
class ScoredSegment:
    seg_id: str
    n_tokens: int
    mean_logprob: float


class BehaviorTester:
    """로컬 모델로 세그먼트 로그확률을 채점한다.

    모델 로딩이 비싸므로 한 번 만들어 재사용한다. 채점은 순전파 한 번으로
    전체 응답을 훑고 세그먼트 구간별로 잘라 평균을 낸다. 세그먼트마다 따로
    돌리면 N배 느려지는데, 응답이 100세그먼트면 감당할 수 없다.
    """

    def __init__(self, model_dir: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_dir = model_dir or _MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=getattr(torch, dtype),
            device_map="cuda:0",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------ #
    def build_prompt(self, messages: list[dict], assumption: str | None = None) -> str:
        """대화 기록 → 프롬프트 문자열. 작업 가정은 응답 직전 고정 위치에 넣는다."""
        msgs = [dict(m) for m in messages]
        if assumption and assumption.strip():
            # 마지막 사용자 턴 뒤에 붙여 위치를 고정한다
            msgs.append({
                "role": "user",
                "content": ASSUMPTION_BLOCK.format(text=assumption.strip()).strip(),
            })
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate(
        self,
        messages: list[dict],
        system: str | None = None,
        max_new_tokens: int = 900,
    ) -> str:
        """파트너 응답을 생성한다. 채점과 같은 모델이어야 로그확률이 의미를 갖는다.

        greedy 고정. 생성 편차가 조건 간 비교를 오염시키면 안 된다.
        """
        torch = self.torch
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def score_segments(
        self,
        messages: list[dict],
        response_text: str,
        segments: list[Any],
        assumption: str | None = None,
    ) -> dict[str, ScoredSegment]:
        """응답을 teacher forcing으로 채점하고 세그먼트별 평균 로그확률을 낸다.

        segments 는 app.segmenter.Segment (start/end 문자 구간 필요).
        """
        torch = self.torch
        prompt = self.build_prompt(messages, assumption)

        enc_prompt = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)
        n_prompt = len(enc_prompt["input_ids"])

        enc_resp = self.tokenizer(
            response_text,
            return_tensors=None,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        resp_ids = enc_resp["input_ids"]
        offsets = enc_resp["offset_mapping"]

        input_ids = torch.tensor(
            [enc_prompt["input_ids"] + resp_ids], device=self.device
        )

        with torch.no_grad():
            logits = self.model(input_ids).logits

        # 다음 토큰 예측이므로 한 칸 민다.
        #
        # log_softmax를 전체 시퀀스에 한 번에 걸면 [L, V] float32 텐서가 두 개
        # 뜨다. 어휘가 15만이라 L=4000이면 2.4GB씩, 모델 16.8GB를 올린
        # 24GB 카드에서 OOM이 난다. 실제로 하루에 터졌다.
        #
        # cross_entropy는 정답 토큰의 음의 로그확률을 바로 내주고 분포를
        # 재료화하지 않는다. 청크로 나눔 것까지 하면 최대 메모리가
        # chunk × V로 제한된다.
        targets = input_ids[0, 1:]
        shifted = logits[0, :-1]
        parts = []
        chunk = 512
        for i in range(0, shifted.shape[0], chunk):
            part = shifted[i:i + chunk].float()
            tgt = targets[i:i + chunk]
            parts.append(-torch.nn.functional.cross_entropy(
                part, tgt, reduction="none"))
            del part
        token_lp = torch.cat(parts) if parts else torch.empty(0, device=self.device)
        del logits, shifted, parts

        # 응답 토큰 j (0-based) 의 로그확률은 token_lp[n_prompt - 1 + j]
        out: dict[str, ScoredSegment] = {}
        for seg in segments:
            s, e = getattr(seg, "start", -1), getattr(seg, "end", -1)
            if s < 0 or e <= s:
                continue
            idxs = [
                j for j, (a, b) in enumerate(offsets)
                if b > a and a < e and b > s          # 구간이 겹치는 토큰
            ]
            if not idxs:
                continue
            pos = [n_prompt - 1 + j for j in idxs]
            pos = [p for p in pos if 0 <= p < token_lp.shape[0]]
            if not pos:
                continue
            vals = token_lp[pos]
            out[seg.id] = ScoredSegment(
                seg_id=seg.id,
                n_tokens=len(pos),
                mean_logprob=float(vals.mean().item()),
            )
        return out

    def delta(
        self,
        base: dict[str, ScoredSegment],
        perturbed: dict[str, ScoredSegment],
    ) -> dict[str, float]:
        """Δ(s, X) = base − perturbed. 양수면 개입이 그 문장을 덜 그럴듯하게 만든 것."""
        return {
            k: base[k].mean_logprob - perturbed[k].mean_logprob
            for k in base if k in perturbed
        }


# --------------------------------------------------------------------------- #
# 판정 — 스튜던트화 max-T (Westfall-Young)
#
# feedback2.md M3 원안은 "세그먼트별 순열 + BH FDR"이었다. 위약 19개면 순열 p의
# 하한이 1/20 = 0.05라 BH가 요구하는 (1/m)·q에 도달할 수 없고, 모의에서 귀무가설
# 거짓양성이 0.72~0.92까지 치솟았다. max-T로 바꾸면 통제된다.
#
# 세 가지를 보정했다. 실측 근거는 exp_m3b.py와 test_behavior_judge.py에 있다.
#
#   1) 스튜던트화
#      세그먼트마다 위약에 대한 민감도가 다르다(실측 표준편차 0.010~0.163, 5.4배).
#      원시 Δ로 max-T를 하면 가장 불안정한 세그먼트가 임계값을 독점한다.
#
#   2) 검정 점을 영분포 풀에 넣고 전부 leave-one-out으로 채점
#      표본 모멘트로 표준화하면 영분포 점들은 자기 모멘트에 기여해 축소되는데
#      검정 점은 아니다. 교환가능성이 깨져 거짓양성이 0.8% → 2.8%로 부푼다.
#      검정 점을 풀에 포함시켜 101개 전부를 LOO로 채점하면 1.2%로 복원된다.
#
#   3) 반대 조건은 "하나라도"
#      반대 값 2개는 의미가 서로 다르므로 같은 세그먼트를 똑같이 흔들 이유가 없다.
#      둘 다 요구하면 교집합만 남아 작동 전제를 놓친다(실측 0/3 → 3/3).
# --------------------------------------------------------------------------- #

@dataclass
class Verdict:
    """한 전제에 대한 판정 결과."""
    verdict: str                       # operative | inert | inconclusive
    segments: list[str]                # 기댄 세그먼트 id
    z_by_segment: dict[str, float]     # 세그먼트별 최대 z (반대 조건 기준)
    threshold: float
    alpha: float                       # 달성된 FWER 상한


def _loo_z_matrix(pool: list[dict[str, float]], seg_ids: list[str]) -> list[dict[str, float]]:
    """풀의 각 점을 나머지로 표준화한다.

    합과 제곱합을 미리 구해 O(B×S)로 계산한다. 소박하게 매번 평균을 다시
    구하면 O(B²×S)가 되어 위약을 늘릴 수 없다.
    """
    n = len(pool)
    out: list[dict[str, float]] = [dict() for _ in range(n)]
    if n < 4:
        return out
    for sid in seg_ids:
        vals = [p.get(sid) for p in pool]
        present = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(present) < 4:
            continue
        s = sum(v for _, v in present)
        sq = sum(v * v for _, v in present)
        m = len(present)
        for i, v in present:
            mo = (s - v) / (m - 1)
            vo = (sq - v * v) / (m - 1) - mo * mo
            sdo = (vo ** 0.5) if vo > 0 else 1e-9
            out[i][sid] = (v - mo) / sdo
    return out


def judge_premise(
    assert_delta: dict[str, float],
    contra_deltas: list[dict[str, float]],
    placebo_deltas: list[dict[str, float]],
) -> Verdict:
    """전제 하나를 판정한다.

    기댐 조건 (세그먼트 s):
        반대 중 하나라도  z(s) > 임계값
        그리고 단언은     z(s) ≤ 임계값

    단언 조건이 있는 이유는, 대체만 보면 "아무 가정이나 넣으면 흔들린다"는 효과와
    구분되지 않기 때문이다. 단언했을 때 변하지 않아야 "이미 그 전제를 쓰고 있었다"가
    된다. 둘 다 크면 개입 자체가 생성을 교란한 것이므로 inconclusive로 둔다.

    임계값은 반대 조건마다 따로 구한다. 그 조건을 위약 풀에 넣고 101개를 전부
    LOO로 채점한 뒤, 위약 쪽 최대 z들의 최대를 쓴다. FWER ≈ 1/(B+1).
    """
    seg_ids = sorted({sid for d in [assert_delta, *contra_deltas] for sid in d}
                     & {sid for d in placebo_deltas for sid in d})
    B = len(placebo_deltas)
    alpha = 1.0 / (B + 1) * max(len(contra_deltas), 1)   # 반대 조건 수만큼 union bound

    if not seg_ids or B < 4:
        return Verdict("inert", [], {}, float("inf"), alpha)

    def scored(target: dict[str, float]) -> tuple[dict[str, float], float]:
        """target을 위약 풀에 넣고 전부 LOO 채점. (target의 z, 위약 max-T 임계값)"""
        pool = [*placebo_deltas, target]
        zs = _loo_z_matrix(pool, seg_ids)
        null_max = [max(z.values()) for z in zs[:-1] if z]
        thr = max(null_max) if null_max else float("inf")
        return zs[-1], thr

    z_assert, thr_a = scored(assert_delta)
    contra_scored = [scored(cd) for cd in contra_deltas]

    hits: list[str] = []
    zmap: dict[str, float] = {}
    disturbed = 0
    thr = max([thr_a] + [t for _, t in contra_scored])

    for sid in seg_ids:
        zc = [z.get(sid, float("-inf")) for z, _ in contra_scored]
        if not zc:
            continue
        zmap[sid] = max(zc)
        over = [zv > t for zv, (_, t) in zip(zc, contra_scored)]
        if any(over):
            if z_assert.get(sid, float("-inf")) <= thr_a:
                hits.append(sid)
            else:
                disturbed += 1

    if hits:
        v = "operative"
    elif disturbed:
        v = "inconclusive"
    else:
        v = "inert"
    return Verdict(v, hits, zmap, thr, alpha)
