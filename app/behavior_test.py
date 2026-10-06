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

        # 다음 토큰 예측이므로 한 칸 민다
        logprobs_all = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = input_ids[0, 1:]
        token_lp = logprobs_all.gather(1, targets.unsqueeze(1)).squeeze(1)

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
