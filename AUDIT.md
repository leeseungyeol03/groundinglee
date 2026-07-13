# GroundLens 도메인 결합 감사 보고서

> Phase 1 산출물. 코드베이스 기준: `d:\leesy2026\CHI27\CHI27_v2`
> 조사 기준: 여행 도메인 하드코딩 vs. 도메인 중립 판별

---

## A. 결합 지점 전수 목록

### ① 파트너 LLM 시스템 프롬프트

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| `config.yaml` | 8–10 | `neutral_system_prompt`: "사용자가 까다로운 제약이 있는 여행 계획을 세울 수 있도록 돕는 협업형 AI 파트너" | 하드코딩(설정값) |
| `config.yaml` | 13–15 | `task_context`: "사용자는 단독 챗봇과 협업하여 여행 계획을 수립한다." | 하드코딩(설정값) |
| `app/config.py` | 19–51 | `assemble_partner_system_prompt()`: corrections/deletions를 주입하는 조립 함수. 자체 도메인 문구 없음. | 중립(조립기) |

> **판정**: 여행 문구 2건 — `config.yaml` 설정값 교체로 해결.

---

### ② Shadow Probe 프롬프트

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| `classification/prompts.py` | 8–14 | `SHADOW_PROBE_PROMPT`: goal/constraints/term_interpretation 3필드. 도메인 문구 없음. | **도메인 중립** |
| `config.yaml` | 17–24 | `shadow_probe_prompt`: 동일 내용의 YAML 사본. 문구 동일. | **도메인 중립** |
| `app/llm.py` | 73–77 | mock `ProbeResult`: `"goal": "[MOCK] 제약을 만족하는 여행 계획 만들기"`, `"여행 계획": "일정, 동선…"` | 하드코딩(mock) |

> **판정**: 실제 프롬프트는 중립. Mock 데이터만 교체 필요.

---

### ③ CG/LG 분류 프롬프트

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| `classification/prompts.py` | 20–39 | `PREPROCESS_UTTERANCE_PROMPT`: 도메인 문구 없음. "목표, 전제, 제약 조건" — 범용 개념. | **도메인 중립** |
| `classification/prompts.py` | 45–185 | `GROUNDING_PROCESS_PROMPT`: 예시가 "P 방식", "Q 효과" 등 추상 기호. | **도메인 중립** |
| `classification/prompts.py` | 191–210 | `PREPROCESS_THINKING_PROMPT`: 도메인 문구 없음. | **도메인 중립** |
| `classification/prompts.py` | 216–297 | `REFLECTION_PROCESS_PROMPT`: 예시가 추상 기호. | **도메인 중립** |
| `classification/module.py` | 79–93 | `step1_1` mock: `"사용자가 여행 조건을 제시했다"`, `"AI가 일정 구조와 우선순위 조정을 제안했다"` | 하드코딩(mock) |
| `classification/module.py` | 151–162 | `step2_1` mock thinking 3줄: "일정은 오전/오후/저녁", "이동 시간", "까다로운 제약은 예산, 동선, 휴식" | 하드코딩(mock) |

> **판정**: 실제 프롬프트는 완전 중립. Mock 출력 5줄 교체 필요.

---

### ④ LG 하위범주(taxonomy) 정의

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| `app/graph.py` | 234 | `_ZONE_TYPES = ("구조적 결정", "암묵적 전제", "용어 해석")` | **도메인 중립** (인지 범주) |
| `classification/module.py` | 234–241 | `_classify_type()`: 접두어 기준 분류 | **도메인 중립** |
| `static/index.html` | 648–653 | zone 라벨 "구조적 결정", "암묵적 전제", "용어 해석" | **도메인 중립** |

> **판정**: 수정 불필요. 3개 범주는 도메인 독립 인지 범주.

---

### ⑤ 주입(injection) 관련 코드/설정

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| — | — | **존재하지 않음** | — |

> **판정**: 전무. Phase 3에서 신규 개발 필요 (High effort).

---

### ⑥ UI 문자열

| 파일 | 라인 | 결합 내용 | 강도 |
|------|------|-----------|------|
| `static/index.html` | 788 | `<h1>여행 계획 협업</h1>` | 하드코딩 |
| `static/index.html` | 789 | "까다로운 제약이 있는 여행 계획을 챗봇과 함께 세웁니다. 어떤 여행을 원하시나요?" | 하드코딩 |
| `static/index.html` | 790 | `<label>여행 의도</label>` | 하드코딩 |
| `static/index.html` | 974 | `toast('여행 의도를 입력하세요.')` | 하드코딩 |
| `static/index.html` | 984 | 초기 AI 메시지: "원하시는 여행 조건을 알려주시면 일정과 제약을 함께 맞춰보겠습니다." | 하드코딩 |
| `static/index.html` | 778 | "연구자 설정 화면입니다. 참가자 ID를 입력한 뒤…" | **도메인 중립** |
| `static/index.html` | 791 | `<textarea id="intent-text">` placeholder 없음 | **도메인 중립** |

> **판정**: 여행 문구 5건 — 문구 교체(low effort). 단, Phase 3 §2.2 스테이지 게이트 구조 추가 시 UI 대규모 신규 개발 필요(high effort).

---

### ⑦ 초기 의도문 입력 UI 및 저장·전달 경로

| 파일 | 라인 | 내용 |
|------|------|------|
| `static/index.html` | 787–793 | `#screen-intent`: textarea에 user_intent 입력 |
| `static/index.html` | 965–986 | `api('/api/session/start', {participant_id, user_intent})` |
| `app/main.py` | 81–116 | `POST /api/session/start`: `user_intent`를 sessions 테이블에 저장, `initial_state()` 호출 |
| `app/graph.py` | 42–59 | `initial_state()`: F_t 첫 항목으로 `"초기 사용자 의도: {user_intent}"` 추가. **파트너 LLM 시스템 프롬프트에는 포함하지 않음** (의도적 설계) |
| `app/config.py` | 19–51 | `assemble_partner_system_prompt()`: `user_intent` 파라미터 없음 — blind partner 설계 확인 |

> **판정**: 저장·전달 경로는 도메인 중립. UI 라벨 교체(low effort). 전달 비차단 구조는 §2.4 제약(프로필 LLM 비전달)과 동일 패턴이므로 재사용 가능.

---

### ⑧ 시나리오/조건 설정 파일

| 파일 | 라인 | 내용 | 강도 |
|------|------|------|------|
| `config.yaml` | 1–25 | `model`, `max_tokens`, `extended_thinking`, `neutral_system_prompt`, `task_context`, `shadow_probe_prompt`, `session_minutes` | 도메인 문구 2건(①과 동일) |
| `.env` | 1–2 | `SAM_API_KEY`, `MOCK_LLM` | **도메인 중립** |

> **판정**: 주입 스케줄, 스테이지 게이트 조건, 프로필 스키마 설정 필드 모두 부재. config.yaml 확장 필요.

---

### ⑨ 로깅 스키마

| 테이블 | 필드 | 도메인 종속 여부 |
|--------|------|----------------|
| `sessions` | `session_id, participant_id, user_intent, paper_id, current_turn, started_at, ended_at, config_snapshot` | `user_intent`만 입력값(중립), 나머지 중립 |
| `messages` | `role, content, turn, timestamp, system_prompt_used, extended_thinking` | **중립** |
| `turn_artifacts` | `C_t, D_t, B_t, A_t, system_prompt_used` | **중립** |
| `groundlens_states` | `state_json` (전체 상태 JSON) | **중립** |
| `corrections` | `target_item_id, correction_text, source_turn, corrected_at_turn` | **중립** |

> **판정**: 현재 스키마는 도메인 중립. 그러나 **injection 로깅 테이블 전무** — 신규 필요: `constraint_profiles`, `injections`, `injection_events` 테이블.

---

### ⑩ 세션 상태머신

| 파일 | 라인 | 내용 |
|------|------|------|
| `app/graph.py` | 215–227 | `run_message_graph()`: partner_llm → grounding → reflection 선형 파이프라인 |
| `app/main.py` | 128–208 | `POST /api/message`: 단일 상태로 처리, 단계 분기 없음 |
| `app/graph.py` | 42–59 | `initial_state()`: `current_turn` 카운터만 있음 |

> **판정**: 단계 개념 없음. `current_stage(S1/S2/S3)`, `stage_gate_met(bool)`, `revision_count`, `turn_in_stage` 필드 신규 추가 필요(medium effort).

---

### ⑪ 평가/채점 관련 코드

| 파일 | 내용 |
|------|------|
| — | **존재하지 않음** |

> **판정**: 전무. 채점 하네스(§2.3 JSON export + 제약 ID 매핑 + 위반 수 산출) 신규 개발 필요 (high effort).

---

### ⑫ 테스트 코드/픽스처

| 파일 | 내용 |
|------|------|
| — | **존재하지 않음** (`tests/` 디렉토리 없음) |

> **판정**: 테스트 없음. Phase 4 시뮬레이션 자동 주행은 별도 스크립트 신규 작성 필요.

---

### ⑬ 모델 설정

| 파일 | 라인 | 내용 | 강도 |
|------|------|------|------|
| `config.yaml` | 1 | `model: "claude-sonnet-4.6"` | 스냅샷 미고정 — 위험 |
| `config.yaml` | 2 | `max_tokens: 4096` | 적절 |
| `config.yaml` | 4–6 | `extended_thinking.enabled: false` | 적절 (SAM 엔드포인트 미지원) |
| `classification/module.py` | 42–54 | `call_llm()`: `temperature=0` 고정 | 적절 (분류 재현성) |
| `app/llm.py` | 56–64 | `get_assistant_response()`: temperature 파라미터 없음 → 엔드포인트 기본값 사용 | 미확인 — 기본값 명시 필요 |

> **판정**: 모델명이 SAM 엔드포인트 내부에서 가리키는 실제 스냅샷 버전이 불투명. `config_snapshot`으로 session 생성 시 기록은 되지만, 실험 기간 중 SAM 엔드포인트가 `claude-sonnet-4.6`을 어느 스냅샷으로 라우팅하는지 별도 확인 필요.

---

## B. 분류

### B-1. 도메인 중립 (수정 불필요)

- `classification/prompts.py` 전체 4개 프롬프트 (PREPROCESS_UTTERANCE, GROUNDING_PROCESS, PREPROCESS_THINKING, REFLECTION_PROCESS)
- LG 3-zone taxonomy (`구조적 결정 / 암묡적 전제 / 용어 해석`)
- DB 스키마 기존 5개 테이블 (필드 수준에서 중립)
- `app/graph.py` 파이프라인 구조 자체
- user_intent 저장·전달 경로 (blind partner 패턴)
- WebSocket push 구조 (`app/ws.py`)

### B-2. 문구만 교체하면 되는 지점 (low effort)

| 위치 | 교체 대상 | 새 문구 (TASK_SPEC.md §2 기반) |
|------|-----------|-------------------------------|
| `config.yaml` L9 | `neutral_system_prompt` 역할 정의 | 방학 계획 파트너 AI (§3.1 T1) |
| `config.yaml` L14 | `task_context` 과업 설명 | "AI와 함께 다음 방학 계획 세우기" (§3.1 T1) |
| `static/index.html` L788 | `<h1>여행 계획 협업</h1>` | `<h1>방학 계획 세우기</h1>` |
| `static/index.html` L789 | 안내 문구 | 방학 과업 안내 |
| `static/index.html` L790 | label "여행 의도" | "방학 목표 및 제약" |
| `static/index.html` L974 | toast 메시지 | "목표와 제약을 입력하세요." |
| `static/index.html` L984 | 초기 AI 메시지 | 방학 계획 시작 멘트 |
| `app/llm.py` L44–50 | mock assistant/thinking 텍스트 | 방학 도메인 mock |
| `app/llm.py` L73–77 | mock probe JSON | 방학 도메인 mock |
| `classification/module.py` L86–87 | step1_1 mock elements | 방학 도메인 mock |
| `classification/module.py` L159–161 | step2_1 mock thinking 3줄 | 방학 도메인 mock |

### B-3. 로직 변경 필요 지점 (medium effort)

| 항목 | 내용 |
|------|------|
| `GroundLensState` 확장 | `current_stage`, `turn_in_stage`, `revision_count`, `stage_gate_met` 필드 추가 |
| `run_message_graph()` | 단계별 분기 로직 — S1 게이트(3턴), S2 게이트(2회 수정) 조건 추가 |
| `POST /api/message` | 단계 상태 반환, 게이트 활성화 여부 포함 |
| DB `sessions` 테이블 | `current_stage TEXT`, `revision_count INTEGER` 필드 추가 |
| `config.yaml` | 파트너 LLM 시스템 프롬프트에 5섹션 템플릿 + 지식 컷오프 가드 + "합리적 초안 먼저 제시" 추가 |
| `[확인 필요]` 마커 | 전제 추출 파이프라인에서 마커 포함 항목을 노드 생성 제외 처리 |

### B-4. 신규 개발 필요 지점 (high effort)

| 항목 | 설명 |
|------|------|
| 사전 설문 폼 | 제약 프로필 10필드 웹 폼 + `constraint_profiles` DB 저장 (파트너 LLM 비전달 경로 확인) |
| 주입 생성기 | 프로필 → 8유형 템플릿 슬롯 채움 → 조건부 시스템 프롬프트 삽입 + 라틴 방격 배정 |
| 발현 확인기 | 주입 전제 발현 자동 판정(규칙 기반 + LLM judge) + 재도입 감지 로깅 |
| 채점 하네스 | 계획 JSON export + 제약 ID ↔ 계획 필드 매핑 + 위반 수 자동 산출 |
| 3단계 게이트 UI | S1/S2/S3 전환 버튼, 게이트 활성화 표시, 진행 상태 UI |
| injection 로깅 DB | `constraint_profiles`, `injections`, `injection_events` 신규 테이블 |
| Phase 4 시뮬레이션 | 자동 주행 스크립트 |

---

## C. 신규 개발 목록 초안 (Phase 3 작업 순서)

Phase 3 T1–T10에 대응:

| T# | 분류 | 예상 공수 | 의존성 |
|----|------|-----------|--------|
| T1 | B-2 + B-3 | 1–2h | 없음 |
| T2 | B-4 (사전 설문) | 4–6h | T1 완료 후 |
| T3 | B-4 (주입 생성기) | 6–8h | T2 (프로필 스키마 확정 후) |
| T4 | B-4 (발현 확인기) | 4–6h | T3 |
| T5 | B-3 (상태머신 3단계) | 4–6h | T1 |
| T6 | B-4 (채점 하네스) | 4–6h | T2, T5 |
| T7 | B-2 (mock 교체) | 1h | 없음 |
| T8 | B-2 + B-3 (`[확인 필요]` 마커) | 2h | T1 |
| T9 | 확인만 필요 (기존 shadow 로그) | 1h | 없음 |
| T10 | B-2 (mock 픽스처) | 1h | T7과 동일 |

---

## D. 리스크/불명확 사항 (연구자 결정 필요)

| ID | 항목 | 내용 |
|----|------|------|
| D-1 | SAM 엔드포인트 모델 스냅샷 | `claude-sonnet-4.6`이 SAM 내부에서 어느 실제 Claude 스냅샷으로 라우팅되는지 확인 필요. 실험 기간 중 라우팅이 바뀌면 §3.2-4(버전 pinning) 위반. SAM 쪽에 스냅샷 ID 확인 요청 권장. |
| D-2 | `[확인 필요]` 마커의 GroundLens 처리 | §2.7: "마커 항목은 노드 생성 제외 또는 별도 상태". 제외가 맞는지, 별도 회색 노드로 표시하는 게 맞는지 연구자 결정 필요. |
| D-3 | 사전 설문과 consent | 제약 프로필이 개인정보(체력, 알바 시간, 예산) 포함 — IRB 동의서와 데이터 보관 정책 정합 여부 확인 필요. |
| D-4 | 라틴 방격 배정 단위 | 세션당 4건 주입을 8유형 중에서 배정 시 — 참가자 수, 세션 수에 따라 방격 설계가 달라짐. GLMM 분석 의도를 고려한 배정 방식 확정 필요. |
| D-5 | 파트너 LLM temperature | `get_assistant_response()`에 temperature 미지정 → SAM 엔드포인트 기본값 사용. 방학 계획 초안 생성에서 창의성 vs. 재현성 trade-off. 연구자가 값 지정 필요. |
| D-6 | 발현 확인기 LLM judge 모델 | §2.6: "LLM judge 이중 체크". 파트너 LLM과 동일 모델 사용 여부, 별도 judge 프롬프트 설계 필요. |
| D-7 | baseline 조건 구현 방식 | 현재 코드에 baseline/GL 조건 분기 없음. 패널 숨김 = CSS `display:none` 수준인지, 아예 별도 세션 타입인지 확정 필요. |
