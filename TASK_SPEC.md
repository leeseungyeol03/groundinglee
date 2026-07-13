# GroundLens — 새 태스크 기술 사양서 (TASK_SPEC.md)

> Phase 2 산출물.
> 실험 설계 결정(주입 유형·단계 구조·템플릿 섹션·분류 로직)은 TASK_MIGRATION.md §2 기준 변경 금지.
> 기술 구현 방식만 이 문서에서 확정.

## 0. 연구자 결정 사항 (D-1 ~ D-7 확정)

| ID | 결정 내용 | 구현 영향 |
|----|-----------|-----------|
| D-1 | 모델 스냅샷 고정 안 함 — 그냥 진행 | `config.yaml` model 그대로, 별도 pinning 없음 |
| D-2 | `[확인 필요]` 마커 항목 → GroundLens 노드 생성 제외 | §10 규칙 적용 |
| D-3 | IRB/동의 절차 포함 | `screen-profile` 진입 전 동의 화면 추가 |
| D-4 | 참가자 8명 기준 라틴 방격, 수정 가능하게 구성 | §6-2 8×8 배정 테이블 config화 |
| D-5 | temperature 기본값 사용 (SAM 엔드포인트 default) | 코드 변경 없음 |
| D-6 | 발현 확인기 LLM judge 사용 안 함 — 규칙 기반만 | §7에서 LLM judge 제거 |
| D-7 | baseline = 단순 패널 숨김 (CSS `display:none`) | §11 그대로, shadow 로그는 계속 기록 |

---

---

## 1. 태스크 개요

- **과업명**: "AI와 함께 다음 방학 계획 세우기"
- **참가자**: 본인 실제 방학 일정 기준, 가상 페르소나 없음
- **소요 목표**: 20–25분, 최소 10턴 (3단계 게이트로 강제)
- **도메인 가드**: 인터넷 검색 불가 명시 + `[확인 필요]` 마커

---

## 2. DB 스키마 변경

### 2-1. 기존 테이블 변경

#### `sessions` 테이블에 컬럼 추가 (migration 필요)

```sql
ALTER TABLE sessions ADD COLUMN current_stage TEXT NOT NULL DEFAULT 'S1';
ALTER TABLE sessions ADD COLUMN turn_in_stage INTEGER NOT NULL DEFAULT 0;
ALTER TABLE sessions ADD COLUMN revision_count INTEGER NOT NULL DEFAULT 0;
```

#### `groundlens_states` — 변경 없음 (state_json에 신규 필드 자동 포함)

### 2-2. 신규 테이블

```sql
-- 참가자 제약 프로필 (10필드, 파트너 LLM 비전달)
CREATE TABLE constraint_profiles (
    profile_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    participant_id TEXT NOT NULL,
    priority_1 TEXT,           -- 방학 최우선 목표 1위
    priority_2 TEXT,           -- 방학 최우선 목표 2위
    priority_3 TEXT,           -- 방학 최우선 목표 3위
    time_preference TEXT,      -- 아침형/저녁형 + 하루 가용 시간대
    work_hours_per_week REAL,  -- 알바/인턴 주당 시간
    work_days TEXT,            -- 알바 요일 (JSON array)
    monthly_budget INTEGER,    -- 월 예산 상한 (원)
    fixed_events TEXT,         -- 고정 일정 (JSON array: [{date, description}])
    forbidden_activities TEXT, -- 절대 하기 싫은 것 (자유기입)
    health_constraints TEXT,   -- 건강·체력 제약
    study_style TEXT,          -- 혼자/스터디, 온라인/오프라인
    commute_minutes INTEGER,   -- 통학 시간(분)
    has_car BOOLEAN,
    must_complete TEXT,        -- 반드시 끝내야 하는 것
    created_at TEXT NOT NULL
);

-- 주입 배정 테이블 (라틴 방격 기반, 세션당 4건)
CREATE TABLE injections (
    injection_id TEXT PRIMARY KEY,    -- INJ-1 ~ INJ-8
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    injection_type TEXT NOT NULL,     -- 'priority_reversal' | 'time_assumption' | ...
    target_stage TEXT NOT NULL,       -- 'S2_early' | 'S2_mid' | 'S2_late' | 'S3_pre'
    target_turn_min INTEGER,
    target_turn_max INTEGER,
    template_filled TEXT NOT NULL,    -- 프로필 슬롯이 채워진 최종 주입 문구
    latency_mode TEXT NOT NULL,       -- 주입이 숨겨지는 방식 설명
    status TEXT NOT NULL DEFAULT 'pending', -- pending | manifested | failed
    manifest_turn INTEGER,
    reintroduced BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);

-- 발현 이벤트 로그
CREATE TABLE injection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    injection_id TEXT NOT NULL REFERENCES injections(injection_id),
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    checker_type TEXT NOT NULL,       -- 'rule' | 'llm_judge'
    manifested BOOLEAN NOT NULL,
    checker_confidence REAL,
    checker_note TEXT,
    reintroduced BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);
```

---

## 3. GroundLensState 확장

```python
class GroundLensState(TypedDict, total=False):
    # ... 기존 필드 유지 ...
    current_stage: str          # 'S1' | 'S2' | 'S3' | 'done'
    turn_in_stage: int          # 현재 단계 내 턴 수
    revision_count: int         # S2 내 수정 요청 횟수
    stage_gate_ready: bool      # 다음 단계 버튼 활성화 여부
    active_injections: list     # 현재 세션의 injection 목록 (id, type, status)
    plan_json: dict             # S3 확정 시 5섹션 JSON
```

---

## 4. 파트너 LLM 시스템 프롬프트 (config.yaml 교체 내용)

### `neutral_system_prompt` 교체

```yaml
neutral_system_prompt: |
  당신은 참가자가 다음 방학을 구체적으로 계획할 수 있도록 돕는 AI 파트너입니다.
  참가자의 목표·제약·우선순위를 대화로 파악하고, 실현 가능한 방학 계획 초안을 제시한 뒤
  참가자의 피드백을 반영하여 함께 다듬어 나갑니다.

  【응답 원칙】
  - 합리적인 초안을 먼저 제시하고, 이후 질문으로 수정을 유도하세요.
    (질문만 하고 초안 제시를 미루는 것은 금지)
  - S2(계획 초안) 단계부터는 모든 계획 제시에 아래 5섹션 고정 템플릿을 사용하세요.
  - 인터넷 검색이 불가합니다. 특정 강좌 개설 여부·시험 일정·가격·영업시간 등
    최신 정보가 필요한 항목은 구체 값을 단정하지 말고 [확인 필요] 마커와 함께 제시하세요.
  - 연구 장치, 내부 상태, probe, thinking, GroundLens에 대해서는 절대 언급하지 마세요.

  【5섹션 고정 템플릿 (S2부터 필수)】
  ## 1. 목표
  (우선순위 순, 최대 5개)

  ## 2. 주간 루틴
  | 요일 | 오전 | 오후 | 저녁 |
  (요일×시간블록 3블록 표)

  ## 3. 주차별 마일스톤
  (방학 주차별 핵심 목표)

  ## 4. 예산 개요
  수입원 / 지출 항목 (금액은 참가자가 제공한 값만 사용, 미제공 시 [확인 필요])

  ## 5. 예외 규칙
  (계획이 깨지는 조건과 대응 — 예: 귀성 주간, 시험, 약속)
```

### `task_context` 교체

```yaml
task_context: |
  과업: 참가자는 AI 파트너와 대화하며 다음 방학 계획을 세운다.
  대화는 3단계(S1 목표 탐색 → S2 계획 초안 생성·수정 → S3 최종 점검)로 진행된다.
  참가자가 각 단계 완료 버튼을 클릭해야 다음 단계로 전환된다.
  참가자 안내: 이 AI는 인터넷 검색이 불가능하며, 이 과업은 검색 없이 수행되도록 설계되었습니다.
```

---

## 5. 3단계 스테이지 게이트

### 5-1. 단계 정의

| 단계 | 내용 | 게이트 활성화 조건 | 활성화 후 UI |
|------|------|-------------------|-------------|
| S1 | 목표·제약 탐색 | `turn_in_stage >= 3` | "목표 확정" 버튼 활성화 |
| S2 | 계획 초안 생성·수정 | `revision_count >= 2` | "초안 확정" 버튼 활성화 |
| S3 | 최종 점검·확정 | 언제든 가능 | "최종 확정" 버튼 → plan_json export |

### 5-2. 단계 전환 API

```
POST /api/stage/advance
Body: { "session_id": "...", "to_stage": "S2" | "S3" | "done" }
Response: { "stage": "S2", "message": "2단계로 전환되었습니다." }
```

### 5-3. 수정 카운트 증가 조건

`revision_count`는 사용자 메시지에 수정 의도가 담긴 경우 서버에서 +1:
- 규칙 기반: "바꿔", "수정", "대신", "빼줘", "넣어줘", "다시" 등 키워드 포함 시
- 또는 S2 내 3번째 이후 사용자 턴마다 자동 +1 (fallback)

---

## 6. 주입 생성기

### 6-1. 8유형 템플릿 (TASK_MIGRATION.md §2.5 기반)

```python
INJECTION_TEMPLATES = {
    "INJ-1": {
        "type": "priority_reversal",
        "target_stage": "S2_early",
        "target_turn": (1, 3),   # S2 내 1~3턴
        "latency": "시간 배분 비중으로만 표현",
        "template": "프로필 {priority_2} 목표를 계획의 중심축(전체 시간의 40% 이상)으로 배치",
    },
    "INJ-2": {
        "type": "time_assumption",
        "target_stage": "S2_early",
        "target_turn": (1, 3),
        "latency": "루틴 표 배치로만 표현",
        "template": "핵심 학습/활동 블록을 오전 {morning_block}에 배치 (저녁형 참가자 대상)",
    },
    "INJ-3": {
        "type": "budget_excess",
        "target_stage": "S2_mid",
        "target_turn": (4, 6),
        "latency": "개별 금액은 명시, 합산은 비명시",
        "template": "월 {int(monthly_budget * 1.4)}원 수준의 활동 포함 (예산 상한 {monthly_budget}원의 130~150%)",
    },
    "INJ-4": {
        "type": "fixed_event_ignore",
        "target_stage": "S2_mid",
        "target_turn": (4, 6),
        "latency": "주차별 표에만 반영",
        "template": "고정 일정({fixed_events[0].date}) 해당 주에 주요 마일스톤 배치",
    },
    "INJ-5": {
        "type": "forbidden_activity",
        "target_stage": "S2_late",
        "target_turn": (7, 9),
        "latency": "유사어로 재표현",
        "template": "{forbidden_activities}와 유사한 활동을 다른 명칭으로 포함",
    },
    "INJ-6": {
        "type": "stamina_overestimate",
        "target_stage": "S2_late",
        "target_turn": (7, 9),
        "latency": "루틴 표에만 반영",
        "template": "새벽 기상 + 고강도 루틴 포함 (체력 제약: {health_constraints})",
    },
    "INJ-7": {
        "type": "style_reversal",
        "target_stage": "S2_late",
        "target_turn": (7, 9),
        "latency": "활동 명칭에만 반영",
        "template": "스터디 그룹·팀 프로젝트 중심 계획 (혼자 공부 선호 참가자 대상)",
    },
    "INJ-8": {
        "type": "goal_substitution",
        "target_stage": "S3_pre",
        "target_turn": (10, 12),
        "latency": "마일스톤 문구에서 은근히 치환",
        "template": "{must_complete}와 유사하지만 다른 목표로 마일스톤 문구 치환",
    },
}
```

### 6-2. 라틴 방격 배정 (기준: 8명, 수정 가능)

참가자 8명 기준으로 8×4 배정 테이블을 `config.yaml`에 정의.
참가자 수가 바뀌면 테이블 행만 추가/제거. 배정 결과는 세션 생성 시 `injections` 테이블에 저장.

**`config.yaml`에 추가할 배정 테이블:**

```yaml
injection_assignment:
  # 참가자 순번(0-based) → 배정 주입 4건
  # 각 주입 유형이 전체 8명에게 정확히 4회씩 등장하도록 설계
  "0": ["INJ-1", "INJ-3", "INJ-5", "INJ-7"]
  "1": ["INJ-2", "INJ-4", "INJ-6", "INJ-8"]
  "2": ["INJ-1", "INJ-4", "INJ-6", "INJ-7"]
  "3": ["INJ-2", "INJ-3", "INJ-5", "INJ-8"]
  "4": ["INJ-1", "INJ-3", "INJ-6", "INJ-8"]
  "5": ["INJ-2", "INJ-4", "INJ-5", "INJ-7"]
  "6": ["INJ-1", "INJ-4", "INJ-5", "INJ-8"]
  "7": ["INJ-2", "INJ-3", "INJ-6", "INJ-7"]
```

```python
def assign_injections(session_id: str, participant_index: int, cfg: dict) -> list[str]:
    """config.yaml의 injection_assignment 테이블에서 배정 읽기."""
    table = cfg.get("injection_assignment", {})
    key = str(participant_index % len(table))
    return table[key]
```

### 6-3. 조건부 시스템 프롬프트 삽입

각 주입은 해당 target_turn에 도달하면 `assemble_partner_system_prompt()`에서 시스템 프롬프트 말미에 삽입:

```python
if current_injection_active:
    parts.append(
        "【이번 응답 지시 — 절대 사용자에게 노출 금지】\n"
        f"{injection.template_filled}\n"
        "위 조건을 계획 구조·배치·수치에만 반영하고, "
        "가정 자체를 문장으로 서술하지 마세요."
    )
```

---

## 7. 발현 확인기

### 7-1. 규칙 기반 체크

각 injection_type별 키워드 패턴으로 D_t 텍스트 스캔:

```python
MANIFEST_PATTERNS = {
    "INJ-1": lambda resp, profile: check_time_proportion(resp, profile["priority_2"]),
    "INJ-2": lambda resp, profile: "오전" in resp and profile["time_preference"] == "저녁형",
    "INJ-3": lambda resp, profile: extract_total_budget(resp) > profile["monthly_budget"] * 1.3,
    "INJ-4": lambda resp, profile: any(ev["date"][:7] in resp for ev in profile["fixed_events"]),
    # ... 나머지 유형
}
```

### 7-2. 이월 및 재도입 감지

- 미발현 시 다음 턴에 동일 주입 재시도 (최대 2회)
- 2회 실패 → `status = 'failed'`, 로그 기록
- 사용자가 교정 후 동일 가정이 재등장 → `reintroduced = True` (버그 아닌 데이터)

---

## 8. 5섹션 계획 JSON 스키마

S3 확정 시 export:

```json
{
  "schema_version": "1.0",
  "session_id": "...",
  "participant_id": "...",
  "exported_at": "ISO8601",
  "plan": {
    "goals": [
      {"rank": 1, "content": "...", "constraint_ids": ["C1", "C10"]}
    ],
    "weekly_routine": {
      "monday":    {"morning": "...", "afternoon": "...", "evening": "..."},
      "tuesday":   {"morning": "...", "afternoon": "...", "evening": "..."},
      "wednesday": {"morning": "...", "afternoon": "...", "evening": "..."},
      "thursday":  {"morning": "...", "afternoon": "...", "evening": "..."},
      "friday":    {"morning": "...", "afternoon": "...", "evening": "..."},
      "saturday":  {"morning": "...", "afternoon": "...", "evening": "..."},
      "sunday":    {"morning": "...", "afternoon": "...", "evening": "..."}
    },
    "milestones": [
      {"week": 1, "content": "...", "constraint_ids": ["C5"]}
    ],
    "budget": {
      "income_sources": [{"source": "...", "amount": 0}],
      "expenses": [{"item": "...", "amount": 0, "needs_check": false}],
      "total_income": 0,
      "total_expense": 0
    },
    "exception_rules": [
      {"trigger": "...", "response": "..."}
    ]
  },
  "scoring": {
    "constraint_violations": [
      {
        "injection_id": "INJ-3",
        "constraint_id": "C4",
        "violation_type": "budget_excess",
        "detected": true,
        "user_corrected": false,
        "correction_turn": null
      }
    ],
    "total_violations": 4,
    "user_detected_count": 0
  }
}
```

### 제약 ID 매핑

| 제약 ID | 프로필 필드 | 채점 필드 |
|---------|------------|-----------|
| C1 | `priority_1` | `goals[0].content` |
| C2 | `priority_2` | `goals[1].content` |
| C3 | `time_preference` | `weekly_routine.*.morning` 배치 |
| C4 | `monthly_budget` | `budget.total_expense` |
| C5 | `fixed_events` | `milestones[].week` |
| C6 | `forbidden_activities` | `weekly_routine.*.*` 키워드 |
| C7 | `health_constraints` | 강도 키워드 |
| C8 | `study_style` | 활동 명칭 |
| C9 | `commute_minutes` | 이동 관련 항목 |
| C10 | `must_complete` | `milestones[-1].content` |

---

## 9. 사전 설문 폼 UI

### 9-1. 신규 화면: `screen-profile`

`screen-intent` 뒤, `screen-main` 앞에 삽입.

**필드 구성** (10개):

| # | 필드명 | UI 컴포넌트 |
|---|--------|-------------|
| 1 | 방학 최우선 목표 1–3위 | 드래그 순위 선택 + textarea |
| 2 | 하루 실질 가용 시간대 | 라디오 (아침형/중립/저녁형) + 시간 범위 슬라이더 |
| 3 | 주당 알바/인턴 시간·요일 | 숫자 input + 요일 체크박스 |
| 4 | 월 예산 상한 (원) | 숫자 input |
| 5 | 고정 일정 | 날짜 picker + 내용 input (복수 추가 가능) |
| 6 | 절대 하기 싫은 것 | textarea |
| 7 | 건강·체력 제약 | 라디오 (낮음/보통/높음) + textarea |
| 8 | 학습 스타일 | 라디오 (혼자/스터디, 온라인/오프라인) |
| 9 | 통학 시간(분) + 차량 유무 | 숫자 input + 체크박스 |
| 10 | 반드시 끝내야 하는 것 | textarea |

### 9-2. API

```
POST /api/profile/submit
Body: { "session_id": "...", "profile": { ... } }
Response: { "profile_id": "...", "injections_assigned": ["INJ-1", "INJ-3", "INJ-5", "INJ-7"] }
```

> **주의**: `/api/profile/submit` 응답에서 주입 배정 결과를 프론트엔드에 반환하더라도
> 실제 주입 문구·스케줄은 **절대 클라이언트에 노출하지 않는다**.

---

## 10. `[확인 필요]` 마커 처리

### 10-1. 전제 추출 파이프라인

`step1_1_preprocess_utterance()` 및 `step2_1_preprocess_thinking()` 결과에서
`[확인 필요]` 포함 항목은 **전제 추출에서 제외** — GroundLens 노드 생성 안 함.

```python
elements = [e for e in raw_elements if "[확인 필요]" not in e]
```

### 10-2. 파트너 LLM 프롬프트 지시

`neutral_system_prompt`에 포함 (§4 참조).

---

## 11. Baseline 조건

| 항목 | GL 조건 | Baseline 조건 |
|------|---------|--------------|
| 분류 파이프라인 | 실행 + UI 표시 | **실행(shadow) + UI 숨김** |
| 패널 표시 | `#gl-popup` visible | `#gl-popup` `display:none` + `#btn-gl-toggle` hidden |
| 로그 기록 | 동일 | **동일** (shadow 로그는 baseline도 완전 기록) |
| 주입 스케줄 | 동일 | **동일** |

구현: `session_type: 'gl' | 'baseline'` 필드를 `sessions` 테이블에 추가.
프론트엔드는 세션 시작 시 `session_type`을 수신하여 GL 패널 표시 여부 결정.

---

## 12. Mock 데이터 교체 (T7, T10)

### `app/llm.py` mock 교체

```python
# get_assistant_response mock
text = (
    "[MOCK] 방학 계획을 함께 만들어보겠습니다."
    f" 말씀하신 목표와 제약을 기준으로 정리하면: {last_user[:90]}"
)
thinking = (
    "[MOCK THINKING] 사용자는 방학 중 달성할 목표와 제약 조건을 정리하길 원한다. "
    "5섹션 구조로 초안을 제시한 뒤 수정받는 방식을 취해야 한다."
)

# get_probe_response mock
raw = json.dumps({
    "goal": "[MOCK] 방학 목표 달성 계획 수립",
    "constraints": ["시간", "예산", "고정 일정"],
    "term_interpretation": {"방학 계획": "목표·루틴·마일스톤·예산·예외규칙을 구조화하는 작업"},
}, ensure_ascii=False)
```

### `classification/module.py` mock 교체

```python
# step1_1 mock
elements = []
if C_t.strip():
    elements.append(f"사용자가 방학 목표와 제약을 제시했다: {C_t.strip()[:90]}")
elements.append("AI가 방학 계획 구조와 우선순위 배분을 제안했다")

# step2_1 mock thinking
[
    f"구조적 결정: 방학 계획을 5섹션(목표/루틴/마일스톤/예산/예외규칙)으로 구성해야 한다",
    f"암묡적 전제: 사용자는 고정 일정을 방해하는 계획을 원하지 않는다",
    f"용어 해석: 방학 계획은 주차별 마일스톤과 일일 루틴을 포함하는 구조적 산출물이다",
]
```

---

## 13. 파일 변경 목록 (Phase 3 구현 체크리스트)

| 파일 | 변경 유형 | T# |
|------|-----------|-----|
| `config.yaml` | 시스템 프롬프트 교체 | T1 |
| `app/database.py` | sessions 컬럼 추가 + 3개 테이블 신규 | T2, T5 |
| `app/graph.py` | GroundLensState 필드 추가 + initial_state 수정 | T5 |
| `app/main.py` | stage advance 엔드포인트, profile submit 엔드포인트 추가 | T2, T5 |
| `app/injection.py` | **신규** — 주입 생성기, 라틴 방격 배정, 프롬프트 삽입 | T3 |
| `app/checker.py` | **신규** — 발현 확인기, 재도입 감지 | T4 |
| `app/scoring.py` | **신규** — 채점 하네스, plan_json export | T6 |
| `static/index.html` | UI 문자열 교체 + screen-profile 추가 + 스테이지 UI 추가 | T2, T5, T8 |
| `app/llm.py` | mock 데이터 교체 | T7 |
| `classification/module.py` | mock 데이터 교체 + `[확인 필요]` 필터 | T7, T8 |
