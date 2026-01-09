# FinAgent Investment Agent

<div align="center">
  <a href="README.md">🇺🇸 English</a> | <a href="README_KR.md">🇰🇷 한국어</a>
</div>

<br>

**FinAgent**는 개인화된 투자 조언을 제공하기 위해 설계된 정교한 **멀티 에이전트 AI 시스템**입니다. 사용자 프로필 분석부터 상반된 관점을 가진 전문가 에이전트 간의 치열한 토론에 이르기까지, 전문화된 에이전트들이 협업하여 사용자의 리스크 성향과 목표에 딱 맞는 데이터 기반의 금융 보고서를 제공합니다.

## 시스템 아키텍처 (System Architecture)

이 시스템은 **LangGraph**를 기반으로 한 그래프 워크플로우를 따르며, 사용자 상호작용 전반에 걸쳐 모듈성과 문맥 유지(Context Retention)를 보장합니다.

![FinAgent Architecture](assets/architecture.png)

### 워크플로우 개요 (Workflow Overview)

1.  **입력 가드레일 (Input Guardrail)**: 사용자의 모든 입력은 안전성과 금융 도메인 관련성을 먼저 검사받습니다. 금융과 무관한 질의는 즉시 필터링됩니다.
2.  **맥락 인식 라우팅 (Context-Aware Routing)**: **`Condition Node`**가 사용자의 의도와 대화 이력을 분석하여 다음 단계를 결정합니다.
    * *신규 사용자 / 프로필 업데이트* → **`User Chat`** (온보딩/인터뷰)
    * *단순 데이터 조회* → **`Retriever`** (검색)
    * *복합 투자 조언* → **`Debate`** (심층 분석 및 토론)
    * *보고서 생성* → **`Finance`** (종합 및 리포트 작성)
3.  **멀티 에이전트 협업 (Multi-Agent Collaboration)**:
    * **User Chat**: 자연스러운 인터뷰를 통해 부족한 KYC 데이터(위험 성향, 소득 등)를 수집하고 Supabase와 동기화합니다.
    * **Debate**: **상승론자(Bull), 하락론자(Bear), 중립론자(Balanced)** 전문가들이 실시간 데이터(Yahoo Finance, Tavily)를 사용하여 리스크와 기회를 분석하는 다자간 토론을 진행합니다.
    * **Retriever**: SQL을 통해 특정 금융 상품 데이터를 조회하거나 RAG를 통해 일반적인 금융 지식을 검색합니다.
4.  **최종 산출물 (Final Output)**: **`Finance Node`**가 수집된 모든 인사이트와 토론 결론을 종합하여 구조화된 마크다운 보고서를 생성합니다.

### 주요 모듈:

1.  **Guardrail**: 안전성 및 도메인 관련성을 보장하며 비금융 질의를 차단합니다.
2.  **Condition**: 사용자 의도와 문맥을 분석하여 흐름을 제어하는 지능형 라우터입니다 (예: 토론, 검색, 프로필 설정 등으로 분기).
3.  **User Chat**: 자연스러운 인터뷰를 통해 사용자 프로필(KYC)을 수집 및 관리하고 데이터베이스와 동기화합니다.
4.  **Retriever**: SQL(상품 조회), RAG(규정 검색), API(시세 조회) 등을 통해 데이터를 가져옵니다.
5.  **Debate**: 공격적(Bull), 보수적(Bear), 중립적(Balanced) 전문가 간의 치열한 논쟁을 시뮬레이션하고 CIO(판사)가 결론을 내립니다.

## 프로젝트 구조 (Project Structure)
```
FinAgent_Investment_Agent/
├── assets/                  # Static assets (images, diagrams)
├── condtition/              # Router & Safety Layer
│   ├── condition.py         # Context-aware routing logic
│   └── guardrail.py         # Input validation & safety checks
├── debate/                  # Debate Engine
│   ├── node.py              # 5-Round Debate Logic (Bull vs Bear vs Judge)
│   └── tools.py             # Tools for debate (News, Market Data)
├── finance/                 # Reporting Engine
│   ├── node.py              # Final Report Generation
│   └── tools.py             # SQL-based Product Recommendation Tools
├── retriever/               # Information Retrieval
│   ├── node.py              # ReAct Agent for Search
│   └── tools.py             # Hybrid Search (SQL + Vector + Web)
├── user_chat/               # User Onboarding
│   ├── models.py            # Pydantic Data Models for Profile
│   └── node.py              # Interview & DB Sync Logic
├── utils/                   # Core Utilities
│   ├── const.py             # Constants & Prompts
│   ├── db.py                # Supabase Connection
│   ├── embedding.py         # BGE-M3 Embedding Loader
│   ├── llm.py               # OpenRouter/HTTPX Client
│   └── state.py             # Shared Agent State (Memory)
├── .env                     # API Keys and Config
├── .gitignore
├── api.py                   # FastAPI REST API Server
├── main.py                  # Application Entry Point (Graph Compiler)
└── README.md                # Project Documentation
```
## 1. 설치 방법 (Installation)

저장소를 클론하고 필요한 의존성 패키지를 설치합니다.

```bash
# 저장소 클론
git clone [https://github.com/your-repo/FinAgent_Investment_Agent.git](https://github.com/FinAgent-Lab/FinAgent_Investment_Agent.git)
cd FinAgent

# 의존성 설치
pip install -r requirements.txt
```

## 2. 환경 설정 (Configuration)

루트 디렉토리에 ```.env``` 파일을 생성하고 API 키를 추가합니다.

```
OPENROUTER_API_KEY="your-api-key"

SUPABASE_URL="your-supbase-url"
SUPABASE_SERVICE_KEY="your-api-key"

TAVILY_API_KEY="your-api-key"
```

## 3. 사용 방법 (Usage)

### 옵션 A: 대화형 CLI 모드
메인 스크립트를 실행하여 대화형 CLI 세션을 시작합니다.

```bash
python main.py
```

### 옵션 B: REST API 서버
프로그래밍 방식과 접근을 위해 FastAPI 서버를 시작합니다:
```bash
python api.py
```
서버는 ```http://localhost:8000```에서 시작됩니다.

### API 엔드포인트
**POST /chat** - 메시지 전송 및 AI 응답 수신
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Should I invest in Nvidia?",
    "session_id": "optional-session-id"
  }'
```
응답 예시:
```json
{
  "session_id": "abc123",
  "user_id": "test_user",
  "message": "Should I invest in Nvidia?",
  "response": "Let me analyze Nvidia for you...",
  "node_executed": "debate",
  "debate_history": [...],
  "timestamp": "2024-01-03T12:00:00"
}
```

**GET /profile/{user_id}** - 사용자 프로필 조회
```bash
curl http://localhost:8000/profile/test_user
```

**POST /profile/{user_id}** - 사용자 프로필 생성/업데이트
```bash
curl -X POST http://localhost:8000/profile/test_user \
  -H "Content-Type: application/json" \
  -d '{
    "name_display": "John Doe",
    "age_range": "30대",
    "risk_tolerance_level": "aggressive",
    "goal_type": "long_term"
  }'
```

**GET /health** - 서버 상태 확인
```bash
curl http://localhost:8000/health
```

## 4. 사용 예시 (Example)
시나리오: 사용자가 엔비디아(Nvidia) 투자에 대해 질문하는 상황
```text
User: 지금 엔비디아 매수해도 될까요? (Should I buy Nvidia right now?)

Guardrail: [finance] -> Allowed: True
Routing to [debate] (Reason: User is asking for investment advice.)

[Debate Arena] Topic: Should I buy Nvidia right now? (5 Rounds)

--- Round 1: Opening Statement ---
Turn: Conservative Expert (보수적 전문가)
Arg: 엔비디아의 밸류에이션은 너무 높습니다 (PER > 70). 중국 관련 규제 리스크도 우려됩니다...

Turn: Aggressive Expert (공격적 전문가)
Arg: AI 수요는 이제 시작일 뿐입니다. 엔비디아는 H100 칩에 대한 독점적 지위를 가지고 있습니다...

... (Rounds 2, 3, 4...) ...

Judge's Verdict (CIO 판결):
   "성장 잠재력은 부인할 수 없으나, 단기적 변동성이 예상됩니다. 
    분할 매수 전략을 추천합니다."

AI Suggestion: "보고서를 작성하기 전에 환율 리스크도 분석해 볼까요?"

User: 네, 그것도 확인해주세요.

... (환율에 대한 추가 토론 진행) ...

[Finance Node] Generating Final Report...
Report Saved to DB.

Agent (finance):
# 📋 투자 자문 보고서: 엔비디아 분석
## 1. 시장 분석...
## 2. 투자 전략...
## 3. 추천 포트폴리오 (상품: TIGER 미국테크TOP10)...
```

## 핵심 구성 요소 (Key Components)
하이브리드 시장 데이터 도구 (```utils/tools.py```) 우리 에이전트는 실시간 주식 데이터를 가져오기 위해 하이브리드 접근 방식을 사용합니다:
- 한국 주식: FinanceDataReader 사용 (네이버 금융 기반).
- 글로벌 주식/지수: Yahoo Finance 사용 (자동 티커 매핑 기능 포함).

맥락 인식 라우터 (```condition/condition.py```) 라우터는 단순히 마지막 메시지만 보지 않습니다. 전체 대화 기록을 분석하여 사용자가 프로필 질문에 대답하는 중인지, 제안에 동의하는 중인지, 아니면 새로운 질문을 하는 중인지를 판단하여 무한 루프를 방지합니다.

### 데이터베이스 동기화 (```user_chat/node.py```)
- Load: 시작 시 기존 사용자 프로필을 불러옵니다.
- Upsert: 대화 중 새로운 정보가 추출될 때마다 Supabase에 프로필을 자동으로 업데이트합니다.
- Strict Schema: 데이터 무결성을 위해 엄격한 데이터 타입(예: risk_tolerance_level Enum)을 강제합니다.

## 산출물 (Outputs)
- 대화 기록 (Conversation History): 문맥 유지를 위해 MemorySaver를 사용하여 인메모리에 저장됩니다.
- 사용자 프로필 (User Profile): Supabase의 user_profile 테이블에 영구 저장됩니다.
- 자문 보고서 (Advisory Report): 최종 보고서는 advisory_reports 테이블에 저장되며 마크다운 형식으로 사용자에게 제공됩니다.