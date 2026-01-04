import json
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from utils.db import get_supabase_client
from user_chat.models import ExtractedInfo

# IMPORTANT: all required fields to collect
REQUIRED_FIELDS = [
    "name_display", "age_range", "income_bracket",
    "invest_experience_yr", "financial_knowledge_level",
    "current_holdings_note", "preferred_asset_types",
    "risk_tolerance_level", "total_investable_amt",
    "goal_type", "goal_description", "preferred_style"
]

class UserProfileChatNode:
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7)
        self.supabase = get_supabase_client()

    async def run(self, state: AgentState) -> Command:
        user_id = state["user_id"]

        # 1. DB 및 현재 상태에서 프로필 로드
        db_profile = self._fetch_profile_from_db(user_id)
        current_profile = state.get("user_profile") or {}
        if db_profile:
            current_profile.update(db_profile)

        # 2. 누락된 필드 확인
        missing_fields = []
        for field in REQUIRED_FIELDS:
            val = current_profile.get(field)
            if val is None or val == "" or val == []:
                missing_fields.append(field)

        is_complete = len(missing_fields) == 0

        # 3. [수정됨] 프로필 완성 시 로직 (초기 질문 복원 포함)
        if is_complete:
            print(f"⏩ Profile Fully Completed.")

            # 저장해둔 초기 질문(original_query)이 있다면 복원
            original_query = state.get("original_query")

            if original_query:
                print(f"🔄 Restoring original query: {original_query}")
                # 안내 메시지와 원래 질문을 메시지 기록에 추가
                # ConditionNode가 이 질문을 보고 즉시 분석을 시작함
                notice_msg = AIMessage(content="모든 정보가 수집되었습니다. 질문하신 내용에 대해 바로 분석을 시작합니다!")
                restore_msg = HumanMessage(content=original_query)

                return Command(
                    update={
                        "user_profile": current_profile,
                        "messages": [notice_msg, restore_msg],
                        "original_query": None  # 사용했으므로 초기화
                    },
                    goto="condition" # 다시 라우터로 이동
                )
            else:
                # 초기 질문 없이 설문만 완료한 경우
                return Command(update={"user_profile": current_profile}, goto="condition")

        # 4. system prompt (DB constraint와 일치하도록 수정)
        system_prompt = f"""
        You are a friendly Investment Onboarding Assistant.
        User Profile Status: {json.dumps(current_profile, ensure_ascii=False)}
        Missing Info: {missing_fields}

        **Conversation Strategy:**
        You need to collect ALL missing fields, but **ask only 1-2 questions at a time**.
        Group related topics naturally:
        1. **Basics:** Name, Age, Job/Income
        2. **Experience:** Investment Years (`invest_experience_yr`), Knowledge Level (`financial_knowledge_level`)
        3. **Assets:** Investable Amount, Current Holdings (`current_holdings_note`)
        4. **Preferences:** Asset Types (`preferred_asset_types`), Risk Level
        5. **Goals:** Goal Type (`goal_type`), Description
        6. **Style:** AI Persona (`preferred_style`)

        **CRITICAL - DATA MAPPING RULES (MUST match DB constraints):**
        You MUST map the user's input to these EXACT values:

        1. `risk_tolerance_level`: (ONLY 3 values allowed)
           - 안전형/보수적/낮은위험 -> 'conservative'
           - 중립형/균형형/중간위험 -> 'moderate'
           - 공격적/적극형/높은위험 -> 'aggressive'

        2. `financial_knowledge_level`: ['beginner', 'intermediate', 'advanced']
        
        3. `goal_type`: (ONLY 5 values allowed)
           - 단기목표(1-2년) -> 'short_term'
           - 중기목표(3-5년) -> 'mid_term'
           - 장기목표/자산증식/부의축적 -> 'long_term'
           - 은퇴준비 -> 'retirement'
           - 미정/불확실 -> 'unknown'
        
        4. `preferred_style`: (ONLY 5 values allowed)
           ['직설', '안정적/안심', '공손/설명형', '차분한코치', '밝은친구형']

        Output JSON matching `ExtractedInfo`. `response_message` is required.
        """

        structured_llm = self.llm.with_structured_output(ExtractedInfo)
        # 최근 대화 6턴만 포함하여 컨텍스트 유지
        messages = [SystemMessage(content=system_prompt)] + state["messages"][-6:]

        # LLM 호출
        result = await structured_llm.ainvoke(messages)

        # 5. DB 저장
        extracted_data = result.model_dump(exclude={"response_message"}, exclude_none=True)
        if extracted_data:
            current_profile.update(extracted_data)
            self._save_profile_to_db(user_id, current_profile)
            print(f"Database synced: {extracted_data.keys()}")

        # 6. 응답 생성 (Fallback 로직 포함)
        ai_response = result.response_message
        if not ai_response:
            remaining = []
            for field in REQUIRED_FIELDS:
                val = current_profile.get(field)
                if val is None or val == "" or val == []:
                    remaining.append(field)

            if remaining:
                ai_response = await self._generate_question_dynamically(remaining[0], current_profile)
            else:
                ai_response = "정보 수집이 완료되었습니다! 이제 투자를 시작해볼까요?"

        return Command(
            update={"user_profile": current_profile, "messages": [AIMessage(content=ai_response)]},
            goto="__end__" # 사용자 응답 대기
        )

    async def _generate_question_dynamically(self, target_field: str, profile: dict) -> str:
        prompt = f"""
        User Profile: {json.dumps(profile, ensure_ascii=False)}
        Missing Field: '{target_field}'
        Generate a natural question to ask for this information.
        """
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return res.content

    def _fetch_profile_from_db(self, user_id: str) -> dict:
        if not self.supabase: return {}
        try:
            res = self.supabase.table("user_profile").select("*").eq("external_user_key", user_id).execute()
            if res.data: return res.data[0]
        except Exception as e:
            print(f"Database load error: {e}")
        return {}

    def _save_profile_to_db(self, user_id: str, data: dict):
        if not self.supabase: return
        try:
            data["external_user_key"] = user_id
            data["updated_at"] = datetime.now().isoformat()
            self.supabase.table("user_profile").upsert(data, on_conflict="external_user_key").execute()
        except Exception as e:
            print(f"Database save error: {e}")
