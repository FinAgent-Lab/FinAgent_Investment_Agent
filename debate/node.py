import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from debate.tools import DEBATE_TOOLS_SCHEMA, TOOL_FUNC_MAP

class DebateNode:
    """
    [Expert Debate Block - 5-Round Debate]
    3 experts (Conservative, Aggressive, Balanced) argue fiercely in Korean,
    going through the process of opening statement -> rebuttal -> deep dive -> closing argument.
    """
    def __init__(self):
        # recommended for Korean debate performance
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7, max_tokens=512)
        self.max_rounds = 3  # extended to 5 rounds

    async def run(self, state: AgentState) -> Command:
        messages = state["messages"]
        user_input = messages[-1].content
        
        topic = await self._resolve_topic(user_input, messages)
        
        print(f"\nDebate Topic: {topic}")
        
        debate_log = []

        # define expert personas (Korean)
        agents = [
            {
                "role": "Conservative",
                "name": "보수적 투자 전문가",
                "style": "리스크 관리 최우선, 회의적, 팩트 체크 중시, 규제/금리 민감",
                "opponent": "Aggressive"
            },
            {
                "role": "Aggressive",
                "name": "공격적 투자 전문가",
                "style": "미래 성장성 중시, 혁신 기술, 낙관적, 하이 리스크 하이 리턴",
                "opponent": "Conservative"
            },
            {
                "role": "Balanced",
                "name": "중립적 투자 전문가",
                "style": "데이터 기반의 중용, 시장 흐름 파악, 양쪽 의견 조율",
                "opponent": "Both"
            }
        ]

        # proceed through debate rounds (1~5)
        for round_i in range(1, self.max_rounds + 1):
            # set theme for each round
            if self.max_rounds == 5:
                if round_i == 1:
                    stage_name = "Round 1: Opening Statement"
                elif round_i in [2, 3]:
                    stage_name = f"Round {round_i}: Rebuttal"
                elif round_i == 4:
                    stage_name = "Round 4: Deep Dive"
                else:
                    stage_name = "Round 5: Closing Argument"
            
            elif self.max_rounds == 3:
                if round_i == 1:
                    stage_name = "Round 1: Opening Statement"
                elif round_i == 2:
                    stage_name = f"Round 2: Rebuttal and Deep Dive"
                else:
                    stage_name = "Round 3: Closing Argument"

            print(f"\n{stage_name}")
            
            for agent in agents:
                role_eng = agent["role"]
                role_kr = agent["name"]
                style = agent["style"]
                opponent = agent["opponent"]
                
                # set instructions for each round
                if round_i == 1:
                    instruction = (
                        f"당신은 '{role_kr}'입니다. 성향: {style}.\n"
                        f"주제 '{topic}'에 대한 당신의 핵심 입장을 명확히 밝히십시오.\n"
                        "도구('search_news', 'get_market_data')를 사용하여 근거 데이터를 제시하세요.\n"
                        "반드시 **한국어**로 답변하세요. 대답은 최대한 간결하게 512토큰 이내로 작성하세요."
                    )
                elif round_i in [2, 3] and self.max_rounds == 5:
                    instruction = (
                        f"당신은 '{role_kr}'입니다.\n"
                        f"앞선 토론 내용, 특히 반대 성향인 '{opponent}'의 주장을 강하게 반박하십시오.\n"
                        "상대방의 논리적 허점이나 데이터의 오류를 지적하세요.\n"
                        "필요하다면 도구를 추가로 사용하여 반박 근거를 찾으세요.\n"
                        "반드시 **한국어**로 답변하세요. 대답은 최대한 간결하게 512토큰 이내로 작성하세요."
                    )
                elif round_i == 4 and self.max_rounds == 5:
                    instruction = (
                        f"당신은 '{role_kr}'입니다.\n"
                        "토론이 막바지에 다다랐습니다. 놓치고 있는 시장의 숨겨진 리스크나 기회를 심층적으로 분석하십시오.\n"
                        "단순한 주장을 넘어, 거시 경제나 산업 트렌드와 연결하여 통찰력을 보여주세요.\n"
                        "반드시 **한국어**로 답변하세요. 대답은 최대한 간결하게 512토큰 이내로 작성하세요."
                    )
                elif round_i == 2 and self.max_rounds == 3:
                    instruction = (
                        f"당신은 '{role_kr}'입니다.\n"
                        "앞선 토론 내용, 특히 반대 성향인 '{opponent}'의 주장을 강하게 반박하십시오.\n"
                        "상대방의 논리적 허점이나 데이터의 오류를 지적하세요.\n"
                        "필요하다면 도구를 추가로 사용하여 반박 근거를 찾으세요.\n"
                        "이러한 과정을 통해 놓치고 있는 시장의 숨겨진 리스크나 기회를 심층적으로 분석하십시오.\n"
                        "단순한 주장을 넘어, 거시 경제나 산업 트렌드와 연결하여 통찰력을 보여주세요.\n"
                        "반드시 **한국어**로 답변하세요. 대답은 최대한 간결하게 512토큰 이내로 작성하세요."
                    )
                else: # Round 5
                    instruction = (
                        f"당신은 '{role_kr}'입니다.\n"
                        "마지막 발언 기회입니다. 투자자를 설득하기 위한 최종 결론을 내리십시오.\n"
                        "당신의 주장이 왜 옳은지 요약하고, 구체적인 행동(매수/매도/보류)을 제안하세요.\n"
                        "반드시 **한국어**로 답변하세요. 대답은 최대한 간결하게 512토큰 이내로 작성하세요."
                    )

                print(f"\n{role_kr}: ", end="")
                
                argument = await self._agent_turn(
                    role=role_kr,
                    topic=topic,
                    history=debate_log,
                    system_prompt=instruction
                )
                
                # log the argument
                log_entry = f"[{role_kr}]: {argument}"
                debate_log.append(log_entry)
                
                # print the argument
                print(argument)

        # --- final verdict ---
        print(f"\nThe CIO is making the final verdict...")
        verdict_msg = await self._judge_verdict(topic, debate_log)
        topic = await self._resolve_topic(user_input, messages)
        
        return Command(
            update={
                "messages": [AIMessage(content=verdict_msg)],
                "collected_data": {
                    "debate_history": debate_log, 
                    "report_topic": topic
                }
            },
            goto="__end__" 
        )

    async def _agent_turn(self, role: str, topic: str, history: list, system_prompt: str) -> str:
        """generate agent argument (including tool usage)"""
        
        # format history (explicit Korean context)
        history_text = ""
        if history:
            history_text = "--- Previous debate history ---\n" + "\n\n".join(history) + "\n---------------------"
        else:
            history_text = "(First statement)"

        context_reminder = f"""
        \n[IMPORTANT CONTEXT]
        The current main topic of discussion is derived as: "{topic}".
        When using tools like 'search_news', MUST include the specific company name (e.g., Nvidia, Samsung) in your query argument.
        Do NOT search for generic terms like 'competitors' or 'market'.
        """

        messages = [
            SystemMessage(content=system_prompt + context_reminder),
            HumanMessage(content=f"Topic: {topic}\n\n{history_text}\n\nIt's your turn to argue logically.")
        ]

        # tool usage loop (maximum 2 times)
        for _ in range(2):
            response = await self.llm.ainvoke(messages, tools=DEBATE_TOOLS_SCHEMA)
            
            if not response.tool_calls:
                return response.content

            ai_msg = AIMessage(content=response.content or "", additional_kwargs={"tool_calls": response.tool_calls})
            messages.append(ai_msg)

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]
                
                print(f"  {role} uses tool: {func_name}")

                try:
                    func = TOOL_FUNC_MAP.get(func_name)
                    args = json.loads(args_str)
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                except Exception as e:
                    result = f"Error: {e}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
        
        final_res = await self.llm.ainvoke(messages)
        
        return final_res.content

    async def _judge_verdict(self, topic: str, log: list) -> str:
        """final verdict by the CIO (Korean)"""
        prompt = f"""
        당신은 AI 투자 자문 시스템의 최고 투자 책임자(CIO/재판관)입니다.
        주제 '{topic}'에 대한 3명 전문가의 5라운드 토론 내용을 검토했습니다.

        참여자:
        1. 보수적 투자 전문가: 리스크 관리 중시
        2. 공격적 투자 전문가: 수익성 중시
        3. 중립적 투자 전문가: 균형점 모색

        **지시사항:**
        1. 각 전문가의 핵심 주장을 요약하십시오.
        2. 가장 논리적이고 데이터에 기반한 주장이 누구인지 평가하십시오.
        3. 현재 시장 상황을 고려하여 최종 투자 결론(Final Verdict)을 내리십시오.
        4. 모든 답변은 **전문적이고 신뢰감 있는 한국어**로 작성하십시오.
        5. **[중요] 정보의 공백(Missing Link)을 찾으십시오.**
           - 토론에서 간과된 리스크는 없습니까? (예: 환율, 금리, 지정학적 리스크)
           - 더 구체적인 데이터가 필요한 부분이 있습니까?
        
        **출력 형식 (한국어):**
        - 먼저 토론 결론을 간략히 브리핑합니다.
        - 그 후, **"완벽한 보고서 작성을 위해 ~~에 대한 추가 조사를 진행해볼까요?"** 라고 유저에게 제안하십시오.
        - 만약 토론이 완벽하다면, 유저에게 추가로 궁금한 점이 있는지 묻거나 보고서 작성을 승인해달라고 하십시오.
        - 답변은 최대한 간결하게 512토큰 이내로 작성하세요.
        
        **토론 기록:**
        {chr(10).join(log)}
        """
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return res.content

    async def _resolve_topic(self, user_input: str, history: list) -> str:
        """
        Contextualize the user's input based on conversation history.
        It converts ambiguous references (e.g., "What about competitors?") 
        into a standalone topic (e.g., "Analyze recent movements of Nvidia's competitors").
        """
        # 히스토리가 없으면 사용자 입력 그대로 반환
        if not history:
            return user_input

        # 최근 대화 내용 구성 (토큰 절약을 위해 최근 6개만)
        recent_history = history[-6:] 
        history_text = ""
        for msg in recent_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            # ToolMessage 등은 제외하고 텍스트만 추출
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                history_text += f"{role}: {msg.content}\n"
        
        # 문맥 재구성 프롬프트 (Contextualization Prompt)
        prompt = f"""
        **Conversation History:**
        {history_text}
        
        **User's Latest Input:** "{user_input}"
        
        **Task:**
        Based on the conversation history, rewrite the user's latest input into a **standalone debate topic** that includes the specific entity (Company/Ticker) being discussed.
        
        - If the user says "Check competitors", and they were discussing 'Nvidia', rewrite it to "Analyze the recent movements of Nvidia's competitors".
        - If the user switches the topic (e.g., "What about Tesla?"), use the new topic.
        - **CRITICAL:** Do NOT answer the question. Only rewrite the topic string in Korean.
        
        **Rewritten Topic:**
        """
        
        # LLM에게 문맥 결합 요청
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        new_topic = response.content.strip()
        print(f"🔄 Contextualized Topic: {user_input} -> {new_topic}") # 디버깅용 로그
        
        return new_topic