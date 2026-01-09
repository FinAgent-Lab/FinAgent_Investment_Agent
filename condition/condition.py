import json
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from utils.llm import ChatOpenRouter

# same required fields as UserChatNode
REQUIRED_FIELDS = [
    "name_display", "age_range", "income_bracket", 
    "invest_experience_yr", "financial_knowledge_level", 
    "current_holdings_note", "preferred_asset_types",
    "risk_tolerance_level", "total_investable_amt", 
    "goal_type", "goal_description", "preferred_style"
]

class ConditionNode:
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0)
        
        self.router_prompt = """
        You are an intelligent Router for a Financial AI System.
        Analyze the conversation context and the user's latest input to select the next step.

        **Context Analysis:**
        - Previous AI Action: Did the AI suggest additional research? Did the AI ask for a preference?
        
        **Routing Logic (Strict Rules):**

        1. **`report_generation` (Go to FinanceNode)**
           - Trigger: When the user wants to **FINALIZE** and see the result.
           - Scenarios:
             - User says "Write the report", "Summarize now", "Give me the conclusion".
             - User **REJECTS** further debate/research and asks for the result.
             - User agrees to the AI's suggestion to write a report.

        2. **`investment_advisory` (Go to DebateNode)**
           - Trigger: When the user wants **MORE** discussion, analysis, or comparison.
           - Scenarios:
             - User asks a deep question: "Compare Nvidia vs Tesla", "What about risks?".
             - User agrees to the AI's suggestion for *additional research*.
             - User answers a preference question to refine the strategy.

        3. **`market_data` (Go to RetrieverNode)**
           - Trigger: Simple factual queries.
           - Scenarios: "Price of Apple", "What is ETF", "Find low-fee funds".

        4. **`profile_management` (Go to UserChatNode)**
           - Trigger: Explicit profile changes.
           - Scenarios: "Change my income", "Update my risk level".

        **Response Format:**
        Output ONLY valid JSON: {"route": "route_name", "reason": "brief reasoning"}
        """

    async def run(self, state: dict) -> Command:
        guardrail_result = state.get("guardrail_result", {})
        user_profile = state.get("user_profile") or {} 
        user_input = state["messages"][-1].content

        if not guardrail_result.get("is_allowed", True):
            return Command(goto="__end__")

        # 2. check profile completeness (if any field is missing, route to UserChat)
        missing_fields = []
        for field in REQUIRED_FIELDS:
            val = user_profile.get(field)
            if val is None or val == "" or val == []:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"Profile Incomplete. Routing to [UserChat].")

            update_dict = {}
            if not state.get("original_query"):
                print(f"   👉 Saving original query: {user_input}")
                update_dict["original_query"] = user_input

            return Command(update=update_dict, goto="user_chat")

        # 3. profile update intent (if already completed and user asks to update)
        if guardrail_result.get("category") == "profile_update":
            explicit_keywords = ["수정", "변경", "바꿔", "update", "change", "다시 입력"]
            if any(k in user_input for k in explicit_keywords):
                return Command(goto="user_chat")

        if guardrail_result.get("category") == "general_chat":
             return Command(goto="retriever")

        # 4. context-based smart routing
        history_context = ""
        if len(state["messages"]) > 1:
            last_bot_msg = state["messages"][-2].content
            history_context = f"AI's Last Question: {last_bot_msg}\n"

        route_decision = await self._decide_route(user_input, history_context)
        target = route_decision.get("route", "market_data")
        
        node_map = {
            "market_data": "retriever",
            "investment_advisory": "debate",
            "report_generation": "finance",
            "profile_management": "user_chat"
        }
        
        next_node = node_map.get(target, "retriever")
        print(f"Routing to [{next_node}] (Reason: {route_decision.get('reason')})")
        
        return Command(goto=next_node)

    async def _decide_route(self, query: str, context: str) -> dict:
        full_prompt = f"{self.router_prompt}\n\n--- Context ---\n{context}"
        messages = [
            SystemMessage(content=full_prompt),
            HumanMessage(content=f"User Input: {query}")
        ]
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception as e:
            print(f"Routing Error: {e}")
            return {"route": "market_data", "reason": "Fallback due to error"}