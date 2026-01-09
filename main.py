import asyncio
import uuid
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from utils.state import AgentState
from condition.guardrail import GuardrailNode
from condition.condition import ConditionNode
from user_chat.node import UserProfileChatNode
from retriever.node import RetrieverNode
from debate.node import DebateNode
from finance.node import FinanceNode
from utils.db import get_supabase_client

load_dotenv()

def create_financial_agent():
    memory = MemorySaver()

    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail", GuardrailNode().run)
    workflow.add_node("condition", ConditionNode().run)
    workflow.add_node("user_chat", UserProfileChatNode().run)
    workflow.add_node("retriever", RetrieverNode().run)
    workflow.add_node("debate", DebateNode().run)
    workflow.add_node("finance", FinanceNode().run)

    workflow.add_edge(START, "guardrail")

    def check_safety(state: AgentState):
        result = state.get("guardrail_result", {})
        return "condition" if result.get("is_allowed", True) else END

    workflow.add_conditional_edges(
        "guardrail",
        check_safety,
        {"condition": "condition", END: END}
    )

    return workflow.compile(checkpointer=memory)

async def run_chat_session():
    app = create_financial_agent()
    TEST_USER_ID = "user_ext_001"
    
    THREAD_ID = str(uuid.uuid4())
    config = {"configurable": {"thread_id": THREAD_ID}}
    
    supabase = get_supabase_client()
    initial_profile = {}
    if supabase:
        res = supabase.table("user_profile").select("*").eq("external_user_key", TEST_USER_ID).execute()
        if res.data:
            initial_profile = res.data[0]
            print(f"Database profile loaded: {initial_profile.get('name_display')}")

    first_run = True

    print(f"[FinAgent] system started (User: {TEST_USER_ID})")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "q", "quit"]:
                print("Goodbye.")
                break
            if not user_input.strip(): continue

            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_id": TEST_USER_ID
            }
            
            if first_run:
                input_state["user_profile"] = initial_profile
                input_state["collected_data"] = {}
                first_run = False

            print("Thinking...")
            
            async for event in app.astream(input_state, config=config):
                for node_name, state_update in event.items():
                    if state_update is None: continue

                    if node_name == "debate":
                        print("\n" + "="*20 + " Debate Log " + "="*20)
                        collected = state_update.get("collected_data", {})
                        history = collected.get("debate_history", [])
                        if history:
                            for log in history: print(f"\n{log}\n{'-'*50}")
                        print("="*55 + "\n")

                    if "messages" in state_update:
                        last_msg = state_update["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            print(f"\nAgent ({node_name}):\n{last_msg.content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_chat_session())