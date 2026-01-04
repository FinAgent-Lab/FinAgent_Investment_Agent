""" 
FinAgent Investment Agent - FastAPI Server
Multi-Agent AI 투자 자문 시스템 REST API
"""

import asyncio
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# FastAPI 앱 초기화
app = FastAPI(
    title="FinAgent Investment Agent API",
    description="Multi-Agent AI 기반 투자 자문 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수: LangGraph 앱 및 세션 관리
financial_agent = None
active_sessions: Dict[str, Dict[str, Any]] = {}


# ===== Pydantic Models =====

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    user_id: str = Field(..., description="사용자 고유 ID")
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (없으면 새로 생성)")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str
    user_id: str
    message: str
    response: str
    node_executed: Optional[str] = None
    debate_history: Optional[List[str]] = None
    timestamp: str

class ProfileResponse(BaseModel):
    """프로필 조회 응답"""
    user_id: str
    profile: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    timestamp: str
    agent_ready: bool


# ===== Helper Functions =====

def create_financial_agent():
    """LangGraph 기반 Financial Agent 생성"""
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


async def load_user_profile(user_id: str) -> Dict[str, Any]:
    """Supabase에서 사용자 프로필 로드"""
    supabase = get_supabase_client()
    if not supabase:
        return {}
    
    try:
        res = supabase.table("user_profile").select("*").eq("external_user_key", user_id).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"Error loading profile for {user_id}: {e}")
    
    return {}


# ===== Startup & Shutdown =====

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 Agent 초기화"""
    global financial_agent
    print("🚀 Initializing FinAgent...")
    financial_agent = create_financial_agent()
    print("✅ FinAgent ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    global active_sessions
    active_sessions.clear()
    print("👋 FinAgent shutdown complete.")


# ===== API Endpoints =====

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    return HealthResponse(
        status="running",
        timestamp=datetime.now().isoformat(),
        agent_ready=financial_agent is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_ready=financial_agent is not None
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 엔드포인트
    
    - 사용자 메시지를 받아 LangGraph Agent를 실행
    - 세션별로 대화 히스토리 관리
    - Debate 로그 포함 가능
    """
    if not financial_agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # 세션 ID 생성 또는 기존 세션 사용
    session_id = request.session_id or str(uuid.uuid4())
    
    # 세션 초기화
    if session_id not in active_sessions:
        profile = await load_user_profile(request.user_id)
        active_sessions[session_id] = {
            "user_id": request.user_id,
            "profile": profile,
            "first_run": True
        }

    session = active_sessions[session_id]
    config = {"configurable": {"thread_id": session_id}}

    # 입력 상태 구성
    input_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id
    }

    if session.get("first_run"):
        input_state["user_profile"] = session["profile"]
        input_state["collected_data"] = {}
        session["first_run"] = False

    # Agent 실행
    response_text = ""
    last_node = None
    debate_history = None

    try:
        async for event in financial_agent.astream(input_state, config=config):
            for node_name, state_update in event.items():
                if state_update is None:
                    continue

                last_node = node_name

                # Debate 히스토리 추출
                if node_name == "debate":
                    collected = state_update.get("collected_data", {})
                    debate_history = collected.get("debate_history", [])

                # 최종 메시지 추출
                if "messages" in state_update:
                    last_msg = state_update["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        response_text = last_msg.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    return ChatResponse(
        session_id=session_id,
        user_id=request.user_id,
        message=request.message,
        response=response_text or "No response generated",
        node_executed=last_node,
        debate_history=debate_history,
        timestamp=datetime.now().isoformat()
    )


@app.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """사용자 프로필 조회"""
    profile = await load_user_profile(user_id)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found for user: {user_id}")

    return ProfileResponse(
        user_id=user_id,
        profile=profile,
        timestamp=datetime.now().isoformat()
    )


@app.post("/profile/{user_id}")
async def update_profile(user_id: str, profile_data: Dict[str, Any]):
    """
    사용자 프로필 업데이트 (Supabase)
    """
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not available")

    try:
        # Upsert 수행
        profile_data["external_user_key"] = user_id
        res = supabase.table("user_profile").upsert(profile_data).execute()
        
        return {
            "status": "success",
            "user_id": user_id,
            "updated_profile": res.data[0] if res.data else profile_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 (메모리 정리)"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    return {
        "active_sessions": list(active_sessions.keys()),
        "count": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }


# ===== 실행 (개발 모드) =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )