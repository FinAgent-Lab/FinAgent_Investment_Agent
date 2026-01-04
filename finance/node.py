import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from finance.tools import FINANCE_TOOLS_SCHEMA, FINANCE_FUNC_MAP
from retriever.tools import search_law_documents_rag, RETRIEVER_TOOLS_SCHEMA

class FinanceNode:
    def __init__(self):
        # Reporting은 온도가 낮아야 할루시네이션이 적음
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0.2)

    async def run(self, state: AgentState) -> Command:
        collected_data = state.get("collected_data", {})
        user_query = state["messages"][0].content
        if state.get("original_query"):
            user_query = state["original_query"]

        # 1. 주제 파악 (Topic Identification)
        topic = await self._identify_report_topic(state["messages"])
        print(f"📝 Report Topic: {topic}")

        # 2. 법률 DB 무조건 조회 (Mandatory Legal Check)
        print(f"⚖️ Performing Mandatory Legal Compliance Check for: {topic}")
        legal_context = ""
        legal_search_success = False

        try:
            legal_context = await search_law_documents_rag(f"{topic} financial regulations compliance restrictions")

            # 검색 성공 여부 판단
            if legal_context and \
                    "No related legal documents found" not in legal_context and \
                    "Legal search failed" not in legal_context and \
                    "Error:" not in legal_context:
                legal_search_success = True
                print(f"✅ Legal search successful: {len(legal_context)} characters retrieved")
            else:
                print(f"⚠️ Legal search returned no results")

        except Exception as e:
            print(f"❌ Legal Search Error: {e}")
            legal_context = f"Legal search error: {str(e)}"

        # 3. 토론 내용 요약 가져오기
        debate_history = collected_data.get("debate_history", [])
        debate_summary = "\n".join(debate_history[-3:]) if debate_history else "No debate history."

        # 4. 최종 보고서 작성 프롬프트 구성
        system_prompt = f"""
        You are the **Chief Investment Officer (CIO)**.
        Write a final investment report for the user based on the gathered data.

        **CRITICAL: LEGAL COMPLIANCE CHECK**
        The following legal/regulatory information was retrieved from our Law Database. 
        **You MUST include a 'Legal & Risk Compliance' section in your report referencing this data.**
        If the data says the investment is illegal or high-risk due to regulations, you MUST warn the user strictly.
        
        <Legal Data>
        {legal_context if legal_context else "No specific legal restrictions found."}
        </Legal Data>

        **Report Structure (Markdown):**
        # [Title]
        ## 1. Executive Summary
        ## 2. Market Analysis (from Debate)
        ## 3. Product Recommendations (if any)
        ## 4. Legal & Compliance Risks (MUST use Legal Data above)
        ## 5. Final Conclusion

        **Context from Debate Team:**
        {debate_summary}

        **User Profile:**
        {json.dumps(state.get('user_profile', {}), ensure_ascii=False)}

        Write in professional Korean (한국어).
        """

        # 5. 최종 보고서 생성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Query: {user_query}\nCreate the final report now.")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            report_content = response.content

            # 6. 법률 검색 결과를 보고서 하단에 추가
            legal_appendix = "\n\n---\n\n## 📋 법률 검색 결과\n\n"

            if legal_search_success:
                legal_appendix += f"✅ **법률 데이터베이스 검색 완료**\n\n"
                legal_appendix += f"검색어: `{topic} financial regulations compliance restrictions`\n\n"
                legal_appendix += "**검색된 법률 문서:**\n\n"
                legal_appendix += f"```\n{legal_context[:500]}...\n```\n"
                legal_appendix += "\n*전체 법률 정보가 위 보고서 작성에 반영되었습니다.*"
            else:
                legal_appendix += "⚠️ **WARNING: 법률 검색 실패**\n\n"
                legal_appendix += f"검색어: `{topic} financial regulations compliance restrictions`\n\n"
                legal_appendix += "**상태:** 관련 법률 문서를 찾지 못했습니다.\n\n"
                legal_appendix += "**원인:**\n"
                legal_appendix += "- 법률 데이터베이스에 해당 주제의 문서가 없음\n"
                legal_appendix += "- 또는 검색 중 오류 발생\n\n"
                legal_appendix += f"**검색 결과:** `{legal_context}`\n\n"
                legal_appendix += "⚠️ *본 보고서는 법률 검토 없이 작성되었으므로 투자 결정 시 주의가 필요합니다.*"

            # 최종 응답에 법률 검색 정보 추가
            final_report = report_content + legal_appendix
            response = AIMessage(content=final_report)

        except Exception as e:
            print(f"Finance Report Generation Error: {e}")
            response = AIMessage(content=f"보고서 생성 중 오류가 발생했습니다: {str(e)}")

        # 최종 결과 반환
        return Command(
            update={"messages": [response]},
            goto="__end__"
        )

    async def _identify_report_topic(self, messages: list) -> str:
        prompt = "Extract the main financial subject (e.g., Samsung Electronics, US Tech ETF) from the conversation. Return ONLY the subject name."
        res = await self.llm.ainvoke(messages + [HumanMessage(content=prompt)])
        return res.content.strip()

    async def _execute_react(self, query: str, system_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {query}\n\nFind products and write the report.")
        ]

        for _ in range(2):
            response = await self.llm.ainvoke(messages, tools=FINANCE_TOOLS_SCHEMA)

            if not response.tool_calls:
                return response.content

            messages.append(AIMessage(content=response.content or "", additional_kwargs={"tool_calls": response.tool_calls}))

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]

                print(f"  Finance Tool: {func_name}({args_str})")

                try:
                    func = FINANCE_FUNC_MAP.get(func_name)
                    if func:
                        args = json.loads(args_str)
                        result = func(**args)
                    else:
                        result = "Function not found"
                except Exception as e:
                    result = f"Error: {e}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        final_res = await self.llm.ainvoke(messages)
        return final_res.content

    async def _sync_user_db(self, user_id: str, profile: dict, topic: str, report: str):
        try:
            now = datetime.now().isoformat()
            report_data = {
                "user_id": user_id,
                "query_summary": topic,
                "final_report": report,
                "created_at": now
            }
            self.supabase.table("advisory_reports").insert(report_data).execute()

            self.supabase.table("user_profile").update({"updated_at": now}).eq("external_user_key", user_id).execute()
            print(f"Report Saved: {topic}")

        except Exception as e:
            print(f"DB Sync Error: {e}")
