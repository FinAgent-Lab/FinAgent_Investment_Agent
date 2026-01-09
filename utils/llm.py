import os
import json
import re
import httpx
from typing import Any, List, Dict, Optional, Union, Type
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

load_dotenv()

class SimpleMessage:
    """response wrapper: contains both text content and tool call information"""
    def __init__(self, content: str, tool_calls: List[Dict] = None):
        self.content = content
        self.tool_calls = tool_calls or []

class StructuredLLMWrapper:
    """wrapper class for with_structured_output"""
    def __init__(self, llm, schema: Type[BaseModel]):
        self.llm = llm
        self.schema = schema

    async def ainvoke(self, messages: List[Any], **kwargs) -> BaseModel:
        # 1. inject schema
        schema_json = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        system_instruction = f"\n\nIMPORTANT: You MUST return valid JSON that matches this schema:\n{schema_json}"
        
        new_messages = messages.copy()
        if isinstance(new_messages[0], SystemMessage):
            new_messages[0] = SystemMessage(content=new_messages[0].content + system_instruction)
        else:
            new_messages.insert(0, SystemMessage(content=system_instruction))

        # 2. call LLM
        response = await self.llm.ainvoke(new_messages, response_format={"type": "json_object"}, **kwargs)
        
        # 3. remove markdown and parse
        content = response.content.strip()
        if content.startswith("```json"): content = content[7:]
        elif content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()

        try:
            json_content = json.loads(content)
            return self.schema(**json_content)
        except json.JSONDecodeError:
            print(f"JSON parsing failed. Raw: {content}")
            try:
                match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if match:
                    return self.schema(**json.loads(match.group()))
            except: pass
            raise

class ChatOpenRouter:
    """
    class that directly calls the OpenRouter API using httpx
    """
    def __init__(self, model: str = "openai/gpt-4o", temperature: float = 0.0, max_tokens: Optional[int] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in .env")

    def _convert_messages(self, messages: List[Union[BaseMessage, Dict]]) -> List[Dict]:
        """convert LangChain Message object to API request Dict (supports ToolMessage)"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(msg)
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                m = {"role": "assistant", "content": msg.content}
                if msg.additional_kwargs.get("tool_calls"):
                     m["tool_calls"] = msg.additional_kwargs["tool_calls"]
                formatted.append(m)
            elif isinstance(msg, ToolMessage):
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            else:
                formatted.append({"role": "user", "content": str(msg.content)})
        return formatted

    @retry(
        retry=retry_if_exception_type(httpx.RemoteProtocolError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def ainvoke(
        self, 
        messages: List[Union[BaseMessage, Dict]], 
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> SimpleMessage:
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(f"{error_msg}")
                    raise Exception(error_msg)

                data = response.json()
                choice = data["choices"][0]
                message_data = choice["message"]
                
                return SimpleMessage(
                    content=message_data.get("content"), 
                    tool_calls=message_data.get("tool_calls")
                )
                
            except Exception as e:
                print(f"Network/API Exception: {e}")
                raise e

    def with_structured_output(self, schema: Type[BaseModel]):
        return StructuredLLMWrapper(self, schema)