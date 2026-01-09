from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage

# update logic for the state
def merge_messages(left: list, right: list):
    """conversation history is appended continuously"""
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]
    return left + right

def merge_dict(left: Dict, right: Dict) -> Dict:
    """
    dictionary data is merged with the existing value (Update)
    e.g., {'age': 30} + {'income': 5000} => {'age': 30, 'income': 5000}
    """
    if not left: left = {}
    if not right: right = {}

    return {**left, **right}

# global state structure shared by all agents
class AgentState(TypedDict):
    # 1. conversation history (cumulative)
    messages: Annotated[List[BaseMessage], merge_messages]
    
    # 2. user identifier (overwrite)
    user_id: str
    
    # 3. user profile (merged -> UserChat gradually fills it)
    user_profile: Annotated[Dict[str, Any], merge_dict]
    
    # 4. collected data (merged -> Debate, Finance share data with each other)
    collected_data: Annotated[Dict[str, Any], merge_dict]
    
    # 5. one-time control flag (overwrite)
    guardrail_result: Dict[str, Any]
    intent: str

    original_query: Optional[str]