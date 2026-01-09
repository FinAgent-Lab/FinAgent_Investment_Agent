import os
import json
import re
from typing import Optional
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime
from tavily import TavilyClient

from utils.db import get_supabase_client
from utils.embedding import get_embedding

# product search (SQL Query)
def search_products_sql(
    keyword: Optional[str] = None, 
    category: Optional[str] = None,
    sort_by: Optional[str] = None 
) -> str:
    """
    Search financial products (ETFs, Funds) from the database (SQL).
    Search 'product_name' by keyword, and sort by 'fee' or 'expected_return' using sort_by.
    """
    supabase = get_supabase_client()
    if not supabase: return "Error: DB connection failed."

    try:
        query = supabase.table("investment_products").select("*")
        
        # 1. keyword search (product_name)
        if keyword and keyword.lower() not in ["low fee", "cheap", "best", "good", "high return"]:
            query = query.ilike("product_name", f"%{keyword}%")
        
        # 2. category filter (product_group)
        if category:
            query = query.ilike("product_group", category)

        # 3. sort logic (database columns: fee, expected_return)
        if sort_by == "fees_asc":
            query = query.order("fee", desc=False) # lowest fee first
        elif sort_by == "return_desc":
            query = query.order("expected_return", desc=True) # highest return first
        else:
            # default sort: relevance (keyword) or expected return
            query = query.order("expected_return", desc=True)

        response = query.limit(5).execute()
        
        if not response.data:
            return f"No products found for the given conditions: {keyword}, {category}, {sort_by}"

        results = []
        for item in response.data:
            info = (
                f"- {item.get('product_name')} ({item.get('product_code')})\n"
                f"  [Fee]: {item.get('fee')}% | [Expected Return]: {item.get('expected_return')}%\n"
                f"  [Type]: {item.get('product_group')} | [Risk]: {item.get('risk_category')}\n"
                f"  [Description]: {item.get('description', '')[:50]}..." 
            )
            results.append(info)
            
        return "\n\n".join(results)

    except Exception as e:
        return f"DB Error: {str(e)}"

# document search (RAG)
async def search_law_documents_rag(query: str) -> str:
    """
    Search legal regulations, compliance guidelines, and financial laws.
    Used to ensure the advice complies with current regulations.
    """
    supabase = get_supabase_client()
    if not supabase: return "Error: DB connection failed."

    # 임베딩 생성
    vector = await get_embedding(query)
    
    try:
        response = supabase.rpc(
            "match_law_documents", 
            {"query_embedding": vector, "match_threshold": 0.4, "match_count": 3}
        ).execute()
        
        if not response.data: return "No related legal documents found."
        
        # 결과 포맷에 [Legal Document] 명시
        return "\n".join([f"[Legal Document: {d.get('title')}]\n{d.get('content')}" for d in response.data])
    except Exception as e:
        return f"Legal search error: {str(e)}"

# web search (API)
def search_web(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key: return "Error: Tavily Key missing."
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query, search_depth="basic", max_results=2)
        return json.dumps(res["results"], ensure_ascii=False)
    except Exception as e:
        return f"Web search error: {e}"

def get_realtime_price(ticker: str) -> str:
    """Get real-time price for a specific ticker (domestic or foreign)."""
    try:
        if bool(re.fullmatch(r"\d{6}", ticker)): # domestic market
            df = fdr.DataReader(ticker, datetime.now().year)
            if df.empty: return f"No Data: {ticker}"
            latest = df.iloc[-1]
            return json.dumps({"ticker": ticker, "price": int(latest['Close']), "date": str(latest.name)}, ensure_ascii=False)
        else: # foreign market
            info = yf.Ticker(ticker).info
            return json.dumps({"ticker": ticker, "price": info.get("currentPrice")}, ensure_ascii=False)
    except Exception as e:
        return f"Price lookup failed: {e}"

# tool schema and map
RETRIEVER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_products_sql",
            "description": (
                "Execute a structured SQL query to find financial products (ETFs, Funds). "
                "Use this tool when the user asks for specific products based on **Theme, Sector, Region** (keyword), "
                "or **Ranking/Comparison** (sort_by). "
                "Do NOT use this for general definitions."
            ),
            "parameters": {
                "type": "object", 
                "properties": {
                    "keyword": {
                        "type": "string", 
                        "description": (
                            "The specific **Subject, Sector, or Region** to filter by (e.g., 'Semiconductor', 'US', 'Samsung'). "
                            "**WARNING:** Do NOT put adjectives like 'cheap', 'best', 'low fee' here. "
                            "If the user asks for 'all products', leave this empty."
                        )
                    },
                    "category": {
                        "type": "string", 
                        "description": "Product type filter. Values: 'ETF', 'FUND', 'STOCK', 'BOND'. Optional."
                    },
                    "sort_by": {
                        "type": "string", 
                        "enum": ["fees_asc", "return_desc"], 
                        "description": (
                            "Sorting criteria. REQUIRED if the user uses adjectives like 'cheapest' or 'highest return'. "
                            "- 'fees_asc': Lowest fees first (Cheap). "
                            "- 'return_desc': Highest expected return first (Best performance)."
                        )
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_law_documents_rag",  # 이름 변경
            "description": (
                "Search for financial laws, regulations, and compliance guidelines. "
                "CRITICAL: You MUST use this tool to verify if the investment advice is legally compliant "
                "before giving a final answer."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_realtime_price",
            "description": "Get real-time price for a specific ticker.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    },
     {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search latest news or general info from the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

RETRIEVER_FUNC_MAP = {
    "search_products_sql": search_products_sql,
    "search_law_documents_rag": search_law_documents_rag,
    "get_realtime_price": get_realtime_price,
    "search_web": search_web
}