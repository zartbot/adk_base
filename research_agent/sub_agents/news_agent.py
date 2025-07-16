import os
import random
import time
import datetime
import pandas as pd
import json
import httpx
from typing import Any


from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import SequentialAgent
import google.genai.types as types

from dotenv import load_dotenv
from ..tools.current_time import get_current_time
from ..callback.model_cb import before_model_cb,after_model_cb


load_dotenv("../../.env")

model = LiteLlm(
     model=os.getenv("KIMI_MODEL"),
   # model=os.getenv("SGLANG_QWEN32B"),
   # api_base=os.getenv("SGLANG_OPENAI_BASE_URL"),
)


async def get_stock_notices(symbol: str , tool_context: ToolContext) -> str:
    """Chinese A-Share stock notices.

    Args:
        symbol: stock symbol
    """
    long_symbol = symbol + ".SH" if symbol.startswith("6") else symbol + ".SZ"

    url = f"https://datacenter.eastmoney.com/securities/api/data/get?type=RTP_F10_ADVANCE_DETAIL_NEW&params={long_symbol}&p=1&source=HSF10&client=PC&v=04314507208280951"

    response = await httpx.AsyncClient().get(url)
    
    print("Saving news data to artifacts...")
    csv_artifact = types.Part(
        inline_data=types.Blob(
            data=json.dumps(response.json()).encode("utf-8"),
            mime_type="application/json",
        )
    )
    await tool_context.save_artifact(f"{symbol}_stock_news.json", csv_artifact)

    return  {
            "action": "get stock news data",
            "status":"success",
            "symbol": symbol,
            "filename": f"{symbol}_stock_news.json",
            "message": "Stock news data saved successfully in artifacts"
        }


NewsFetchAgent = Agent(
    name="news_fetch_agent",
    model=model,
    description="Stock news fetch agent",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_notices`查询股票的公告信息. 
    """,
    tools=[get_current_time, get_stock_notices],
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
)


NewsAnalysisAgent = Agent(
    name="news_analysis_agent",
    model=model,
    description="Stock news analysis agent.",
    instruction="""
    You are an expert financial analyst. I will provide you with a list of news related to dedicated stock . Your tasks:

    1. **Retrive Data**
    - Retrieve the latest stock news from the `artifacts`.
    
    2.**Sentiment Analysis:**
    - For each news, evaluate its sentiment as '正面', '负面', or '中性'.
    - Present your evaluation in a dictionary format like: {"news": "正面", ...}

    3. **Comprehensive Summary & Recommendation:**
    - Summarize the overall sentiment and key points from all news.
    - Based on the sentiment analysis statistics, output an evaluation score between 0 and 100 for each stock, where a higher score indicates better stock performance.
    """,
    output_key = "news_analysis",
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,)

NewsAgent =  SequentialAgent(
    name="news_agent",
    description="Stock News analysis agent.",
    sub_agents=[NewsFetchAgent,NewsAnalysisAgent],
)
