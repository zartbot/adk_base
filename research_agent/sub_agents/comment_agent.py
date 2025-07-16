import os
import re
import json
import random
import time
import datetime
import pandas as pd
import httpx
from typing import Any
from google.adk.agents import SequentialAgent

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

from dotenv import load_dotenv
from ..tools.current_time import get_current_time
from ..callback.model_cb import before_model_cb,after_model_cb
import google.genai.types as types

load_dotenv("../../.env")

model = LiteLlm(
     model=os.getenv("KIMI_MODEL"),
   # model=os.getenv("SGLANG_QWEN32B"),
   # api_base=os.getenv("SGLANG_OPENAI_BASE_URL"),
)


async def get_stock_comments(symbol: str,  tool_context: ToolContext) -> str:
    """Chinese A-Share stock comments.

    Args:
        symbol: stock symbol
    """
    
    url = f"https://guba.eastmoney.com/list,{symbol}.html"
    response = await httpx.AsyncClient().get(url)
    
    pattern = r'"post_title"\s*:\s*"([^"]*)"'
    result = re.findall(pattern, response.text)
    #temp_df = pd.DataFrame(result)
    print("Saving comment data to artifacts...")
    csv_artifact = types.Part(
        inline_data=types.Blob(
            data=json.dumps(result).encode("utf-8"),
            mime_type="application/json",
        )
    )
    await tool_context.save_artifact(f"{symbol}_stock_comment.json", csv_artifact)

    return  {
            "action": "get stock comments data",
            "status":"success",
            "symbol": symbol,
            "filename": f"{symbol}_stock_comment.json",
            "message": "Stock Comment data saved successfully in artifacts"
        }


FetchCommentAgent = Agent(
    name="fetch_comment_agent",
    model=model,
    description="Stock Comment fetch agent.",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_comments`查询股票的评论信息.
    
    你需要根据股票的 symbol 获取股票的评论信息, 并存储于`artifacts`中.
    """,
    tools=[get_current_time, get_stock_comments],
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,)


CommentAnalysisAgent = Agent(
    name="comment_analysis_agent",
    model=model,
    description="Stock Comment analysis agent.",
    instruction="""
    You are an expert financial analyst. I will provide you with a list of comments related to dedicated stock . Your tasks:

    1. **Retrive Data**
    - Retrieve the latest stock comments from the `artifacts`.
    
    2.**Sentiment Analysis:**
    - For each comment, evaluate its sentiment as '正面', '负面', or '中性'.
    - Present your evaluation in a dictionary format like: {"Comment": "正面", ...}

    3. **Comprehensive Summary & Recommendation:**
    - Summarize the overall sentiment and key points from all comments.
    - Based on the sentiment analysis statistics, output an evaluation score between 0 and 100 for each stock, where a higher score indicates better stock performance.
    """,
    output_key = "comment_analysis",    
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,)


CommentAgent =  SequentialAgent(
    name="comment_agent",
    description="Stock Comment analysis agent.",
    sub_agents=[FetchCommentAgent,CommentAnalysisAgent],
)
