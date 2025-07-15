import os
import re
import json
import random
import time
import datetime
import pandas as pd
import httpx
from typing import Any


from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from ..tools.current_time import get_current_time
from ..callback.model_cb import before_model_cb,after_model_cb

load_dotenv("../../.env")

model = LiteLlm(
    model=os.getenv("KIMI_MODEL"),
)


async def get_stock_comments(symbol: str) -> str:
    """Chinese A-Share stock comments.

    Args:
        symbol: stock symbol
    """
    
    url = f"https://guba.eastmoney.com/list,{symbol}.html"
    response = await httpx.AsyncClient().get(url)
    
    pattern = r'"post_title"\s*:\s*"([^"]*)"'
    result = re.findall(pattern, response.text)
    return json.dumps(result)


CommentAgent = Agent(
    name="comment_agent",
    model=model,
    description="Stock Comment analysis agent.",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_comments`查询股票的评论信息
    当你获取完股票的评论信息后, 你需要对评论的情感进行分类, 例如针对看空和看多的评论进行详细的分类分析, 并针对看空和看多的评论数量, 最后输出一个评估分数, 评估分数数值为0-100, 分数越高, 表示股票表现越好.
    """,
    tools=[get_current_time, get_stock_comments],
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,)
