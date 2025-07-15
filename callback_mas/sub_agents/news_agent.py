import os
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


async def get_stock_notices(symbol: str) -> str:
    """Chinese A-Share stock notices.

    Args:
        symbol: stock symbol
    """
    long_symbol = symbol + ".SH" if symbol.startswith("6") else symbol + ".SZ"

    url = f"https://datacenter.eastmoney.com/securities/api/data/get?type=RTP_F10_ADVANCE_DETAIL_NEW&params={long_symbol}&p=1&source=HSF10&client=PC&v=04314507208280951"

    response = await httpx.AsyncClient().get(url)
    return response.json()

NewsAgent = Agent(
    name="news_agent",
    model=model,
    description="Stock news analysis agent",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_notices`查询股票的公告信息. 
    当你获取完股票的公告信息后, 你需要对公告中的利好和利空信息进行详细的分类分析, 最后输出一个评估分数, 评估分数数值为0-100, 分数越高, 表示股票表现越好.
    """,
    tools=[get_current_time, get_stock_notices],
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
)
