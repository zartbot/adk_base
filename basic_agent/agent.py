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

load_dotenv("../.env")

model = LiteLlm(
    model=os.getenv("LOCAL_QWEN30B"),
)


async def get_current_time() -> str:
    """
    获取当前时间
    Returns:
        str: 当前的时间, 格式为：2023-05-01 12:00:00
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def get_stock_notices(symbol: str) -> str:
    """Chinese A-Share stock notices.

    Args:
        symbol: stock symbol
    """
    long_symbol = symbol + ".SH" if symbol.startswith("6") else symbol + ".SZ"

    url = f"https://datacenter.eastmoney.com/securities/api/data/get?type=RTP_F10_ADVANCE_DETAIL_NEW&params={long_symbol}&p=1&source=HSF10&client=PC&v=04314507208280951"

    response = await httpx.AsyncClient().get(url)
    return response.json()


async def make_hq_request(url: str, params: object) -> dict[str, Any] | None:
    """Make a request to the HQ API with proper error handling."""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


async def get_stock_hist(
    symbol: str, period: str, start_date: str, end_date: str
) -> str:  # pd.DataFrame:
    """Chinese A-Share stock historical data.

    Args:
        symbol: stock symbol
        period: daily, weekly, monthly
        start_date: start date
        end_date: End date
    """

    market_code = 1 if symbol.startswith("6") else 0
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    date1 = start_date.replace("-", "")
    date2 = end_date.replace("-", "")
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict[period],
        "fqt": "1",  # 1. 前复权 2. 后复权 0. 不复权
        "secid": f"{market_code}.{symbol}",
        "beg": date1,
        "end": date2,
    }
    data_json = await make_hq_request(url, params)

    if not (data_json["data"] and data_json["data"]["klines"]):
        return pd.DataFrame()
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
    temp_df["股票代码"] = symbol
    temp_df.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
        "股票代码",
    ]
    temp_df["日期"] = pd.to_datetime(temp_df["日期"], errors="coerce").dt.date
    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"], errors="coerce")
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    temp_df = temp_df[
        [
            "日期",
            "股票代码",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "振幅",
            "涨跌幅",
            "涨跌额",
            "换手率",
        ]
    ]

    return temp_df.to_json()


root_agent = Agent(
    name="basic_agent",
    model=model,
    description="Stock historical data",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_notices`查询股票的公告信息.
    """,
    tools=[get_current_time, get_stock_hist, get_stock_notices],
)
