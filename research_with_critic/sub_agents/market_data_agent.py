import os
import random
import time
import datetime
import pandas as pd
import httpx
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import SequentialAgent

from dotenv import load_dotenv
from ..tools.current_time import get_current_time
from ..callback.model_cb import before_model_cb,after_model_cb
import google.genai.types as types


load_dotenv("../../.env")

model = LiteLlm(
       model=os.getenv("KIMI_MODEL"),
       #model=os.getenv("SGLANG_QWEN32B"),
       #api_base=os.getenv("SGLANG_OPENAI_BASE_URL"),
)


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
    symbol: str, period: str, start_date: str, end_date: str, tool_context: ToolContext
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
    print("Saving Market data to artifacts...")
    csv_artifact = types.Part(
        inline_data=types.Blob(
            data=temp_df.to_csv(index=False).encode("utf-8"),
            mime_type="text/csv",
        )
    )
    await tool_context.save_artifact(f"{symbol}_stock_hist.csv", csv_artifact)

    return  {
            "action": "get stock market data",
            "status":"success",
            "symbol": symbol,
            "filename": f"{symbol}_stock_hist.csv",
            "message": "Market data saved successfully in artifacts"
        }


FetchMarketDataAgent = Agent(
    name="fetch_market_data_agent",
    model=model,
    description="Stock market data",
    instruction="""
    你是一个非常有用的助手, 可以通过调用`get_current_time`获取当前的时间, 并通过`get_stock_hist`查询股票的行情信息. 
    
    当你需要执行行情查询前, 首先需要通过`get_current_time`获取当前的时间
    """,
    tools=[get_current_time, get_stock_hist],
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
)



TechnicalAnalysisAgent = Agent(
    name="TechnicalAnalysisAgent",
    model=model,
    description="Technical analysis for dedicated stocks.",
    instruction="""
    你是一个非常资深的股票分析师, 并擅长通过K 线图、MACD、RSI 等技术指标的方式来分析和预测股票未来的走势和风险.
    
    你需要分析的数据以CSV格式存储于`artifacts`中. 
    
    请根据Artifacts中的对应股票的行情数据:
    1. 为每一只股票输出一个关于行情走势的技术分析报告, 并通过K线图、MACD、RSI、BOLLinger Bands 等指标来描述股票的走势和风险
    2. 比较投资组合内的不同股票的波动率,换手率, 价格振幅, 波动周期, 成交活跃度, 风险收益特征
    3. 计算投资组合的beta, alpha, sharpe ratio, sortino ratio等指标
    4. 基于技术分析的结果, 预测股票未来的走势和风险. 
    
    """,
    output_key = "technical_analysis",    
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
 )
 
 
 # Create the sequential agent with minimal callback
MarketDataAgent = SequentialAgent(
    name="MarketDataAgent",
    sub_agents=[FetchMarketDataAgent, TechnicalAnalysisAgent],
    description="A pipeline that validates, scores, and recommends actions for sales leads",

)
