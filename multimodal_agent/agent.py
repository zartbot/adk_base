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
    model=os.getenv("LOCAL_GEMMA3N"),
)


root_agent = Agent(
    name="multimodal_agent",
    model=model,
    description="MultiModal Agent",
    instruction="""
    你是一个非常有用的助手.
    """,
   # tools=[get_current_time, get_stock_hist, get_stock_notices],
)
