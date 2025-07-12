import os
import datetime
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv

load_dotenv("../.env")

model = LiteLlm(
    model=os.getenv("KIMI_MODEL"),
)

async def get_current_time() -> str:
    """
    获取当前时间
    Returns:
        str: 当前的时间, 格式为：2023-05-01 12:00:00
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


root_agent = Agent(
    name="browser_agent",
    model=model,
    description="Agent for browser use",
    instruction="""
    你是一个非常有用的助手, 可以通过使用playwright浏览器工具获取信息
    """,
    tools=[get_current_time,
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y", 
                     "@playwright/mcp@latest"
                ],
                timeout=20)
            )
        ],
)