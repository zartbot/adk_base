import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

from .sub_agents.news_agent import NewsAgent
from .sub_agents.market_data_agent import MarketDataAgent
from .sub_agents.comment_agent import CommentAgent
from .sub_agents.portfolio_agent import PortfolioAgent
from .tools.current_time import get_current_time

load_dotenv("../.env")
model = LiteLlm(
    model=os.getenv("KIMI_MODEL"),
)

root_agent = Agent(
    name="basic_multi_agent",
    model=model,
    description="Manager agent",
    instruction="""
    You are a manager agent that is responsible for overseeing the work of the other agents.
    
    Always delegate the task to the appropriate agent. Use your best judgement 
    to determine which agent to delegate to.
    
    **Portfolio Information:**
    <portfolio>
    Portfolio information: {portfolio}
    </portfolio>
    
    You are responsible for delegating tasks to the following agent:
    - `MarketDataAgent` : For getting market data.
    - `NewsAgent` : For getting stocks's news.
    - `CommentAgent` : For getting stocks's comments.
    - `PortfolioAgent` : For managing portfolio

    You also have access to the following tools:
    - get_current_time
    
    """,
    sub_agents=[MarketDataAgent, NewsAgent,CommentAgent,PortfolioAgent],
    tools=[
        get_current_time,
    ],
)
