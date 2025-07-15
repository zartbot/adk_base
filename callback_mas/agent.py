import os
from dotenv import load_dotenv
from typing import Optional
from google.genai import types

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm


from .sub_agents.news_agent import NewsAgent
from .sub_agents.market_data_agent import MarketDataAgent
from .sub_agents.comment_agent import CommentAgent
from .sub_agents.portfolio_agent import PortfolioAgent
from .tools.current_time import get_current_time

from .callback.agent_cb import before_agent_callback, after_agent_callback,add_state_cb
from .callback.model_cb import before_model_cb, after_model_cb


load_dotenv("../.env")
model = LiteLlm(
    model=os.getenv("KIMI_MODEL"),
)
import sys
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

    

root_agent = Agent(
    name="callback_mas_agent",
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
    before_agent_callback=[before_agent_callback,add_state_cb],
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,


)
