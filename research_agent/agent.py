import os
import logging
from dotenv import load_dotenv
from typing import Optional
from google.genai import types

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import ParallelAgent,SequentialAgent,LoopAgent

from .sub_agents.news_agent import NewsAgent
from .sub_agents.market_data_agent import MarketDataAgent
from .sub_agents.comment_agent import CommentAgent
from .sub_agents.portfolio_agent import PortfolioAgent
from .sub_agents.report_agenty import ReportWriterAgent
from .tools.current_time import get_current_time

from .callback.agent_cb import before_agent_callback, after_agent_callback,add_state_cb
from .callback.model_cb import before_model_cb, after_model_cb


load_dotenv("../.env")
model = LiteLlm(
    model=os.getenv("KIMI_MODEL"),
    #model=os.getenv("SGLANG_QWEN32B"),
    #api_base=os.getenv("SGLANG_OPENAI_BASE_URL"),
)
import sys
#import logging

logging.basicConfig(
    level=logging.ERROR,  #logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)


Parallel_Research_Agent = ParallelAgent(
     name="parallel_research_agent",
     sub_agents=[MarketDataAgent, NewsAgent, CommentAgent],
     description="Runs multiple research agents in parallel to gather information."
 )

Deep_Research_Agent = SequentialAgent(
     name="deep_research_agent",
     description="Uses multi-agent-system to analyze stock data.",
     sub_agents=[Parallel_Research_Agent, ReportWriterAgent],
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
    - `PortfolioAgent` : For managing portfolio
    - `Deep_Research_Agent`: Deep research for the portfolio.

    You also have access to the following tools:
    - get_current_time
    
    """,
    sub_agents=[PortfolioAgent,Deep_Research_Agent],
    tools=[
        get_current_time,
    ],
    before_agent_callback=[before_agent_callback,add_state_cb],
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
)
