import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search

from dotenv import load_dotenv
load_dotenv("../.env")

model = LiteLlm(
    model=os.getenv("OPENROUTER_GEMINI_FLASH"), 
)

root_agent = Agent(
    name="google_search_agent",
    model=model,
    instruction="Answer questions using Google Search when needed. Always cite sources.",
    description="Professional search assistant with Google Search capabilities",
    tools=[google_search]
)