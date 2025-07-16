import os
import pandas as pd
from typing import Any


from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import SequentialAgent
import google.genai.types as types

from dotenv import load_dotenv
from ..tools.current_time import get_current_time
from ..callback.model_cb import before_model_cb,after_model_cb


load_dotenv("../../.env")

model = LiteLlm(
     model=os.getenv("KIMI_MODEL"),
   # model=os.getenv("SGLANG_QWEN32B"),
   # api_base=os.getenv("SGLANG_OPENAI_BASE_URL"),
)

ReportWriterAgent = Agent(
    name="report_writer_agent",
    model=model,
    description="Stock news analysis agent.",
    instruction="""
    You are an expert financial analyst.
    
    Your task is to analyze the portfolio and synthesize a comprehensive report by combining the following information:
    - Technical analysis : {technical_analysis}
    - Comment analysis:: {comment_analysis}
    - News analysis: {news_analysis}
    
    **Portfolio Information:**
    <portfolio>
    Portfolio information: {portfolio}
    </portfolio>
    
    Your report should be concise and informative. Create a well-formatted report with:
    1. A summary of the portfolio's current market value and performance.
    2. Technical analysis section, give the detailed analysis of {technical_analysis}  and its impact on the portfolio.
    3. News analysis section, give the detailed analysis of {news_analysis}  and its impact on the portfolio.
    4. Comment analysis section, give the detailed analysis of {comment_analysis} and its impact on the portfolio.
    5. An analysis of the portfolio's risk and potential system risks.
    6. Summary of the portfolio's performance and potential risks.
    7. Recommendations based on any concerning metrics     
    8. Add an appendix section:
       - print the original {technical_analysis}
       - print the original {comment_analysis} 
       - print the original {news_analysis}
    
    Use markdown formatting to make the report readable and professional.
    Highlight any concerning values and provide practical recommendations.
    At the end of the report, include a notice with `**风险提示**：本报告基于公开数据，不构成投资建议；市场有风险，投资需谨慎。`
    """,
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,)
