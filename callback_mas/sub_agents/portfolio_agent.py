import os
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv


from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.sessions import InMemorySessionService
from typing import Optional
from google.genai import types
from ..callback.model_cb import before_model_cb,after_model_cb

load_dotenv("../.env")

model = LiteLlm(
    model=os.getenv("KIMI_MODEL"), 
    #model=os.getenv("LOCAL_QWEN30B"), 
)



def before_agent_callback(
    callback_context: CallbackContext
    #toolContext: ToolContext,
    #llm_req:LlmRequest
    
) -> Optional[types.Content]:
    """
    This callback runs before the agent processes a request.
 
    Args:
        callback_context (CallbackContext): Contains state and context information
        toolContex (ToolContext): Contains tool information
        llm_req (LlmRequest):The LLM request being sent

    Returns:
        Optional LlmResponse to override model response
    """
    print(callback_context.state)
    callback_context.state["portfolio"] == {}
    
    return None
    

def add_stock(symbol: str, name: str, num: int, tool_context: ToolContext) -> dict:
    """Add a stock to the user's portfolio.

    Args:
        symbol: The stock symbol to add
        name: The name of the stock
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: add_stock called for '{symbol}: {name}:{num}' ---")
    
    # Get current portfolio from state
    portfolio = tool_context.state.get("portfolio")
    
    #Add the new stock
    portfolio[symbol] ={"name": name, "num": num}
    
    # Update state with the new list of stocks
    tool_context.state["portfolio"] = portfolio
    
    return {
        "action": "add_stock",
        "symbol": symbol,
        "name": name,
        "num": num, 
        "message": f"Added stock: {symbol}",
    }

def view_portfolio(tool_context: ToolContext) -> dict:
    """View all current stocks in the portfolio.

    Args:
        tool_context: Context for accessing session state

    Returns:
        The list of stocks
    """
    print("--- Tool: view_portfolio called ---")
    
    # Get portfolio from state
    portfolio = tool_context.state.get("portfolio")
    
    return {
        "action": "view_portfolio",
        "message": f"Current portfolio: {portfolio}",
    }

def update_portfolio(symbol: str,name:str, num: int, tool_context: ToolContext) -> dict:
    """Add a stock to the portfolio.

    Args:
        symbol: The stock symbol
        num: The number of shares
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: update_portfolio called for '{symbol}' ---")
    
    portfolio = tool_context.state.get("portfolio")
    
    if symbol in portfolio.keys():
        portfolio[symbol] = {"name": name, "num": num}
        tool_context.state["portfolio"] = portfolio
        return {
            "action": "update_portfolio",
            "status":"success",
            "symbol": symbol,
            "num": num,
            "message": f"Updated {symbol} to {num} shares",
        }
    else:
        tool_context.state["portfolio"] = portfolio
        return {
            "action": "update_portfolio",
            "status": "error",
            "message": f"'{symbol}' is not a valid stock symbol.",
        }

def delete_stock(symbol: str, tool_context: ToolContext) -> dict:
    """Delete a stock from the portfolio.

    Args:
        symbol: The stock symbol to delete
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: delete_stock called for symbol {symbol} ---")
    portfolio = tool_context.state.get("portfolio")
    
    if symbol in portfolio.keys():
        del portfolio[symbol]
        tool_context.state["portfolio"] = portfolio
        return {
            "action": "delete_stock",
            "status":"success",
            "symbol": symbol,
            "message": f"delete {symbol} from portfolio",
        }
    else:
        return {
            "action": "delete_portfolio",
            "status": "error",
            "message": f"'{symbol}' is not a valid in portfolio.",
        }

PortfolioAgent = Agent(
    name="portfolio_agent",
    model=model,
    description="A smart agent for managing a portfolio of stocks.",
    instruction="""
    你是一个非常有用的助手可以用来管理你的投资组合.
    
    用户的信息存在state中:
    - portfolio: {portfolio}
    
    你可以通过以下工具来管理你的投资组合:
    1. 添加股票进入投资组合: `add_stock`
    2. 查看投资组合: `view_portfolio`
    3. 更新投资组合: `update_portfolio`
    4. 删除投资组合中的股票: `delete_stock`
    
    """,
    tools=[add_stock,view_portfolio,update_portfolio,delete_stock],
    before_agent_callback=before_agent_callback,
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
)
