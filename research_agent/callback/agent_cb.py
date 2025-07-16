from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.sessions import InMemorySessionService
from typing import Optional
from google.genai import types
from datetime import datetime


def add_state_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    if "portfolio" not in state:
        state["portfolio"] = {}


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """Simple callback to add a timestamp to the message.

    Args:
        callback_context (CallbackContext): Contains state and context information

    Returns:
        Optional[types.Content]: None to continue with the normal execution
    """

    timestamp = datetime.now()
    state = callback_context.state
    state["request_start_time"] = timestamp

    print("[BEFORE_CALLBACK] START AGENT EXECUTION...")

    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Callback function to be executed after the agent has finished executing.

    Args:
        callback_context (CallbackContext): The callback context.
    Returns:
        Optional[types.Content]: The content to be returned to the user.
    """

    state = callback_context.state

    timestamp = datetime.now()
    duration = None
    if "request_start_time" in state:
        duration = (timestamp - state["request_start_time"]).total_seconds()
        
    if duration is not None:
        print(f"Duration: {duration:.2f} seconds")        

    print("[AFTER_CALLBACK] FINISHED AGENT EXECUTION...")

    return None

    # toolContext: ToolContext,
    # llm_req:LlmRequest
