from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.sessions import InMemorySessionService
from typing import Optional
from google.genai import types
from datetime import datetime

def before_model_cb(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Callback function to be called before model call.

    Args:
        callback_context (CallbackContext): callback_context
        llm_request (LlmRequest): LLM Request info

    Returns:
        Optional[LlmResponse]: Response from LLM
    """
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    # Inspect the last user message in the request contents
    for content in llm_request.contents:
        print(f"[Callback Before model for {agent_name}]Role: {content.role}")
        part_cnt = 0
        for p in content.parts:
            print(f"[Callback Before model for {agent_name}]Part[{part_cnt}]: {p}")
            part_cnt += 1
    return None

def after_model_cb(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Callback function that is called after the model is called.

    Args:
        callback_context (CallbackContext): callback context
        llm_response (LlmResponse): LLM response

    Returns:
        Optional[LlmResponse]: Response to be returned to the user
        
    """
    agent_name = callback_context.agent_name
    print(f"[Callback] After model call for agent: {agent_name}")

    # Skip processing if response is empty or has no text content
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return None
    
    # Inspect the last user message in the request contents
    for part in llm_response.content.parts:
        part_cnt = 0
        if hasattr(part, "text") and part.text:
            print(f"[Callback After model for {agent_name}]Part[{part_cnt}]: {part.text}")
            part_cnt += 1
    return None

