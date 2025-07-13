import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  

from basic_agent.agent import root_agent as basic_agent

#创建Session DB
session_service = InMemorySessionService()


# 定义使用的APP名称, 用户名, 此处为了简化采用了固定的Session ID
APP_NAME = "basic_agent"
USER_ID = "zartbot"
SESSION_ID = "session_001"  

# --- Runner --- 
runner = Runner(
    agent=basic_agent,  # 需要执行的Agent
    app_name=APP_NAME,  # 关联的APP名称
    session_service=session_service,  # 使用的内存Session DB
)
print(f"Runner created for agent '{runner.agent.name}'.")

async def call_agent_async(query: str, runner, user_id, session_id):

    #创建Session
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(
        f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'"
    )

    #打印需要执行的Query
    print(f"\n>>> User Query: {query}")


    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."  # 默认回复

    #读取任务过程中的Event
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):

       #查看Function Call的请求和回复事件
        calls = event.get_function_calls()
        if calls:
            for call in calls:
                tool_name = call.name
                arguments = call.args 
                print(f"  Tool: {tool_name}, Args: {arguments}")
        responses = event.get_function_responses()
        if responses:
            for response in responses:
                tool_name = response.name
                result_dict = response.response 
                print(f"  Tool Result: {tool_name} -> {result_dict}")
               # session.add_tool_response(tool_name, result_dict)
                await session_service.append_event(session,event)
        
        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif (
                event.actions and event.actions.escalate
            ):  # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found
           
    print(f"<<< Agent Response: {final_response_text}")
    for ev in session.events:
        print(f"---->Events (`events`):         {ev}") # Initially empty



async def run_conversation():
    await call_agent_async(
        "查询当前的时间", runner=runner, user_id=USER_ID, session_id=SESSION_ID
    )


# Execute the conversation using await in an async context (like Colab/Jupyter)

if __name__ == "__main__":
    asyncio.run(run_conversation())
