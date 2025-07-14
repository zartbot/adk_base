import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)


from portfolio_agent.agent import root_agent as portfolio_agent

initial_state = {"portfolio": {}}

#创建Session DB
session_service = InMemorySessionService()


# 定义使用的APP名称, 用户名, 此处为了简化采用了固定的Session ID
APP_NAME = "portfolio_agent"
USER_ID = "zartbot"
SESSION_ID = "session_001"  

# --- Runner --- 
runner = Runner(
    agent=portfolio_agent,  # 需要执行的Agent
    app_name=APP_NAME,  # 关联的APP名称
    session_service=session_service,  # 使用的内存Session DB
)
print(f"Runner created for agent '{runner.agent.name}'.")

async def call_agent_async(query: str, runner, user_id, session_id):

    #创建Session
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID,
        state=initial_state,
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


async def run_conversation():
   # await call_agent_async(
   #     "添加1000股寒武纪(股票代码:688256), 和800股平安银行(股票代码:000001)进入投资组合, 然后再将平安银行减仓400股, 最后查看当前的投资组合", runner=runner, user_id=USER_ID, session_id=SESSION_ID
   # )
   await call_agent_async(
       "查看当前投资组合", runner=runner, user_id=USER_ID, session_id=SESSION_ID
   )
    

if __name__ == "__main__":
    asyncio.run(run_conversation())
