import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types  

from stateful_multi_agent.agent import root_agent as multi_agent

initial_state = {"portfolio": {}}

#创建Session DB
db_url = "sqlite:///./portfolio.db"
session_service = DatabaseSessionService(db_url=db_url)


# 定义使用的APP名称, 用户名
APP_NAME = "stateful_multi_agent"
USER_ID = "zartbot"

# --- Runner --- 
runner = Runner(
    agent=multi_agent,  # 需要执行的Agent
    app_name=APP_NAME,  # 关联的APP名称
    session_service=session_service,
)
print(f"Runner created for agent '{runner.agent.name}'.")

async def call_agent_async(query: str, runner, user_id):

    #Session查询
    existing_session =await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    if existing_session and len(existing_session.sessions) > 0 :
        SESSION_ID = existing_session.sessions[0].id
        print(f"Continuing existing session: {SESSION_ID}")
    else:   
        new_session = await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            state=initial_state,
        )
        SESSION_ID = new_session.id
        print(
            f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'"
        )
    

    #打印需要执行的Query
    print(f"\n>>> User Query: {query}")


    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."  # 默认回复

    #读取任务过程中的Event
    async for event in runner.run_async(
        user_id=user_id, session_id=SESSION_ID, new_message=content
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
   while True:
       user_input = input("You: ")
       if user_input.lower() in ["exit", "quit"]:
           print("Ending conversation.")
           break
       
       await call_agent_async(user_input,runner=runner, user_id=USER_ID)

if __name__ == "__main__":
    asyncio.run(run_conversation())
