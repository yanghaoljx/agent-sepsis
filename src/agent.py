import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AnyMessage, HumanMessage
import operator
from src.validators.agent_validators import *
from src.agent_tools import  retrieve_faq_info, sepsis_early_screening, get_patient_basic_info,predict_sepsis_prognosis_mortality
from datetime import datetime
from src.utils import get_model
import logging


logger = logging.getLogger(__name__)

class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


tools = [retrieve_faq_info,sepsis_early_screening,get_patient_basic_info,predict_sepsis_prognosis_mortality]

tool_node = ToolNode(tools)


model = get_model()
model = model.bind_tools(tools = tools)

def should_continue(state: MessagesState) -> Literal["tools", "human_feedback"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "human_feedback"

#The commented part is because it breaks the UI with the input function
def should_continue_with_feedback(state: MessagesState) -> Literal["agent", "end"]:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, dict):
        if last_message.get("type","") == 'human':
            return "agent"
    if (isinstance(last_message, HumanMessage)):
        return "agent"
    return "end"


def call_model(state: MessagesState):
    # --- 关键修改：System Prompt ---
    # 明确指示 ID 提取规则和工具路由逻辑
    system_prompt = f"""
    You are a clinical decision support assistant specialized in sepsis.
    
    Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M, %A')}

    YOUR TASKS:
    1. **Identify Patient ID**: Always look for a patient identifier in the user's query. 
       - Format: Typically starts with "IP" followed by numbers (e.g., "IP2333", "IP9981").
       - Action: Extract this string exactly to use as the `patient_id` argument for tools.

    2. **Tool Routing**:
       - If user asks about **General Situation/Info** (e.g., "How is IP2333?", "Patient info") -> Call `get_patient_basic_info`.
       - If user asks about **Early Screening/Current Status** (e.g., "Risk of sepsis?", "Screening result") -> Call `sepsis_early_screening`.
       - If user asks about **Prognosis/Future Trend** (e.g., "What is the outcome?", "Predict risk") -> Call `predict_sepsis_prognosis_mortality`.
       - If user asks generic hospital questions (parking, hours) -> Call `retrieve_faq_info` (if available).

    3. **Response Style**:
       - Be professional, concise, and clinically objective.
       - Do NOT make up patient info. If ID is missing, ask for it.
       - Summarize the tool output clearly for the doctor.
       - Use Markdown tables for basic info and abnormal indicators.
        - Use specific emojis for status: 🔴 (High), 🔵 (Low), ⚠️ (Critical).
    """

    # 保持历史消息
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def read_human_feedback(state: MessagesState):
    pass


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_feedback", read_human_feedback)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"human_feedback": 'human_feedback',
    "tools": "tools"}
)
workflow.add_conditional_edges(
    "human_feedback",
    should_continue_with_feedback,
    {"agent": 'agent',
    "end": END}
)
workflow.add_edge("tools", 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer, 
                       interrupt_before=['human_feedback'])

if __name__ == '__main__':
    while True:
        question = input("\nUser (输入 'exit' 退出): ")
        if question.lower() == 'exit':
            break

        # 启动流
        events = app.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "42"}},
            stream_mode="values" # 改用 values 模式更容易观察消息链
        )

        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                # 只打印助手（且不是在请求工具）的最终回复
                if isinstance(last_msg, (AnyMessage)) and last_msg.type == "assistant":
                    if last_msg.content:
                        print(f"\nAI: {last_msg.content}")
                    if last_msg.tool_calls:
                        print(f"  [正在调用工具: {last_msg.tool_calls[0]['name']}...]")
