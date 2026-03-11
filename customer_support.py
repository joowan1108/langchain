from langchain.chat_models import init_chat_model
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
import uuid

from typing import Callable
from typing_extensions import NotRequired
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

model = init_chat_model('google_genai:gemini-2.5-flash')


"""
state machine pattern: task을 하는 도중에 agent 행동 방식을 dynamic하게 바꾸는 방법

고객 문의 -> 기기 보험 들었는지 여부 체크 o -> 문의 종류 확인 -> hardware -> human paid repair options
                                                       -> software -> trouble shooting
                                     x -> 문의 종류 확인 -> hardware -> warranty repair instructions
                                                       -> software -> trouble shooting
"""

SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """State for customer support workflow"""
    #Not required는 첫 대화 상태에서 이 정보가 없어도 되도록 한 것이다,
    current_step : NotRequired[SupportStep] #이 field가 각 turn에 어떤 prompt와 tool을 쓸 것인지 결정함
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    #Command는 agent framework에 지시를 해주는 역할을 한다.
    return Command(
        #warranty_status와 current_step을 update하도록 한다
        update={
            "messages": [
                #호출한 도구가 agent workflow에서 agent에게 return하는 message 정하기
                #원래는 정의 안 해도 되지만 이 workflow에서는 하나의 agent의 state을 dynamic하게 바꾸면서 작동하도록 하였기 때문에 Command을 써야 하는데 Command을 쓰면 tool의 return값을
                #직접 알려줘야 한다. 이때, toolmessage을 정의하지 않는다면 agent는 tool의 return 값을 알지 못함
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id = runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        },
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str)-> str:
    """human specialist가 필요할 때"""
    return f"because of {reason}, escalating to human support"

@tool
def provide_solution(solution: str) -> str:
    """고객 문의 사항에 해결책 제공"""
    return f"Solution provided: {solution}"


#각 step에 필요한 prompt나 tool들을 dictionary로 정의: (prompt, tools, required state)

# Define prompts as constants for easy reference
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

Be specific and helpful in your solutions."""

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}


# AI에게 질문을 던지기 직전의 상태(request)와, 실제로 AI를 호출하는 스위치(handler)을 받아 agent와 모델이 주고받는 순간을 control하는 함수를 만들 때 @wrap_model_call annotations을 붙임
#handler는 모델을 호출하는 역할, request는 ai 모델에 질문을 전달하기 직전의 상태를 의미함
@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step"""
    current_step = request.state.get("current_step", "warranty_collector")

    #ai 모델 호출 이전 state의 configurations을 가져옴
    stage_config = STEP_CONFIG[current_step]

    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    system_prompt = stage_config["prompt"].format(**request.state) #request state에 있는 key 값이 prompt에 필요한 변수 값에 들어가도록

    #모델에게 현재 상황에 맞는 prompt와 tools dynamic하게 정의
    request = request.override(
        system_prompt = system_prompt,
        tools = stage_config["tools"],
    )
    #model에 전달
    return handler(request)

# Collect all tools from all step configurations
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]


# Agent 정의
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState, #state의 구조 미리 설정
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(), #current_step을 계속 파악하기 위해
)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 새로 추가: 지금까지 출력한 메시지의 총 개수를 추적하는 변수
printed_msg_count = 0

# ==========================================
# Turn 1: Initial message - starts with warranty_collector step
# ==========================================
print("\n=== Turn 1: Warranty Collection ===")
result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone screen is cracked")]},
    config
)

# 이전 개수(printed_msg_count)부터 끝까지만 슬라이싱하여 새로운 메시지만 출력
for msg in result['messages'][printed_msg_count:]:
    msg.pretty_print()

# 현재까지 누적된 메시지 개수로 업데이트
printed_msg_count = len(result['messages'])


# ==========================================
# Turn 2: User responds about warranty
# ==========================================
print("\n=== Turn 2: Warranty Response ===")
result = agent.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty")]},
    config
)

for msg in result['messages'][printed_msg_count:]:
    msg.pretty_print()

printed_msg_count = len(result['messages'])
print(f"Current step: {result.get('current_step')}")


# ==========================================
# Turn 3: User describes the issue
# ==========================================
print("\n=== Turn 3: Issue Description ===")
result = agent.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
    config
)

for msg in result['messages'][printed_msg_count:]:
    msg.pretty_print()

printed_msg_count = len(result['messages'])
print(f"Current step: {result.get('current_step')}")


# ==========================================
# Turn 4: Resolution
# ==========================================
print("\n=== Turn 4: Resolution ===")
result = agent.invoke(
    {"messages": [HumanMessage("What should I do?")]},
    config
)

for msg in result['messages'][printed_msg_count:]:
    msg.pretty_print()

printed_msg_count = len(result['messages'])
print(f"Current step: {result.get('current_step')}")

#https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support