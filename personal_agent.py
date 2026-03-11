from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import os
from langchain.tools import tool
import datetime


from dotenv import load_dotenv
load_dotenv()


model_name = os.getenv("MODEL_NAME")
model = init_chat_model(model_name)

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
    location: str=""
) -> str:
    """Calendar event 생성, ISO datetime format"""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send email via email api. Requires properly formatted addresses"""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


'''
Calendar 시간 비어있는지 확인
'''
@tool 
def get_available_time_slots(
    attendees: list[str],
    date:str,
    duration_minutes: int
) -> list[str]:
    """특정 시간에 attendees 캘린더가 비어있는지 확인"""
    return ["09:00", "14:00", "16:00"]


"""CALENDAR"""
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

CALENDAR_AGENT_PROMPT = (
    f"Current Date: {current_date}. "
    "You are a calendar scheduling assistant. "
    "When a user provides a date, resolve it to an ISO format using the Current Date as a reference. "
    "If the user's request is clear (e.g., 'March 13 at 7PM'), do not ask for confirmation; "
    "immediately call the appropriate tool (create_calendar_event or get_available_time_slots)."
)
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

# query = "Schedule a team meeting with john@example.com at 2026/03/13 at 2PM for 1 hour"
# for step in calendar_agent.stream(
#     {"messages": [{"role": "user", "content": query}]}
# ):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()




"""EMAIL"""
EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. You have to write the subject and body according to the request "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)
# query = "Send the design team a reminder about reviewing the new mockups. Email is design@gmail with subject REMINDER REVIEW MOCKUPS."

# for step in email_agent.stream(
#     {"messages": [{"role": "user", "content": query}]}
# ):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()


"""Wrap sub-agents: Supervisor가 각 sub-agent을 필요에 따라 invoke할 수 있도록 하는 과정"""

@tool
def schedule_event(request: str) -> str:
    """
    Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


"""Supervisor Agent"""
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model,
    tools = [schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)
query = (
    "Schedule a team standup with john@example.com at 2026/03/13 at 2PM for 1 hour, "
    "and send them an email reminder about reviewing the new mockups. The email address is design@gmail.com."
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()