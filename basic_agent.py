from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy

from dataclasses import dataclass

load_dotenv()

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@dataclass
class ResponseFormat:
    """Response schema for the agent"""
    response:str
    weather_conditions : str | None = None

@tool
def get_weather_for_location(city:str) -> str:
    """Get weather for a given city"""
    return f"It is sunny in {city}"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Korea" if user_id == "1" else "JP"

config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model = "google_genai:gemini-3-flash-preview",
    system_prompt = SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])


