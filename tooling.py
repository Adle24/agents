from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field
from typing import Literal


class WeatherInput(BaseModel):
    """Input for weather queries"""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        description="Units of temperature", default="celsius"
    )
    include_forecast: bool = Field(description="Include 5-day forecast", default=False)


@tool(args_schema=WeatherInput)
def get_weather(
    location: str, units: str = "celsius", include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


@tool
def search_database(query: str, limit: int = 10) -> str:
    """search the customer database for records matching the query.

    args:
        query (str): the query to search for
        limit (int): the number of records to return
    """
    return f"found {limit} records matching the query '{query}'"


@tool("web_search")
def search(query: str) -> str:
    """search the web for information."""
    return f"results for '{query}'"


@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems.",
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))


@tool
def summarize_conversations(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far"""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"
