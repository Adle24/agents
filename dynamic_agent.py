from langchain.agents import create_agent, AgentState
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    wrap_tool_call,
    dynamic_prompt,
    AgentMiddleware,
)
from langchain.tools import tool
from typing import TypedDict
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy


basic_model = ChatOllama(model="llama3.2:3b", temperature=0)
advanced_model = ChatOllama(model="qwen3:4b", temperature=0)


class CustomState(AgentState):
    user_preferences: str


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState


class Context(TypedDict):
    user_role: str


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


@tool
def search(query: str) -> str:
    """search for information"""
    return f"results for: {query}"


@tool
def get_weather(location: str) -> str:
    """get weather information for location"""
    return f"weather for: {location}, Sunny 32C"


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


@wrap_tool_call
def handle_tool_errors(request, handler):
    """handle tool execution errors with custom messages"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"tool error: {e}", tool_call_id=request.tool_call_id["id"]
        )


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """choose model based on conversation complexity"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection, user_role_prompt, CustomMiddleware()],
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    response_format=ToolStrategy(ContactInfo),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567",
            }
        ]
    },
    context={"user_role": "expert"},
)

print(result["structured_response"])
