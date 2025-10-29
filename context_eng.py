from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    wrap_model_call,
    ModelResponse,
    SummarizationMiddleware,
)
from typing import Callable
from langchain.tools import tool, ToolRuntime


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if user is authenticated."""
    # Read from State: check current auth status
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    message_count = len(request.messages)
    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra conxise."

    return base


@wrap_model_call
def inject_file_context(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    # Read from State: get uploaded files metadata
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        # Build context about available files
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        # Inject file context before recent messages
        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)


agent = create_agent(
    model="ollama:qwen3:4b",
    middleware=[
        state_aware_prompt,
        inject_file_context,
        SummarizationMiddleware(
            model="ollama:qwen3:4b",
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        ),
    ],
    tools=[check_authentication],
)
