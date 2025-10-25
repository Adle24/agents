from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    before_model,
    after_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    AgentMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any, Callable
from typing_extensions import NotRequired


# custom decorator-based middleware
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"about to call model with {len(state['messages'])} messages")
    return None


@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]

    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end",
        }

    return None


@wrap_model_call
def retry_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse | None:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")


@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}. Be concise and friendly."


# class-base middleware
class LoggingMiddleware(AgentMiddleware):
    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)

        if count > 10:
            return {"jumt_to": "end"}

        return None

    def after_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return {
            "model_call_count": state.get("model_call_count", 0) + 1,
        }


global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

agent = create_agent(
    model="ollama:qwen3:4b",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="ollama:qwen3:4b",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
            summary_prompt="Custom prompt for summarization",
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        ),
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
        ModelFallbackMiddleware("ollama:llama3.2:3b"),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        LLMToolSelectorMiddleware(
            model="ollama:qwen3:4b", max_tools=3, always_include=["search"]
        ),
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=50.0,
            jitter=True,
        ),
    ],
)
