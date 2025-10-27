from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Any
from langgraph.runtime import Runtime


class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I cannot process requests containing inappropriate content. Please rephrase your request.",
                        }
                    ],
                    "jump_to": "end",
                }

        return None


agent = create_agent(
    model="ollama:qwen3:4b",
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware(
            "api-key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True, "delete_databa": True, "search": False}
        ),
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My email is john.doe@example.com and card is 4532-1234-5678-9010",
            }
        ],
    },
    config=config,
)

result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,  # Same thread ID to resume the paused conversation
)
