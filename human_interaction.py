from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


agent = create_agent(
    model="ollama:qwen3:4b",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"write_file": True, "read_file": False})
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "some_id"}}
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config,
)

agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ),
    config=config,  # Same thread ID to resume the paused conversation
)
