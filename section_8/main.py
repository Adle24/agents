from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware


class Joke(BaseModel):
    setup: str = Field(description="the setup of the joke")
    punchline: str = Field(description="the punchline to the joke")


agent = create_agent(
    model="ollama:qwen3:4b",
    response_format=Joke,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval, editing, or rejection for sending emails
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # Auto-approve reading emails
                "read_email_tool": False,
            }
        ),
    ],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me a joke about cats"}]}
)

print(result["structured_response"])

for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What is the best sites to practice ML"}
        ]
    },
    stream_mode="values",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
