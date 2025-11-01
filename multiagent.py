from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState


class CustomState(AgentState):
    example_state_key: str


sub_agent = create_agent(model="ollama:qwen3:4b")


@tool("sub_agent", description="sub agent")
def call_sub_agent(query: str, runtime: ToolRuntime[None, CustomState]):
    result = sub_agent.invoke(
        {
            "messages": [{"role": "user", "content": query}],
            "example_state_key": runtime.state["example_state_key"],
        }
    )

    return result["messages"][-1].content


agent = create_agent(model="ollama:qwen3:4b", tools=[call_sub_agent])
