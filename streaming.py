from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get a weather for a given city"""
    return f"It's always sunny in {city}"


agent = create_agent("ollama:qwen3:4b", tools=[get_weather])

for chunk in agent.stream(
{"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates"
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")

