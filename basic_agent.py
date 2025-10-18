from langchain.agents import create_agent
from langchain_ollama import ChatOllama


model = ChatOllama(model="gemma3:4b", temperature=0)


agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Almaty"}]}
)

print(repr(result))
