from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool


model = ChatOllama(model="qwen3:4b", temperature=0)

generate_prompt = """
You are an essay assistant tasked with writing excellent 3-paragraph essays.
Generate the best essay possible for the user's request.
If the user provides critique, respond with a revised version of your
previous attempts
"""

reflection_prompt = """
You are a teacher grading an essay submission. Generate critique and
recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth,
style, etc.
"""


reflection_agent = create_agent(model=model, system_prompt=reflection_prompt)


@tool("subagent1_name", description="subagent1_description")
def call_subagent1(query: str):
    result = reflection_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


agent = create_agent(model=model, system_prompt=generate_prompt, tools=[call_subagent1])
