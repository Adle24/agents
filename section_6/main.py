import ast
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def calculator(query: str) -> str:
    """ "A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()

tools = [calculator, search]
model = ChatOllama(model="qwen3:4b", temperature=0)

agent = create_agent(model=model, tools=tools)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How old was the 30th president of the United States when he died?",
            }
        ]
    }
)

print(response["messages"])
