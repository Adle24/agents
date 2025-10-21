from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    justification: str


model = ChatOllama(model="qwen3:4b", temperature=0)
agent = create_agent(
    model,
    system_prompt="You are a helpful assistant that responds to questions with three exclamation marks.",
    response_format=ToolStrategy(Answer),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What weighs more, a pound of bricks or a pound of feathers?",
            }
        ]
    }
)

# for chunk in agent.stream({
#     "messages": [
#         {
#             "role": "user",
#             "content": "Search for AI news and summarize the findings"
#         }
#     ]
# }, stream_mode="values"):
#     latest_message = chunk["messages"][-1]
#
#     if latest_message.content:
#         print(f"agent: {latest_message.content}")
