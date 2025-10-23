from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage


model = init_chat_model("ollama:qwen3:4b")
system_message = SystemMessage("You are a helpful assistnat. Answer in English.")
human_message = HumanMessage("Hello, who are you?")

messages = [system_message, human_message]
response = model.invoke(messages)

print(type(response))
print(response.usage_metadata)
