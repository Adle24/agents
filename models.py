from langchain.chat_models import init_chat_model


# model configuration
model = init_chat_model(model="ollama:qwen3:4b", temperature=0.7)


# model invocation
response = model.invoke("Are you a large language model?")
print(response.content)

# model streaming
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)

# batching
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ]
)

for response in responses:
    print(response.content)

# multimodal model
multimodal_model = init_chat_model(model="ollama:gemma3:4b", temperature=0.7)
response = multimodal_model.invoke("Create a picture of a cat")
print(response.content_blocks)

# reasoning models
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
