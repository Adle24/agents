from ollama import chat
from ollama import ChatResponse

prompt = """
Given the statement, classify it as either "Compliment", "Complaint", or
"Neutral":
1. "The sun is shining." - Neutral
2. "Your support team is fantastic!" - Compliment
3. "I had a terrible experience with your software." - Complaint
You must follow the following principles:
- Only return the single classification word. The response should be either
"Compliment", "Complaint", or "Neutral".
- Perform the classification on the text enclosed within ''' delimiters.
'''{content}'''
Classification:
"""

response: ChatResponse = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
)
