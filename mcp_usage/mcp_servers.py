from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"],
        },
        "weather": {
            "transport": "streamable_http",  # HTTP-based remote server
            "url": "http://localhost:8000/mcp",
        },
    }
)

tools = client.get_tools()

agent = create_agent(
    "ollama:qwen3:4b",
    tools=tools,
)

math_response = agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)
weather_response = agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
