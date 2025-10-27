from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )


agent = create_agent(
    model="ollama:qwen3:4b",
    response_format=ToolStrategy(
        schema=ProductReview,
        tool_message_content="Action item captured and added to meeting notes!",
        handle_errors="Please provide a valid rating between 1-5 and include a comment.",
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
            }
        ]
    }
)

print(result["structured_response"])
