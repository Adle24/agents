import pandas as pd
from ollama import chat
from ollama import ChatResponse


prompt_a = """Product description: A pair of shoes that can
fit any foot size.
Seed words: adaptable, fit, omni-fit.
Product names:"""

prompt_b = """Product description: A home milkshake maker.
Seed words: fast, healthy, compact.
Product names: HomeShaker, Fit Shaker, QuickShake, Shake
Maker
Product description: A watch that can tell accurate time in
space.
Seed words: astronaut, space-hardened, eliptical orbit
Product names: AstroTime, SpaceGuard, Orbit-Accurate,
EliptoTime.
Product description: A pair of shoes that can fit any foot
size.
Seed words: adaptable, fit, omni-fit.
Product names:"""

test_prompts = [prompt_a, prompt_b]


def get_response(prompt: str):
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

    return response["message"]["content"]


responses = []
num_tests = 5

for idx, prompt in enumerate(test_prompts):
    val_name = chr(ord("A") + idx)

    for i in range(num_tests):
        response = get_response(prompt)
        data = {
            "variant": val_name,
            "prompt": prompt,
            "response": response,
        }

        responses.append(data)


df = pd.DataFrame(responses)
df.to_csv("responses.csv", index=False)
