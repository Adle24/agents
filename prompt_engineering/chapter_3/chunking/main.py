with open("hubspot_blog_post.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = [text[i : i + 200] for i in range(0, len(text), 200)]


def sliding_window(text_to_chunk, window_size, step_size):
    if window_size > len(text_to_chunk) or step_size < 1:
        return []

    return [
        text[i : i + window_size]
        for i in range(0, len(text) - window_size + 1, step_size)
    ]


text = "This is an example of sliding window text chunking."
window_size = 20
step_size = 5
chunks = sliding_window(text, window_size, step_size)
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk}")
