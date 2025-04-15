import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./env")
with open("API_KEYS.txt") as f:
    for line in f:
        key_value = line.strip().split("=")
        if len(key_value) == 2:
            key, value = key_value
            os.environ[key] = value

    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)


response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": "List 3 distinct differences between deep thinking models and standard LLMs like GPT-4o",
        },
    ],
    model="gpt-4o-mini",
    temperature=1,
    max_tokens=4096,
    top_p=1,
)

print(response.choices[0].message.content)