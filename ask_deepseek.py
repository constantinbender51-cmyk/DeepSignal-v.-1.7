from openai import OpenAI   # 1.0+ interface

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"   # DeepSeek is OpenAI-compatible
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "what is the colour of love?"}],
    temperature=0.7
)

print(completion.choices[0].message.content)
