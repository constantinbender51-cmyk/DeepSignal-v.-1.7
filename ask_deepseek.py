#!/usr/bin/env python3
import os, openai
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com/v1"
with open("deepseek_prompt.txt") as f:
    question = f.read().strip()
r = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": question}],
    temperature=0.7
)
print(r.choices[0].message.content)
