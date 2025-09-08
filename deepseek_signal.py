# deepseek_signal.py
import json
import openai
from typing import Tuple
import os 

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def get_signal(last_50: list) -> Tuple[str, float, float]:
    prompt = (
        "You are a crypto strategist. Request indicators in the reason key. Last 50 1-h candles:\n"
        f"{json.dumps(last_50)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":<pct>,"target":<pct>,"reason":<str>}'
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    if not raw:
        return "FLAT", 0.0, 0.0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "FLAT", 0.0, 0.0
    return obj["action"], float(obj["stop"]), float(obj["target"]), obj["reason"]
