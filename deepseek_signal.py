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
    # Reduce the input data from 50 to 20 points (use the most recent 20)
    last_20 = last_50[-20:] if len(last_50) >= 20 else last_50
    
    prompt = (
        "You are a crypto strategist. Generate a signal based on OHLC. Last 20 1-h candles:\n"
        f"{json.dumps(last_20)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":<pct>,"target":<pct>}'
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={
            'type': 'json_object'
        }

    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    if not raw:
        return "FLAT", 0.0, 0.0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "FLAT", 0.0, 0.0
    return obj["action"], float(obj["stop"]), float(obj["target"])
