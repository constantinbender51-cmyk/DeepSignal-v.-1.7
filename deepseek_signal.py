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
        "You are a crypto strategist. If you can identify a trading signal in either direction, buy or sell and set stop/target to appropriate levels according to sound risk management practices. If unsure, FLAT, 0.0, 0.0, <string>. If no stop or target has been triggered after 24 hours, the trade will be automatically closed. Last 50 1-h candles:\n"
        f"{json.dumps(last_50)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":<percent>,"target":<percent>,"reason":<string>}'
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
    if obj["stop"] is None or obj["target"] is None:
        print("nontype response",raw);
        
    return obj["action"], float(obj["stop"]), float(obj["target"]), obj["reason"]
