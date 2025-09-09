# deepseek_signal.py
import json
import openai
from typing import Tuple
import os 

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# deepseek_signal.py  (only the prompt-building part changes)
def get_signal(last_50: list) -> Tuple[str, float, float, str]:
    # ---- 1.  compute sma20 and sma50 on the last 50 closes ----------
    closes = [c["c"] for c in last_50]          # close prices
    sma20  = sum(closes[-20:]) / 20
    sma50  = sum(closes) / 50

    # ---- 2.  0.1 % proximity filter ---------------------------------
    if abs(sma20 - sma50) / sma50 > 0.001:    # NOT within 0.1 %
        return "FLAT", 0.0, 0.0, "preselect"  # skip LLM

    # ---- 3.  otherwise ask DeepSeek ---------------------------------
    prompt = (
        "You are a crypto strategist. If you can identify a trading signal in either direction, "
        "buy or sell and set stop/target to appropriate levels according to sound risk-management "
        "practices. If unsure, reply FLAT, 0.0, 0.0, <string>. If no stop or target has been "
        "triggered after 24 hours, the trade will be automatically closed. "
        "Last 50 1-h candles:\n"
        f"{json.dumps(last_50)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":<percent>,"target":<percent>,"reason":<string>}'
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        response_format={"type": "json_object"}
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    if not raw:
        return "FLAT", 0.0, 0.0, "empty_llm"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "FLAT", 0.0, 0.0, "bad_json"

    return obj["action"], float(obj["stop"]), float(obj["target"]), obj["reason"]
