#!/usr/bin/env python3
import os
import json
import pandas as pd
from openai import OpenAI
from kraken_futures import KrakenFuturesApi
from kraken_ohlc import get_ohlc

SYMBOL   = "XBTUSD"        # spot symbol (use pf_xbtusd for futures)
INTERVAL = 60              # 1 h
SIZE_BTC = 0.0001          # micro size

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1")
kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))

def deepseek_signal(df: pd.DataFrame) -> tuple[str, float, float]:
    """Return (action, stop%, target%)."""
    last_50 = (df.tail(50)
                 .reset_index()
                 .assign(time=lambda d: d['time'].astype('int64')//1_000_000_000)
              ).to_dict(orient="records")

    prompt = (
        f"You are a crypto strategist.  Last 50 1-h candles:\n{json.dumps(last_50)}\n"
        "Reply **strictly** in this format (no extra text):\n"
        "ACTION,BUY|SELL|FLAT;STOP,-1;TARGET,+2\n"
        "Example: ACTION,BUY;STOP,-1;TARGET,+2"
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw   = resp.choices[0].message.content.strip()
    parts = {k: v for k, v in (x.split(",") for x in raw.split(";"))}
    action = parts["ACTION"]
    stop   = float(parts["STOP"])
    target = float(parts["TARGET"])
    return action, stop, target
    
if __name__ == "__main__":
    df = get_ohlc(SYMBOL, INTERVAL)
    signal, stop, target = deepseek_signal(df)
    from execute import execute_trade
    execute_trade(signal, SIZE_BTC)

