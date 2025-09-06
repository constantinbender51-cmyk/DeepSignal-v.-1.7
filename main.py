#!/usr/bin/env python3
import os
import json
from openai import OpenAI
from kraken_futures import KrakenFuturesApi  # auth + private trade
from kraken_ohlc import get_ohlc            # reuse existing fetcher

SYMBOL   = "XBTUSD"        # spot symbol (use pf_xbtusd for futures)
INTERVAL = 60              # 1 h
SIZE_BTC = 0.0001          # micro size

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1")
kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))

def deepseek_signal(df: pd.DataFrame) -> str:
    """Return BUY, SELL, or FLAT."""
    last_50 = df.tail(50).reset_index().to_dict(orient="records")
    prompt = (f"You are a crypto strategist.  Last 50 1-h candles:\n"
              f"{json.dumps(last_50)}\n"
              f"Reply ONLY the word BUY, SELL, or FLAT for next hour.")
    resp = client.chat.completions.create(model="deepseek-chat",
                                          messages=[{"role": "user", "content": prompt}],
                                          temperature=0)
    return resp.choices[0].message.content.strip().upper()

if __name__ == "__main__":
    import pandas as pd
    import time

    df = get_ohlc(SYMBOL, INTERVAL)
    signal = deepseek_signal(df)

    from execute import execute_trade
    execute_trade(signal, SIZE_BTC)
