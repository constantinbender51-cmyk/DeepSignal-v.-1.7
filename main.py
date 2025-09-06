#!/usr/bin/env python3
import os, json, pandas as pd, time
from openai import OpenAI
from kraken_futures import KrakenFuturesApi
from kraken_ohlc import get_ohlc
from execute import add_slice
import threading

# ---------- config ----------
SYMBOL   = "XBTUSD"
INTERVAL = 60
SIZE_BTC = 0.0001
# ----------------------------

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1")
kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))

# ---- signal generation ----
def deepseek_signal(df: pd.DataFrame) -> tuple[str, float, float]:
    last_50 = (df.tail(50)
                 .reset_index()
                 .assign(time=lambda d: d['time'].astype('int64')//1_000_000_000)
              ).to_dict(orient="records")
    prompt = (
        f"You are a crypto strategist.  Last 50 1-h candles:\n{json.dumps(last_50)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":-1,"target":2}'
    )
    resp = client.chat.completions.create(model="deepseek-chat",
                                          messages=[{"role": "user", "content": prompt}],
                                          temperature=0)
    obj = json.loads(resp.choices[0].message.content.strip())
    return obj["action"], float(obj["stop"]), float(obj["target"])

# ---- 10-s position babysitter ----
def run_manage():
    from manage import check_all_triggers          # local import to avoid circular
    while True:
        check_all_triggers()
        time.sleep(10)

# ---- main loop: hourly signal + a few seconds ----
if __name__ == "__main__":
    threading.Thread(target=run_manage, daemon=True).start()

    while True:
        now = time.time()
        # wait until next hour + 5 s
        next_hour = (int(now) // 3600 + 1) * 3600 + 5
        time.sleep(max(0, next_hour - now))

        df = get_ohlc(SYMBOL, INTERVAL)
        signal, stop, target = deepseek_signal(df)
        add_slice(signal, stop, target)
      
