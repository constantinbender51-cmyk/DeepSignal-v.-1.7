# main.py  (live entry-point)
import os, time, json, pandas as pd, threading
from datetime import datetime, timedelta
from kraken_futures import KrakenFuturesApi
from kraken_ohlc import get_ohlc
from deepseek_signal import get_signal          # shared module
from execute import add_slice

SYMBOL, INTERVAL, SIZE_BTC = "XBTUSD", 60, 0.0001

client = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))

# ---- 10-s trigger babysitter (unchanged) ----
def run_manage():
    from manage import check_all_triggers
    while True:
        check_all_triggers()
        time.sleep(10)

# ---- hourly signal ----
def live_loop():
    while True:
        # wait until next hour + 5 s
        now = time.time()
        next_bar = (int(now) // 3600 + 1) * 3600 + 5
        time.sleep(max(0, next_bar - now))

        df = get_ohlc(SYMBOL, INTERVAL)
        last_50 = (df.tail(50)
                     .reset_index()
                     .assign(time=lambda d: d['time'].astype('int64')//1_000_000_000)
                  ).to_dict(orient="records")

        action, stop, target = get_signal(last_50)   # same func as back-test
        add_slice(action, stop, target)              # logs UID + bracket

if __name__ == "__main__":
    threading.Thread(target=run_manage, daemon=True).start()
    live_loop()
