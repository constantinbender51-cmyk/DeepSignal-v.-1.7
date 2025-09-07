# backtest.py
import pandas as pd
import json
import random
import os
from datetime import timedelta
from deepseek_signal import get_signal

CSV_PATH = "xbtusd_1h_8y.csv"   # 70k candles
OUT_PATH = "backtest_trades.csv"
SIZE_BTC = 0.0001
FEE      = 0.0007          # 0.02 % maker + 0.05 % taker + buffer
SLIP     = 0.0001          # 1 bp slippage

def bar_exit(bar: pd.Series, side: str, stop: float, target: float, entry_px: float):
    """Return exit_px, exit_time, hit_stop bool within *same* bar."""
    stop_px  = entry_px * (1 + stop / 100) if side == "long" else entry_px * (1 - stop / 100)
    tgt_px   = entry_px * (1 + target / 100) if side == "long" else entry_px * (1 - target / 100)

    if side == "long":
        if bar["low"] <= stop_px:
            return stop_px - SLIP, bar["time"], True
        if bar["high"] >= tgt_px:
            return tgt_px + SLIP, bar["time"], False
    else:  # short
        if bar["high"] >= stop_px:
            return stop_px + SLIP, bar["time"], True
        if bar["low"] <= tgt_px:
            return tgt_px - SLIP, bar["time"], False
    return None, None, None  # still open

def run():
    df = pd.read_csv(CSV_PATH, parse_dates=["open_time"])
    df = df.rename(columns={"open_time": "time"})   # unify name
    candles = df[["time", "open", "high", "low", "close", "volume"]].to_dict("records")

    random.shuffle(candles)

    trades = []
    for i, cand in enumerate(candles):
        if i < 50:  # warm-up
            continue
        last_50 = [dict(time=c["time"].timestamp(), o=c["open"], h=c["high"],
                        l=c["low"], c=c["close"], v=c["volume"])
                   for c in candles[i-50:i]]
        action, stop, target = get_signal(last_50)
        if action == "FLAT":
            continue

        entry_px = cand["open"] * (1 + 5/3600/100)  # 5 sec slippage
        exit_px, exit_time, hit_stop = bar_exit(cand, action.lower(), stop, target, entry_px)

        if exit_px:  # closed same bar
            pnl_pct = (exit_px - entry_px) / entry_px * (1 if action == "BUY" else -1) - FEE
            trades.append({
                "open_time": cand["time"], "close_time": exit_time,
                "side": action, "entry": entry_px, "exit": exit_px,
                "stop": stop, "target": target, "hit_stop": hit_stop,
                "pnl_pct": pnl_pct
            })
        if len(trades) >= 1000:
            break

    pd.DataFrame(trades).to_csv(OUT_PATH, index=False)
    print("Back-test done:", OUT_PATH)

if __name__ == "__main__":
    run()
