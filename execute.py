#!/usr/bin/env python3
from kraken_futures import KrakenFuturesApi
import os

kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))

def execute_trade(side: str, size_btc: float):
    """side: BUY or SELL (case-insensitive)."""
    side = side.lower()
    if side not in {"buy", "sell"}:
        print("No trade – signal was FLAT or garbage.")
        return

    params = {
        "orderType": "mkt",
        "symbol": "pf_xbtusd",
        "side": side,
        "size": size_btc,
    }
    print("Sending", side.upper(), "for", size_btc, "BTC")
    resp = kraken.send_order(params)
    print("Kraken reply:", resp)

if __name__ == "__main__":
    execute_trade("BUY", 0.0001)  # quick sanity check
