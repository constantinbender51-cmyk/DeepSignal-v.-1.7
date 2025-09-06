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

def place_bracket(action: str, size: float, stop: float, target: float):
    if action.upper() not in {"BUY","SELL"}:
        print("No trade – FLAT or garbage.")
        return
    # 1) market entry
    order = kraken.send_order({
        "orderType": "mkt",
        "symbol": "pf_xbtusd",
        "side": action.lower(),
        "size": size,
    })
    if order.get("sendStatus", {}).get("status") != "placed":
        print("Entry failed:", order)
        return
    # 2) immediately attach stop/target via edit (Kraken supports it)
    oid = order["sendStatus"]["order_id"]
    kraken.edit_order({
        "order_id": oid,
        "stopLoss": stop,
        "takeProfit": target,
    })
    print("Bracket attached.")

if __name__ == "__main__":
    execute_trade("BUY", 0.0001)  # quick sanity check
