import json, uuid, os
from kraken_futures import KrakenFuturesApi

kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))
SIZE = 0.0001

def add_slice(side: str, stop: float, target: float):
    """Fire market entry and save a unique trigger file for this 0.0001 slice."""
    kraken.send_order({
        "orderType": "mkt",
        "symbol": "pf_xbtusd",
        "side": side.lower(),
        "size": SIZE,
    })
    uid = str(uuid.uuid4())[:8]
    with open(f"trigger_{uid}.json", "w") as f:
        json.dump({"side": side.lower(), "stop": stop, "target": target}, f)
    print(f"Slice {uid} added  side={side}  stop={stop}%  target={target}%")
