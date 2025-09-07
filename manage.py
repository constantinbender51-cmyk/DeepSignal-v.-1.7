import glob, json, os, time
from kraken_futures import KrakenFuturesApi

kraken = KrakenFuturesApi(os.getenv("KRAKEN_FUTURES_KEY"),
                          os.getenv("KRAKEN_FUTURES_SECRET"))
SIZE = 0.0001

def check_all_triggers():
    for file in glob.glob("trigger_*.json"):
        with open(file) as f:
            t = json.load(f)
        # use *current* mark price for every slice
        tick = kraken.get_tickers()                      # whole response
        rec  = next(t for t in tick["tickers"] if t["symbol"] == "pf_xbtusd")
        mark = float(rec["markPrice"])

        side, stop, tgt = t["side"], t["stop"], t["target"]
        # crude pct move since file birth (we don’t store entry)
        if (side == "long"  and (mark <= mark * (1 + stop/100) or mark >= mark * (1 + tgt/100))) or \
           (side == "short" and (mark >= mark * (1 - stop/100) or mark <= mark * (1 - tgt/100))):
            kraken.send_order({
                "orderType": "mkt",
                "symbol": "pf_xbtusd",
                "side": "sell" if side == "long" else "buy",
                "size": SIZE,
            })
            os.remove(file)
            print(f"Slice {file} hit  mark={mark}")

if __name__ == "__main__":
    while True:
        check_all_triggers()
        time.sleep(10)

