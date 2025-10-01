#!/usr/bin/env python3
"""
live_sma_cross.py
200/5 SMA cross-over bot for Kraken-Futures PF_XBTUSD.
Places one market order per cross, sized to 5× available margin.
Run:
    export KRAKEN_FUTURES_KEY="YOUR_KEY"
    export KRAKEN_FUTURES_SECRET="YOUR_SECRET"
    python live_sma_cross.py
"""

import os
import time
import math
import logging
from typing import Dict, Any, Optional

import pandas as pd
import requests

from kraken_futures import KrakenFuturesApi   # the file you already have
from kraken_ohlc import get_ohlc, VALID_GRANULARITY

SYMBOL = "PF_XBTUSD"
INTERVAL = 60                       # 60-min candles
LEVERAGE = 5
KRAKEN_SPOT_OHLC_PAIR = "XXBTZUSD"  # spot pair for OHLC (Kraken uses this key)
MIN_ORDER_BTC = 0.0001              # contract tick size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sma-cross")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def usd_to_btc(usd: float, btc_price: float) -> float:
    """Convert USD value to BTC amount."""
    return usd / btc_price


def round_to_tick(qty: float) -> float:
    """Round BTC quantity to contract tick."""
    return round(qty / MIN_ORDER_BTC) * MIN_ORDER_BTC


def fetch_btc_price() -> float:
    """Latest mark price for PF_XBTUSD."""
    r = requests.get("https://futures.kraken.com/derivatives/api/v3/tickers", timeout=10)
    r.raise_for_status()
    for t in r.json()["tickers"]:
        if t["symbol"] == SYMBOL:
            return float(t["markPrice"])
    raise RuntimeError(f"{SYMBOL} mark price not found")


def get_ohlc_60min(n: int = 220) -> pd.DataFrame:
    """Download last n 60-min candles."""
    return get_ohlc(KRAKEN_SPOT_OHLC_PAIR, INTERVAL).tail(n)


def compute_smas(df: pd.DataFrame) -> tuple[float, float]:
    """Return current 5-SMA and 200-SMA from close prices."""
    closes = df["close"]
    return closes.tail(5).mean(), closes.tail(200).mean()


def get_position(api: KrakenFuturesApi) -> float:
    """Return signed BTC position size (long+, short-)."""
    pos = api.get_open_positions()
    for p in pos.get("openPositions", []):
        if p["symbol"] == SYMBOL:
            side_mult = 1 if p["side"] == "long" else -1
            return float(p["size"]) * side_mult
    return 0.0


def get_usd_available_margin(api: KrakenFuturesApi) -> float:
    """Return USD value in flex.availableMargin."""
    acc = api.get_accounts()
    return float(acc["accounts"]["flex"]["availableMargin"])


def place_market_order(api: KrakenFuturesApi, order_size: float) -> Dict[str, Any]:
    """Send one market order. order_size signed: buy>0, sell<0."""
    side = "buy" if order_size > 0 else "sell"
    size = abs(order_size)
    params = {
        "orderType": "market",
        "symbol": SYMBOL,
        "side": side,
        "size": str(size),
    }
    return api.send_order(params)


# ------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------
def run():
    api = KrakenFuturesApi(
        os.getenv("KRAKEN_FUTURES_KEY", ""),
        os.getenv("KRAKEN_FUTURES_SECRET", ""),
    )

    # seed 200 candles
    log.info("Seeding historical candles…")
    df = get_ohlc_60min(220)
    prev_5, prev_200 = compute_smas(df)

    log.info("Entering main loop (check every hour + 5 s)")
    while True:
        try:
            # wait until :05 past the hour
            t = time.gmtime()
            seconds_until = (65 - t.tm_sec) % 3600
            if seconds_until == 0:
                seconds_until = 3600
            log.info("Sleeping %s s until next check", seconds_until)
            time.sleep(seconds_until)

            # refresh data
            df = get_ohlc_60min(220)
            cur_5, cur_200 = compute_smas(df)
            btc_price = fetch_btc_price()

            # cross detection
            bullish_cross = prev_5 <= prev_200 and cur_5 > cur_200
            bearish_cross = prev_5 >= prev_200 and cur_5 < cur_200

            if not (bullish_cross or bearish_cross):
                log.info("No cross | 5-SMA=%.2f 200-SMA=%.2f", cur_5, cur_200)
                prev_5, prev_200 = cur_5, cur_200
                continue

            # sizing
            usd_margin = get_usd_available_margin(api)
            target_btc = usd_to_btc(usd_margin, btc_price) * LEVERAGE
            target_btc = round_to_tick(target_btc)
            target_qty = target_btc if bullish_cross else -target_btc

            current_qty = get_position(api)
            order_size = target_qty - current_qty

            if abs(order_size) < MIN_ORDER_BTC:
                log.info("Order size %.6f BTC too small, skipping", order_size)
                prev_5, prev_200 = cur_5, cur_200
                continue

            # send order
            log.info(
                "%s cross | current=%.4f BTC | target=%.4f BTC | order=%.4f BTC",
                "Bullish" if bullish_cross else "Bearish",
                current_qty,
                target_qty,
                order_size,
            )
            resp = place_market_order(api, order_size)
            log.info("Order sent | recv=%s", resp)

            prev_5, prev_200 = cur_5, cur_200

        except KeyboardInterrupt:
            log.info("Interrupted by user – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    run()
