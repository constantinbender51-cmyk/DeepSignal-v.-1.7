#!/usr/bin/env python3
"""
live_sma_cross.py
200/5 SMA cross-over bot for Kraken-Futures PF_XBTCUSD.
Places one market order per cross, sized to 5× available margin.
Runs exactly at the top of every hour (12:00, 13:00 …).
Performs a capability test on first launch (OHLC + margin query + 0-size order).
"""

import os
import time
import math
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import requests

from kraken_futures import KrakenFuturesApi
from kraken_ohlc import get_ohlc

SYMBOL          = "PF_XBTUSD"
INTERVAL        = 60
LEVERAGE        = 5
KRAKEN_SPOT_PAIR= "XXBTZUSD"
MIN_ORDER_BTC   = 0.0001

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
    return usd / btc_price

def round_to_tick(qty: float) -> float:
    return round(qty / MIN_ORDER_BTC) * MIN_ORDER_BTC

def fetch_btc_price() -> float:
    r = requests.get("https://futures.kraken.com/derivatives/api/v3/tickers", timeout=10)
    r.raise_for_status()
    for t in r.json()["tickers"]:
        if t["symbol"] == SYMBOL:
            return float(t["markPrice"])
    raise RuntimeError(f"{SYMBOL} mark price not found")

def get_ohlc_60min(n: int = 220) -> pd.DataFrame:
    return get_ohlc(KRAKEN_SPOT_PAIR, INTERVAL).tail(n)

def compute_smas(df: pd.DataFrame) -> tuple[float, float]:
    closes = df["close"]
    return closes.tail(5).mean(), closes.tail(200).mean()

def get_position(api: KrakenFuturesApi) -> float:
    pos = api.get_open_positions()
    for p in pos.get("openPositions", []):
        if p["symbol"] == SYMBOL:
            side_mult = 1 if p["side"] == "long" else -1
            return float(p["size"]) * side_mult
    return 0.0

def get_usd_available_margin(api: KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["availableMargin"])

def place_market_order(api: KrakenFuturesApi, order_size: float) -> Dict[str, Any]:
    side = "buy" if order_size > 0 else "sell"
    params = {
        "orderType": "market",
        "symbol": SYMBOL,
        "side": side,
        "size": str(abs(order_size)),
    }
    return api.send_order(params)

# ------------------------------------------------------------------
# capability test
# ------------------------------------------------------------------
def capability_test(api: KrakenFuturesApi):
    log.info("=== capability test start ===")
    try:
        # 1. OHLC
        df = get_ohlc_60min(200)
        sma5, sma200 = compute_smas(df)
        log.info("OHLC ok | 5-SMA=%.2f 200-SMA=%.2f", sma5, sma200)

        # 2. margin
        usd = get_usd_available_margin(api)
        log.info("Margin ok | available USD %.2f", usd)

        # 3. 0-size order
        place_market_order(api, 0.0)
        log.info("0-size order ok")
    except Exception as e:
        log.exception("Capability test FAILED: %s", e)
        raise
    log.info("=== capability test passed ===")

# ------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------
def run():
    api = KrakenFuturesApi(
        os.getenv("KRAKEN_FUTURES_KEY", ""),
        os.getenv("KRAKEN_FUTURES_SECRET", ""),
    )

    # first-run tests
    capability_test(api)

    # seed SMAs
    log.info("Seeding SMA history…")
    df = get_ohlc_60min(220)
    prev_5, prev_200 = compute_smas(df)

    log.info("Entering main loop – trades at the top of every hour")
    while True:
        try:
            # wait until next whole hour
            now = datetime.utcnow()
            seconds_until = 3600 - now.minute * 60 - now.second
            if seconds_until <= 0:
                seconds_until += 3600
            log.info("Sleeping %s s until next hour", seconds_until)
            time.sleep(seconds_until)

            # refresh
            df = get_ohlc_60min(220)
            cur_5, cur_200 = compute_smas(df)
            btc_price = fetch_btc_price()

            bullish = prev_5 <= prev_200 and cur_5 > cur_200
            bearish = prev_5 >= prev_200 and cur_5 < cur_200

            if not (bullish or bearish):
                log.info("No cross | 5-SMA=%.2f 200-SMA=%.2f", cur_5, cur_200)
                prev_5, prev_200 = cur_5, cur_200
                continue

            # sizing
            usd_margin = get_usd_available_margin(api)
            target_btc = round_to_tick(usd_to_btc(usd_margin, btc_price) * LEVERAGE)
            target_qty = target_btc if bullish else -target_btc
            current_qty = get_position(api)
            order_size = target_qty - current_qty

            if abs(order_size) < MIN_ORDER_BTC:
                log.info("Order size %.6f BTC too small, skipping", order_size)
                prev_5, prev_200 = cur_5, cur_200
                continue

            log.info(
                "%s cross | current=%.4f BTC | target=%.4f BTC | order=%.4f BTC",
                "Bullish" if bullish else "Bearish",
                current_qty,
                target_qty,
                order_size,
            )
            resp = place_market_order(api, order_size)
            log.info("Order sent | recv=%s", resp)

            prev_5, prev_200 = cur_5, cur_200

        except KeyboardInterrupt:
            log.info("User interrupt – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(60)   # small back-off, then retry at next hour


if __name__ == "__main__":
    run()
