#!/usr/bin/env python3
"""
live_sma_cross_daily.py
16-SMA / 128-SMA cross-over bot for Kraken-Futures pf_xbtusd – DAILY version.
Single-order logic: every day at 00:01 UTC send one market order that
moves the BTC exposure to 5 × portfolioValue (in BTC terms).
"""

import os
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import requests

from kraken_futures import KrakenFuturesApi
from kraken_ohlc import get_ohlc

SYMBOL           = "pf_xbtusd"
INTERVAL         = 1440          # 1-day candles
LEVERAGE         = 5
KRAKEN_SPOT_PAIR = "XXBTZUSD"
MIN_ORDER_BTC    = 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sma-cross-daily")

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
        if t["symbol"].upper() == SYMBOL.upper():
            return float(t["markPrice"])
    raise RuntimeError(f"{SYMBOL} mark price not found")

# ------------------------------------------------------------------
# 16-/128-day SMA cross – DAILY version
# ------------------------------------------------------------------
def get_ohlc_daily(n: int = 150) -> pd.DataFrame:
    """Pull enough history for 128-day SMA (plus a small buffer)."""
    return get_ohlc(KRAKEN_SPOT_PAIR, INTERVAL).tail(n)

def compute_smas(df: pd.DataFrame) -> tuple[float, float]:
    """Return (16-day SMA, 128-day SMA)."""
    closes = df["close"]
    return closes.tail(16).mean(), closes.tail(128).mean()

def get_position(api: KrakenFuturesApi) -> float:
    raw = api.get_open_positions()
    for p in raw.get("openPositions", []):
        if p.get("symbol", "").upper() == SYMBOL.upper():
            side_mult = 1 if p.get("side") == "long" else -1
            return float(p.get("size", 0)) * side_mult
    return 0.0

def get_portfolio_value_usd(api: KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])

# ------------------------------------------------------------------
# single-order helper
# ------------------------------------------------------------------
def place_market_order(api: KrakenFuturesApi, order_size: float) -> Dict[str, Any]:
    side = "buy" if order_size > 0 else "sell"
    params = {
        "orderType": "mkt",
        "symbol": SYMBOL,
        "side": side,
        "size": round(abs(order_size), 4),
    }
    return api.send_order(params)

# ------------------------------------------------------------------
# capability test
# ------------------------------------------------------------------
def capability_test(api: KrakenFuturesApi):
    log.info("=== capability test start ===")
    try:
        df = get_ohlc_daily(150)
        sma16, sma128 = compute_smas(df)
        log.info("OHLC ok | 16-SMA=%.2f 128-SMA=%.2f", sma16, sma128)

        pv_usd = get_portfolio_value_usd(api)
        log.info("Portfolio value ok | %.2f USD", pv_usd)

        btc_price = fetch_btc_price()
        log.info("Mark-price fetch ok | %s = %.2f", SYMBOL, btc_price)

        current_btc = get_position(api)
        pv_btc = usd_to_btc(pv_usd, btc_price)
        target_btc = 5 * pv_btc
        delta_btc = target_btc - current_btc

        if abs(delta_btc) >= MIN_ORDER_BTC:
            log.info("Test single order | current %.4f BTC → target %.4f BTC (delta %.4f)",
                     current_btc, target_btc, delta_btc)
            place_market_order(api, delta_btc)
            time.sleep(2)
            new_pos = get_position(api)
            log.info("Test order ok | new position %.6f BTC", new_pos)
            # flatten
            flatten = -new_pos
            if abs(flatten) >= MIN_ORDER_BTC:
                place_market_order(api, flatten)
                time.sleep(1)
        else:
            log.info("Test delta %.6f BTC too small, skipping order", delta_btc)

        # imagined bearish cross
        target_btc = -5 * pv_btc
        delta_btc = target_btc - get_position(api)
        if abs(delta_btc) >= MIN_ORDER_BTC:
            log.info("Imagined bearish cross | delta %.4f BTC", delta_btc)
            place_market_order(api, delta_btc)
            time.sleep(2)
            new_pos = get_position(api)
            log.info("Imagined bearish cross ok | position %.6f BTC", new_pos)
            flatten = -new_pos
            if abs(flatten) >= MIN_ORDER_BTC:
                place_market_order(api, flatten)

    except Exception as e:
        log.exception("Capability test FAILED: %s", e)
        raise
    log.info("=== capability test passed ===")

# ------------------------------------------------------------------
# main loop – daily at 00:01 UTC
# ------------------------------------------------------------------
def run():
    api = KrakenFuturesApi(
        os.getenv("KRAKEN_FUTURES_KEY", ""),
        os.getenv("KRAKEN_FUTURES_SECRET", ""),
    )

    capability_test(api)

    log.info("Seeding SMA history…")
    df = get_ohlc_daily(150)
    prev_16, prev_128 = compute_smas(df)

    log.info("Entering main loop – trades every day at 00:01 UTC")
    while True:
        try:
            now = datetime.utcnow()
            next_run = (now.replace(hour=0, minute=1, second=0, microsecond=0) +
                        timedelta(days=1) if now.time() >= datetime.strptime("00:01", "%H:%M").time()
                        else now.replace(hour=0, minute=1, second=0, microsecond=0))
            seconds_until = (next_run - now).total_seconds()
            log.info("Sleeping %.0f s until next run at %s UTC", seconds_until, next_run.strftime("%Y-%m-%d %H:%M"))
            time.sleep(seconds_until)

            # refresh
            df = get_ohlc_daily(150)
            cur_16, cur_128 = compute_smas(df)
            btc_price = fetch_btc_price()

            bullish = prev_16 <= prev_128 and cur_16 > cur_128
            bearish = prev_16 >= prev_128 and cur_16 < cur_128

            if not (bullish or bearish):
                log.info("No cross | 16-SMA=%.2f 128-SMA=%.2f", cur_16, cur_128)
                prev_16, prev_128 = cur_16, cur_128
                continue

            # single-order logic
            pv_usd = get_portfolio_value_usd(api)
            pv_btc = usd_to_btc(pv_usd, btc_price)
            current_btc = get_position(api)
            target_btc = LEVERAGE * pv_btc if bullish else -LEVERAGE * pv_btc
            delta_btc = target_btc - current_btc

            if abs(delta_btc) < MIN_ORDER_BTC:
                log.info("Cross detected but delta %.6f BTC too small, skipping", delta_btc)
            else:
                log.info("%s cross | current %.4f BTC → target %.4f BTC (delta %.4f)",
                         "Bullish" if bullish else "Bearish", current_btc, target_btc, delta_btc)
                resp = place_market_order(api, delta_btc)
                log.info("Order sent | recv=%s", resp)

            prev_16, prev_128 = cur_16, cur_128

        except KeyboardInterrupt:
            log.info("User interrupt – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(300)  # 5-min back-off, then retry next day

if __name__ == "__main__":
    run()
