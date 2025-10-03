#!/usr/bin/env python3
"""
live_sma_cross_daily.py
5-SMA / 200-SMA cross-over bot for Kraken-Futures pf_xbtusd – DAILY version.
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
# 90-/30-day SMA cross – DAILY version
# ------------------------------------------------------------------
def get_ohlc_daily(n: int = 100) -> pd.DataFrame:
    """Pull enough history for 90-day SMA (plus a small buffer)."""
    return get_ohlc(KRAKEN_SPOT_PAIR, INTERVAL).tail(n)

def compute_smas(df: pd.DataFrame) -> tuple[float, float]:
    """Return (30-day SMA, 90-day SMA)."""
    closes = df["close"]
    return closes.tail(30).mean(), closes.tail(90).mean()

def get_position(api: KrakenFuturesApi) -> float:
    raw = api.get_open_positions()
    for p in raw.get("openPositions", []):
        if p.get("symbol", "").upper() == SYMBOL.upper():
            side_mult = 1 if p.get("side") == "long" else -1
            return float(p.get("size", 0)) * side_mult
    return 0.0

def get_usd_balance(api: KrakenFuturesApi) -> float:
    """
    Return the quantity of USD sitting in the flex wallet.
    """
    flex = api.get_accounts()["accounts"]["flex"]
    for cur in flex["currencies"]:
        if cur["currency"].upper() == "USD":
            return float(cur["quantity"])
    raise RuntimeError("USD balance not found in flex wallet")

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
        df = get_ohlc_daily(220)
        sma5, sma200 = compute_smas(df)
        log.info("OHLC ok | 5-SMA=%.2f 200-SMA=%.2f", sma5, sma200)

        pv_usd = get_usd_balance(api)
        log.info("USD balance ok | %.2f USD", pv_usd)

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
    df = get_ohlc_daily(220)
    prev_5, prev_200 = compute_smas(df)

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
            df = get_ohlc_daily(220)
            cur_5, cur_200 = compute_smas(df)
            btc_price = fetch_btc_price()

            bullish = prev_5 <= prev_200 and cur_5 > cur_200
            bearish = prev_5 >= prev_200 and cur_5 < cur_200

            if not (bullish or bearish):
                log.info("No cross | 5-SMA=%.2f 200-SMA=%.2f", cur_5, cur_200)
                prev_5, prev_200 = cur_5, cur_200
                continue

            # single-order logic
            pv_usd = get_usd_balance(api)
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

            prev_5, prev_200 = cur_5, cur_200

        except KeyboardInterrupt:
            log.info("User interrupt – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(300)  # 5-min back-off, then retry next day

if __name__ == "__main__":
    run()
