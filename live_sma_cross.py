#!/usr/bin/env python3
"""
live_sma_cross_daily.py
200/5 SMA cross-over bot for Kraken-Futures pf_xbtusd – DAILY version.
Places one market order per cross, sized to 5× available margin.
Runs every day at 00:01 UTC (1 minute after the daily candle closes).
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
# helpers (unchanged)
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

def get_usd_available_margin(api: KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["availableMargin"])

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
# capability test (unchanged logic, just uses daily candles)
# ------------------------------------------------------------------
def capability_test(api: KrakenFuturesApi):
    log.info("=== capability test start ===")
    try:
        df = get_ohlc_daily(200)
        sma5, sma200 = compute_smas(df)
        log.info("OHLC ok | 5-SMA=%.2f 200-SMA=%.2f", sma5, sma200)

        usd = get_usd_available_margin(api)
        log.info("Margin ok | available USD %.2f", usd)

        btc_price = fetch_btc_price()
        log.info("Mark-price fetch ok | %s = %.2f", SYMBOL, btc_price)

        test_size = round_to_tick(usd_to_btc(usd, btc_price) * LEVERAGE)
        log.info("Sending test market order size=%s BTC (≈ 5× margin)", test_size)
        place_market_order(api, test_size)

        time.sleep(2)
        pos = get_position(api)
        if abs(pos) < MIN_ORDER_BTC:
            raise RuntimeError("Test order did not show in openPositions")
        log.info("Test order ok | detected position %.6f BTC", pos)

        flatten_size = -pos
        place_market_order(api, flatten_size)
        log.info("Flattened test position – account clean")

        # imagined bullish cross
        current_qty = get_position(api)
        flat_size = -current_qty
        if abs(flat_size) >= MIN_ORDER_BTC:
            place_market_order(api, flat_size); time.sleep(1)

        usd_margin = get_usd_available_margin(api)
        target_btc = round_to_tick(usd_to_btc(usd_margin, btc_price) * LEVERAGE)
        target_qty = target_btc
        if abs(target_qty) >= MIN_ORDER_BTC:
            place_market_order(api, target_qty)
            time.sleep(2)
            pos = get_position(api)
            if pos < MIN_ORDER_BTC: raise RuntimeError("Imagined bullish cross failed")
            log.info("Imagined bullish cross ok | position %.6f BTC", pos)
            place_market_order(api, -pos)  # flatten

        # imagined bearish cross
        usd_margin = get_usd_available_margin(api)
        target_btc = round_to_tick(usd_to_btc(usd_margin, btc_price) * LEVERAGE)
        target_qty = -target_btc
        if abs(target_qty) >= MIN_ORDER_BTC:
            place_market_order(api, target_qty)
            time.sleep(2)
            pos = get_position(api)
            if pos > -MIN_ORDER_BTC: raise RuntimeError("Imagined bearish cross failed")
            log.info("Imagined bearish cross ok | position %.6f BTC", pos)
            place_market_order(api, -pos)  # flatten

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

            # flatten
            current_qty = get_position(api)
            flat_size = -current_qty
            if abs(flat_size) >= MIN_ORDER_BTC:
                log.info("Flattening current position (%.6f BTC)", current_qty)
                place_market_order(api, flat_size)
                time.sleep(1)

            # new leg
            usd_margin = get_usd_available_margin(api)
            target_btc = round_to_tick(usd_to_btc(usd_margin, btc_price) * LEVERAGE)
            target_qty = target_btc if bullish else -target_btc
            if abs(target_qty) >= MIN_ORDER_BTC:
                log.info("%s cross | new target=%.4f BTC", "Bullish" if bullish else "Bearish", target_qty)
                resp = place_market_order(api, target_qty)
                log.info("Order sent | recv=%s", resp)
            else:
                log.info("New target size %.6f BTC too small, skipping", target_qty)

            prev_5, prev_200 = cur_5, cur_200

        except KeyboardInterrupt:
            log.info("User interrupt – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(300)  # 5-min back-off, then retry next day

if __name__ == "__main__":
    run()
