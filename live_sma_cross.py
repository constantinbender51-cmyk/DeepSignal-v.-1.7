#!/usr/bin/env python3
"""
live_sma_cross.py
200/5 SMA cross-over bot for Kraken-Futures pf_xbtusd.
Places one market order per cross, sized to 5× available margin.
Runs exactly at the top of every hour.
Capability test: downloads OHLC, queries margin, sends a 5× leveraged order
and verifies it appears in openPositions.
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

SYMBOL          = "pf_xbtusd"
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
        if t["symbol"].upper() == SYMBOL.upper():   # case-insensitive
            return float(t["markPrice"])
    raise RuntimeError(f"{SYMBOL} mark price not found")

def get_ohlc_60min(n: int = 220) -> pd.DataFrame:
    return get_ohlc(KRAKEN_SPOT_PAIR, INTERVAL).tail(n)

def compute_smas(df: pd.DataFrame) -> tuple[float, float]:
    closes = df["close"]
    return closes.tail(5).mean(), closes.tail(200).mean()

def get_position(api: KrakenFuturesApi) -> float:
    """Return signed BTC position size (long+, short-)."""
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

        # 3. fetch_btc_price
        btc_price = fetch_btc_price()
        log.info("Mark-price fetch ok | %s = %.2f", SYMBOL, btc_price)

        # 4. 5× leveraged test order + position check
        test_size = round_to_tick(usd_to_btc(usd, btc_price) * LEVERAGE)
        log.info("Sending test market order size=%s BTC (≈ 5× margin)", test_size)
        place_market_order(api, test_size)

        time.sleep(2)
        pos = get_position(api)
        if abs(pos) < MIN_ORDER_BTC:
            raise RuntimeError("Test order did not show in openPositions")
        log.info("Test order ok | detected position %.6f BTC", pos)

        # flatten immediately
        flatten_size = -pos
        place_market_order(api, flatten_size)
        log.info("Flattened test position")

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

            # ----------------------------------------------------------
            # 1.  Flatten first so all margin is released
            # ----------------------------------------------------------
            current_qty = get_position(api)
            flat_size   = -current_qty
            if abs(flat_size) >= MIN_ORDER_BTC:
                log.info("Flattening current position (%.6f BTC)", current_qty)
                place_market_order(api, flat_size)
                time.sleep(1)          # give the engine a tick

            # ----------------------------------------------------------
            # 2.  Re-check buying power now that margin is back
            # ----------------------------------------------------------
            usd_margin = get_usd_available_margin(api)
            target_btc = round_to_tick(usd_to_btc(usd_margin, btc_price) * LEVERAGE)
            target_qty = target_btc if bullish else -target_btc

            # ----------------------------------------------------------
            # 3.  Enter the new directional leg
            # ----------------------------------------------------------
            order_size = target_qty          # already flat, so no offset needed
            if abs(order_size) >= MIN_ORDER_BTC:
                log.info(
                    "%s cross | flat then new target=%.4f BTC | order=%.4f BTC",
                    "Bullish" if bullish else "Bearish",
                    target_qty,
                    order_size,
                )
                resp = place_market_order(api, order_size)
                log.info("Order sent | recv=%s", resp)
            else:
                log.info("New target size %.6f BTC too small, skipping", order_size)

            prev_5, prev_200 = cur_5, cur_200

        except KeyboardInterrupt:
            log.info("User interrupt – shutting down")
            break
        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(60)   # small back-off, then retry at next hour


if __name__ == "__main__":
    run()
