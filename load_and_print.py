import pandas as pd
import time

# ------------------------------------------------------------------
# 1. LOAD CSV & CONVERT TO DAILY CANDLES
# ------------------------------------------------------------------
def load_and_convert_to_daily(path='xbtusd_1h_8y.csv'):
    try:
        df = pd.read_csv(path)
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)

        # Resample to daily candles
        daily = df.resample('D').agg({
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return daily
    except FileNotFoundError:
        print("Error: 'xbtusd_1h_8y.csv' not found."); return None
    except Exception as e:
        print(f"Data load error: {e}"); return None

# ------------------------------------------------------------------
# 2. PRINT DAILY CANDLES WITH 0.01 sec DELAY
# ------------------------------------------------------------------
def print_daily_candles(daily_df):
    if daily_df is None or daily_df.empty:
        print("No daily candles to display.")
        return

    for ts, row in daily_df.iterrows():
        print(f"{ts.date()} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.0f}")
        time.sleep(0.01)

# ------------------------------------------------------------------
# 3. 200-SMA CROSS STRATEGY (5× lev, 100 USD margin, 2 % stop-loss)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 3. 5-SMA CROSS STRATEGY (200…196, 5× lev, 100 USD each, 2 % SL, 0.25 % fee)
# ------------------------------------------------------------------
def run_sma_cross(daily_df, leverage=5, initial_margin=100, fee=0.0025, stop_pct=0.02):
    """Trade five parallel SMA-cross legs (200,199,198,197,196)."""
    if daily_df is None or daily_df.empty:
        print("No data – nothing to back-test.")
        return

    close = daily_df['close']
    lookbacks = [200, 199, 198, 197, 196]
    smas = {n: close.rolling(n).mean() for n in lookbacks}

    # --- per-leg state ---------------------------------------------------------
    class Leg:
        def __init__(self, n):
            self.n = n
            self.side = 0          # 1 long, -1 short, 0 flat
            self.position = 0.0
            self.entry_price = 0.0
            self.balance = initial_margin  # 100 USD dedicated to this leg

        def pos_size(self, price):
            return self.balance * leverage / price

    legs = {n: Leg(n) for n in lookbacks}
    max_bal_total = initial_margin * len(lookbacks)
    max_dd_total = 0.0

    # --- main loop -------------------------------------------------------------
    for ts, row in daily_df.iterrows():
        price = row['close']

        for n, leg in legs.items():
            sma = smas[n].loc[ts]

            # skip until SMA available
            if pd.isna(sma):
                continue

            prev_side = leg.side

            # --- signal generation ---------------------------------------------
            if price > sma and leg.side != 1:
                leg.side = 1
            elif price < sma and leg.side != -1:
                leg.side = -1

            # --- stop-loss check (while in position) -----------------------------
            if leg.position != 0:
                stop_dist = leg.entry_price * stop_pct
                if (leg.position > 0 and price <= leg.entry_price - stop_dist) or \
                   (leg.position < 0 and price >= leg.entry_price + stop_dist):
                    stop_price = leg.entry_price - stop_dist if leg.position > 0 else leg.entry_price + stop_dist
                    pnl = leg.position * (stop_price - leg.entry_price)
                    leg.balance += pnl - abs(pnl) * fee
                    leg.position = 0.0
                    leg.side = 0
                    leg.entry_price = 0.0
                    prev_side = 0  # force fresh cross before next entry

            # --- execute SMA-cross trade only on side change --------------------
            if leg.side != prev_side and leg.side != 0:
                # close old position if still open
                if leg.position != 0:
                    pnl = leg.position * (price - leg.entry_price)
                    leg.balance += pnl - abs(pnl) * fee
                # open new position
                leg.entry_price = price
                leg.position = leg.pos_size(price) * leg.side
                leg.balance -= abs(leg.position * price) * fee

            # --- mark-to-market -------------------------------------------------
            mtm = leg.position * (price - leg.entry_price) if leg.position != 0 else 0.0
            leg.equity = leg.balance + mtm

        # --- portfolio-level stats ---------------------------------------------
        total_eq = sum(leg.equity for leg in legs.values())
        max_bal_total = max(max_bal_total, total_eq)
        dd_total = (max_bal_total - total_eq) / max_bal_total
        max_dd_total = max(max_dd_total, dd_total)

        # --- pretty print: one line per day ------------------------------------
        leg_str = " | ".join(
            f"SMA{n}:{leg.side:+1.0f} {leg.equity:6.2f}" for n, leg in legs.items()
        )
        print(f"{ts.date()} | {leg_str} | Total:{total_eq:8.2f}")
        time.sleep(0.01)

    # --- final summary ---------------------------------------------------------
    final_equities = {n: leg.equity for n, leg in legs.items()}
    final_total = sum(final_equities.values())
    print("\n===== 5-LEG SUMMARY (0.25 % fee, 2 % SL, 5× lev) =====")
    for n, eq in final_equities.items():
        print(f"SMA{n:3d}  : {eq:8.2f} USD  |  return {((eq/initial_margin-1)*100):+6.2f} %")
    print(f"Combined : {final_total:8.2f} USD  |  return {((final_total/(initial_margin*5)-1)*100):+6.2f} %")
    print(f"Max DD   : {max_dd_total*100:6.2f} %")
    print("="*60)

    return {"legs": final_equities, "total": final_total, "max_dd": max_dd_total}
# ------------------------------------------------------------------
# 4. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_and_convert_to_daily()
    run_sma_cross(daily)
    
