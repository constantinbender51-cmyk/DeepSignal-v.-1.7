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
# 2. (silent) DAILY CANDLES  –  no prints, no sleep
# ------------------------------------------------------------------
def print_daily_candles(daily_df):
    """Kept for compatibility; does nothing."""
    pass

# ------------------------------------------------------------------
# 3. SINGLE SMA CROSS STRATEGY – no per-bar output
# ------------------------------------------------------------------
def run_one_sma(daily_df, lookback=200, leverage=5, initial_margin=100,
                fee=0.0025, stop_pct=0.10):
    if daily_df is None or daily_df.empty:
        return None

    close = daily_df['close']
    sma = close.rolling(lookback).mean()
    df = daily_df.copy()
    df['sma'] = sma
    df.dropna(inplace=True)

    position = 0.0
    balance = initial_margin
    max_bal = balance
    max_dd = 0.0
    side = 0
    entry_price = 0.0
    trades = 0

    def pos_size(price):
        return balance * leverage / price

    for ts, row in df.iterrows():
        price = row['close']
        ma = row['sma']
        prev_side = side

        # signal generation
        if price > ma and side != 1:
            side = 1
        elif price < ma and side != -1:
            side = -1

        # ---------- intra-day stop check ----------
        if position != 0:
            stop_dist = entry_price * stop_pct
            if position > 0:                       # long
                stop_price = entry_price - stop_dist
                if row['low'] <= stop_price:
                    pnl = position * (stop_price - entry_price)
                    balance += pnl - abs(pnl) * fee
                    position = 0.0
                    side = 0
                    prev_side = 0
                    trades += 1
            else:                                  # short
                stop_price = entry_price + stop_dist
                if row['high'] >= stop_price:
                    pnl = position * (stop_price - entry_price)
                    balance += pnl - abs(pnl) * fee
                    position = 0.0
                    side = 0
                    prev_side = 0
                    trades += 1
        # ------------------------------------------

        # cross execution
        if side != prev_side and side != 0:
            if position != 0:
                pnl = position * (price - entry_price)
                balance += pnl - abs(pnl) * fee
                trades += 1
            entry_price = price
            position = pos_size(price) * side
            balance -= abs(position * price) * fee

        # mark-to-market
        mtm = position * (price - entry_price) if position else 0.0
        eq = balance + mtm
        max_bal = max(max_bal, eq)
        dd = (max_bal - eq) / max_bal
        max_dd = max(max_dd, dd)
        # (no print, no sleep)

    # final exit
    if position != 0:
        pnl = position * (df['close'].iloc[-1] - entry_price)
        balance += pnl - abs(pnl) * fee
        trades += 1

    final_eq = balance
    return {
        'lookback': lookback,
        'final_eq': final_eq,
        'return_pct': (final_eq / initial_margin - 1) * 100,
        'max_dd_pct': max_dd * 100,
        'trades': trades
    }

# ------------------------------------------------------------------
# 4. MULTI-SMA BACK-TEST – only final table printed
# ------------------------------------------------------------------
def run_sma_cross(daily_df):
    if daily_df is None or daily_df.empty:
        print("No data – nothing to back-test.")
        return

    lookbacks = [200, 199, 198, 197, 196]
    results = []

    for lb in lookbacks:
        res = run_one_sma(daily_df, lookback=lb)
        if res:
            results.append(res)

    print("\n===== INDEPENDENT RUNS SUMMARY =====")
    for r in results:
        print(f"SMA{r['lookback']:3.0f} | Final: {r['final_eq']:8.2f} USD | "
              f"Return: {r['return_pct']:+7.2f} % | Max DD: {r['max_dd_pct']:6.2f} %")
    print("="*42)
# ------------------------------------------------------------------
# 4. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_and_convert_to_daily()
    run_sma_cross(daily)
    
