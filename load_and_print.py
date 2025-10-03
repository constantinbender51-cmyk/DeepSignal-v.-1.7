import pandas as pd
import time
from itertools import product

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
# 2. GENERIC SMA-CROSS BACK-TESTER
#    – stops if equity ≤ 0
#    – returns one-line metrics dict
# ------------------------------------------------------------------
def sma_backtest(daily_df, sma_len, leverage=5, init_margin=100, fee=0.0005):
    close = daily_df['close']
    sma   = close.rolling(sma_len).mean()
    df    = daily_df.copy()
    df['sma'] = sma
    df.dropna(inplace=True)

    balance = float(init_margin)
    position = 0.0
    side = 0
    entry_price = 0.0
    max_eq = balance
    max_dd = 0.0
    trades = 0

    for price in df['close']:
        # --- signal -------------------------------------------------
        prev_side = side
        if price > sma.iloc[0] and side != 1:
            side = 1
        elif price < sma.iloc[0] and side != -1:
            side = -1
        # else hold current side

        # --- execution ---------------------------------------------
        if side != prev_side:
            # close old
            if position != 0:
                pnl = position * (price - entry_price)
                balance += pnl - abs(pnl) * fee
                position = 0.0
                if balance <= 0:
                    break
            # open new
            if side != 0:
                entry_price = price
                position = (balance * leverage / price) * side
                balance -= abs(position * price) * fee
                trades += 1
                if balance <= 0:
                    break

        # --- mark-to-market ----------------------------------------
        eq = balance + (position * (price - entry_price) if position != 0 else 0)
        max_eq = max(max_eq, eq)
        dd = (max_eq - eq) / max_eq
        max_dd = max(max_dd, dd)
        if eq <= 0:
            break

    final_eq = eq if 'eq' in locals() else balance
    return {
        'sma'      : sma_len,
        'trades'   : trades,
        'final_eq' : final_eq,
        'return_pct': (final_eq / init_margin - 1) * 100,
        'max_dd_pct': max_dd * 100
    }

# ------------------------------------------------------------------
# 3. RUN EVERY SMA 1 → 400
# ------------------------------------------------------------------
def scan_all_smas(daily_df):
    if daily_df is None or daily_df.empty:
        print("No data – aborting scan."); return
    print("sma_len | trades | final_eq | return_% | max_dd_%")
    for n in range(1, 401):
        res = sma_backtest(daily_df, n)
        print(f"{res['sma']:5d} | {res['trades']:6d} | "
              f"{res['final_eq']:8.2f} | {res['return_pct']:7.2f} | "
              f"{res['max_dd_pct']:7.2f}")

# ------------------------------------------------------------------
# 4. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_and_convert_to_daily()
    scan_all_smas(daily)
