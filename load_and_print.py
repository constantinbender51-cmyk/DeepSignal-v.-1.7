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
# 3. SINGLE SMA CROSS STRATEGY (parameterised look-back)
# ------------------------------------------------------------------
def run_one_sma(daily_df, lookback=200, leverage=5, initial_margin=100,
                fee=0.0025, stop_pct=0.02):
    """Run ONE SMA-cross leg and return a summary dict."""
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

    def pos_size(price):
        return balance * leverage / price

    for ts, row in df.iterrows():
        price = row['close']
        ma = row['sma']

        prev_side = side

        # signal
        if price > ma and side != 1:
            side = 1
        elif price < ma and side != -1:
            side = -1

        # stop-loss
        if position != 0:
            stop_dist = entry_price * stop_pct
            if (position > 0 and price <= entry_price - stop_dist) or \
               (position < 0 and price >= entry_price + stop_dist):
                stop_price = entry_price - stop_dist if position > 0 else entry_price + stop_dist
                pnl = position * (stop_price - entry_price)
                balance += pnl - abs(pnl) * fee
                position = 0.0
                side = 0
                prev_side = 0

        # execute cross
        if side != prev_side and side != 0:
            if position != 0:
                pnl = position * (price - entry_price)
                balance += pnl - abs(pnl) * fee
            entry_price = price
            position = pos_size(price) * side
            balance -= abs(position * price) * fee

        # mark-to-market
        mtm = position * (price - entry_price) if position else 0.0
        eq = balance + mtm
        max_bal = max(max_bal, eq)
        dd = (max_bal - eq) / max_bal
        max_dd = max(max_dd, dd)

        # print identical format to original script
        print(f"{ts.date()} | "
              f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} "
              f"V:{row['volume']:.0f} | SMA:{ma:.2f} | "
              f"{'LONG' if side==1 else 'SHORT' if side==-1 else 'FLAT'} | "
              f"Eq:{eq:.2f} USD")
        time.sleep(0.01)

    final_eq = balance + (position * (df['close'].iloc[-1] - entry_price) if position else 0)
    return {
        'lookback': lookback,
        'final_eq': final_eq,
        'return_pct': (final_eq / initial_margin - 1) * 100,
        'max_dd_pct': max_dd * 100
    }


# ------------------------------------------------------------------
# 4. RUN MULTIPLE SMAs (200,199,198,197,196 for now)
# ------------------------------------------------------------------
def run_sma_cross(daily_df):
    """Run each SMA independently and print a table."""
    if daily_df is None or daily_df.empty:
        print("No data – nothing to back-test.")
        return

    lookbacks = [200, 199, 198, 197, 196]          # easy to extend to range(1,201)
    results = []

    for lb in lookbacks:
        print(f"\n========== SMA {lb} ==========")
        res = run_one_sma(daily_df, lookback=lb)
        if res:
            results.append(res)

    # summary table
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
    
