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
# 3-bis. SINGLE SMA CROSS STRATEGY – 50/200 CROSS FILTER  (SILENT)
# ------------------------------------------------------------------
def run_one_sma(daily_df, lookback=200, leverage=1, initial_margin=100,
                fee=0.0025, stop_pct=1):
    """
    Runs the 50/200 SMA-cross strategy once.
    Returns a dict with all common performance metrics plus
    max_loss_trade, max_gain_trade, n_losses, n_gains (no prints).
    """
    if daily_df is None or daily_df.empty:
        return None

    close = daily_df['close']
    sma_slow = close.rolling(lookback).mean()
    sma_fast = close.rolling(50).mean()
    df = daily_df.copy()
    df['sma_slow'] = sma_slow
    df['sma_fast'] = sma_fast
    df.dropna(inplace=True)

    position = 0.0
    balance = initial_margin
    max_bal = balance
    max_dd = 0.0
    side = 0
    entry_price = 0.0
    trades = 0

    # ---- new tracking variables ----
    max_gain_trade = 0.0
    max_loss_trade = 0.0
    n_gains = 0
    n_losses = 0

    def pos_size(price):
        return balance * leverage / price

    for ts, row in df.iterrows():
        price = row['close']
        ma_slow = row['sma_slow']
        ma_fast = row['sma_fast']
        prev_side = side

        # 1. raw 200-SMA direction
        raw_side = 0
        if price > ma_slow:
            raw_side = 1
        elif price < ma_slow:
            raw_side = -1

        # 2. only act if 50-SMA agrees
        if raw_side == 1 and ma_fast > ma_slow:
            side = 1
        elif raw_side == -1 and ma_fast < ma_slow:
            side = -1
        else:
            side = 0

        # --- intra-day stop check ---
        stopped = False
        if position != 0:
            stop_dist = entry_price * stop_pct
            if position > 0:                       # long
                stop_price = entry_price - stop_dist
                if row['low'] <= stop_price:
                    pnl = position * (stop_price - entry_price)
                    fee_cost = abs(pnl) * fee
                    balance += pnl - fee_cost
                    total_fees += fee_cost
                    if pnl > 0:
                        n_gains += 1
                        max_gain_trade = max(max_gain_trade, pnl)
                    else:
                        n_losses += 1
                        max_loss_trade = min(max_loss_trade, pnl)
                    position = 0.0
                    stopped = True                 # mark stop but do NOT ++trades yet
            else:                                  # short
                stop_price = entry_price + stop_dist
                if row['high'] >= stop_price:
                    pnl = position * (stop_price - entry_price)
                    fee_cost = abs(pnl) * fee
                    balance += pnl - fee_cost
                    total_fees += fee_cost
                    if pnl > 0:
                        n_gains += 1
                        max_gain_trade = max(max_gain_trade, pnl)
                    else:
                        n_losses += 1
                        max_loss_trade = min(max_loss_trade, pnl)
                    position = 0.0
                    stopped = True

        # --- cross at close if still flat OR side changed ---
        if side != prev_side:                      # real regime change
            if position != 0:                      # close old leg
                pnl = position * (price - entry_price)
                fee_cost = abs(pnl) * fee
                balance += pnl - fee_cost
                total_fees += fee_cost
                if pnl > 0:
                    n_gains += 1
                    max_gain_trade = max(max_gain_trade, pnl)
                else:
                    n_losses += 1
                    max_loss_trade = min(max_loss_trade, pnl)
                trades += 1                        # count only when side flips
            if side != 0:                          # enter new leg
                entry_price = price
                notional = pos_size(price) * side
                fee_cost = abs(notional * price) * fee
                balance -= fee_cost
                total_fees += fee_cost
                position = notional
                if stopped:                        # stop counted above, do not count again
                    trades -= 1

        # mark-to-market at close
        mtm = position * (price - entry_price) if position else 0.0
        eq = balance + mtm
        max_bal = max(max_bal, eq)
        dd = (max_bal - eq) / max_bal
        max_dd = max(max_dd, dd)

    # final exit if still in position
    if position != 0:
        pnl = position * (df['close'].iloc[-1] - entry_price)
        balance += pnl - abs(pnl) * fee
        # ---- track trade stats ----
        if pnl > 0:
            n_gains += 1
            max_gain_trade = max(max_gain_trade, pnl)
        else:
            n_losses += 1
            max_loss_trade = min(max_loss_trade, pnl)
        # ---------------------------
        trades += 1

    final_eq = balance
    return {
        'lookback': lookback,
        'final_eq': final_eq,
        'return_pct': (final_eq / initial_margin - 1) * 100,
        'max_dd_pct': max_dd * 100,
        'trades': trades,
        'max_loss_trade': max_loss_trade,
        'max_gain_trade': max_gain_trade,
        'n_losses': n_losses,
        'n_gains': n_gains
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
        print(f"SMA{r['lookback']:3.0f} | "
              f"Final: {r['final_eq']:8.2f} USD | "
              f"Return: {r['return_pct']:+7.2f} % | "
              f"Max DD: {r['max_dd_pct']:6.2f} % | "
              f"Trades: {r['trades']:3.0f} | "
              f"Best: {r['max_gain_trade']:8.2f} | "
              f"Worst: {r['max_loss_trade']:8.2f} | "
              f"W: {r['n_gains']:3.0f} | "
              f"L: {r['n_losses']:3.0f}")
    print("="*95)

# ------------------------------------------------------------------
# 4. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_and_convert_to_daily()
    run_sma_cross(daily)
    
