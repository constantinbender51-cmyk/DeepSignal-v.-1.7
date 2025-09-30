import pandas as pd
import numpy as np
from datetime import datetime

# ---------- 1. DATA LOADING -------------------------------------------------
def load_and_process_data():
    """Load and process the CSV data with proper ISO8601 parsing"""
    try:
        df = pd.read_csv('xbtusd_1h_8y.csv')
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df['close_time'] = pd.to_datetime(df['close_time'], format='ISO8601')
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        print("Error: File 'xbtusd_1h_8y.csv' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ---------- 2. SMA CROSSOVER DETECTION --------------------------------------
def calculate_sma_crossovers(df):
    """Calculate 50-day and 200-day SMAs and their crossover signals"""
    if df is None or len(df) < 4800:
        print(f"Not enough data for SMA calculation. Need at least 4800 hours, have {len(df)}")
        return df, [], []

    df['sma_50_day']  = df['close'].rolling(window=1200).mean()   # 50*24
    df['sma_200_day'] = df['close'].rolling(window=4800).mean()   # 200*24

    df['prev_sma_50']  = df['sma_50_day'].shift(1)
    df['prev_sma_200'] = df['sma_200_day'].shift(1)

    bullish_cross = (df['sma_50_day'] > df['sma_200_day']) & (df['prev_sma_50'] <= df['prev_sma_200'])
    bearish_cross = (df['sma_50_day'] < df['sma_200_day']) & (df['prev_sma_50'] >= df['prev_sma_200'])

    return df, bullish_cross, bearish_cross

# ---------- 3. SIMPLE CROSSOVER BACK-TEST -----------------------------------
def simulate_trading(df, bullish_signals, bearish_signals):
    """Simulate trading every crossover (no AI filter)"""
    trades, equity_curve = [], [10_000]
    position, entry_p, entry_t = None, 0, None
    peak = 10_000
    max_dd = 0

    start_idx = 4800   # first index with valid 200-day SMA

    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        eq = equity_curve[-1]

        # --- Bullish crossover: go long ---
        if bullish_signals.iloc[i] and position != 'long':
            if position == 'short':        # close short
                pnl = (entry_p - price) / entry_p
                eq += eq * pnl
                trades.append({'type':'short_close','entry_time':entry_t,'exit_time':df.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(df.index[i]-entry_t).total_seconds()/86400,
                               'equity_after':eq})
                equity_curve.append(eq)
                peak = max(peak, eq)
                max_dd = max(max_dd, (peak - eq)/peak*100)

            position, entry_p, entry_t = 'long', price, df.index[i]

        # --- Bearish crossover: go short ---
        elif bearish_signals.iloc[i] and position != 'short':
            if position == 'long':         # close long
                pnl = (price - entry_p) / entry_p
                eq += eq * pnl
                trades.append({'type':'long_close','entry_time':entry_t,'exit_time':df.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(df.index[i]-entry_t).total_seconds()/86400,
                               'equity_after':eq})
                equity_curve.append(eq)
                peak = max(peak, eq)
                max_dd = max(max_dd, (peak - eq)/peak*100)

            position, entry_p, entry_t = 'short', price, df.index[i]

        else:   # no signal
            equity_curve.append(eq)

    # --- Final exit if still open ---
    if position:
        price = df['close'].iloc[-1]
        if position == 'long':
            pnl = (price - entry_p) / entry_p
        else:
            pnl = (entry_p - price) / entry_p
        eq += eq * pnl
        equity_curve.append(eq)
        trades.append({'type':f'{position}_close','entry_time':entry_t,'exit_time':df.index[-1],
                       'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                       'duration_days':(df.index[-1]-entry_t).total_seconds()/86400,
                       'equity_after':eq})
        max_dd = max(max_dd, (peak - eq)/peak*100)

    return trades, equity_curve, max_dd

# ---------- 4. RESULTS PRINTER ----------------------------------------------
def print_trade_results(trades, equity_curve, max_drawdown):
    if not trades:
        print("No trades were executed.")
        return

    print("="*100)
    print("TRADE RESULTS - PLAIN 50-DAY vs 200-DAY SMA CROSSOVER STRATEGY  (NO AI)")
    print("="*100)

    total_pnl_pct = 0
    wins = 0
    for t in trades:
        print(f"{t['type']:<12} | {t['entry_time']} -> {t['exit_time']} | "
              f"P&L: {t['pnl_pct']:+.2f}% | Equity: ${t['equity_after']:,.0f}")
        total_pnl_pct += t['pnl_pct']
        if t['pnl_pct'] > 0:
            wins += 1

    final_eq = equity_curve[-1]
    print("="*100)
    print(f"Final equity: ${final_eq:,.0f}  (Total return: {(final_eq/10_000-1)*100:+.2f}%)")
    print(f"Max draw-down: {max_drawdown:.2f}%")
    print(f"Win rate: {wins/len(trades)*100:.1f}%  ({wins}/{len(trades)} trades)")
    print("="*100)

# ---------- 5. ONE-CLICK RUN -----------------------------------------------
if __name__ == "__main__":
    df = load_and_process_data()
    df, bull, bear = calculate_sma_crossovers(df)
    trades, eq_curve, dd = simulate_trading(df, bull, bear)
    print_trade_results(trades, eq_curve, dd)
      
