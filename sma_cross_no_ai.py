import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------------------------------
# 1. DATA LOADING  (unchanged)
# ------------------------------------------------------------------
def load_and_process_data():
    try:
        df = pd.read_csv('xbtusd_1h_8y.csv')
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        print("Error: 'xbtusd_1h_8y.csv' not found."); return None
    except Exception as e:
        print(f"Data load error: {e}"); return None

# ------------------------------------------------------------------
# 2. 200-hour vs 5-hour SMA CROSSOVER
# ------------------------------------------------------------------
def calculate_sma_crossovers(df):
    if df is None or len(df) < 200:
        print(f"Need ≥200 hourly bars, got {len(df) if df is not None else 0}")
        return df, None, None

    df['sma_5']   = df['close'].rolling(window=5).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    df['prev_sma_5']   = df['sma_5'].shift(1)
    df['prev_sma_200'] = df['sma_200'].shift(1)

    # Bullish: 5-h crosses above 200-h
    bull = (df['sma_5'] > df['sma_200']) & (df['prev_sma_5'] <= df['prev_sma_200'])
    # Bearish: 5-h crosses below 200-h
    bear = (df['sma_5'] < df['sma_200']) & (df['prev_sma_5'] >= df['prev_sma_200'])

    return df, bull, bear

# ------------------------------------------------------------------
# 3. SIMPLE LONG/SHORT BACK-TEST  (unchanged logic)
# ------------------------------------------------------------------
def simulate_trading(df, bull, bear):
    trades, equity = [], [10_000]
    pos, entry_p, entry_t = None, 0, None
    peak, max_dd = 10_000, 0

    start = 200   # first valid 200-h SMA
    for i in range(start, len(df)):
        price = df['close'].iloc[i]
        eq = equity[-1]

        if bull.iloc[i] and pos != 'long':
            if pos == 'short':          # close short
                pnl = (entry_p - price) / entry_p
                eq += eq * pnl
                trades.append({'type':'short_close','entry_time':entry_t,'exit_time':df.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(df.index[i]-entry_t).total_seconds()/86400,
                               'equity_after':eq})
                equity.append(eq); peak = max(peak, eq); max_dd = max(max_dd, (peak-eq)/peak*100)
            pos, entry_p, entry_t = 'long', price, df.index[i]

        elif bear.iloc[i] and pos != 'short':
            if pos == 'long':           # close long
                pnl = (price - entry_p) / entry_p
                eq += eq * pnl
                trades.append({'type':'long_close','entry_time':entry_t,'exit_time':df.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(df.index[i]-entry_t).total_seconds()/86400,
                               'equity_after':eq})
                equity.append(eq); peak = max(peak, eq); max_dd = max(max_dd, (peak-eq)/peak*100)
            pos, entry_p, entry_t = 'short', price, df.index[i]
        else:
            equity.append(eq)

    # final exit
    if pos:
        price = df['close'].iloc[-1]
        pnl = (price - entry_p)/entry_p if pos == 'long' else (entry_p - price)/entry_p
        eq += eq * pnl; equity.append(eq); max_dd = max(max_dd, (peak-eq)/peak*100)
        trades.append({'type':f'{pos}_close','entry_time':entry_t,'exit_time':df.index[-1],
                       'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                       'duration_days':(df.index[-1]-entry_t).total_seconds()/86400,
                       'equity_after':eq})
    return trades, equity, max_dd

# ------------------------------------------------------------------
# 4. RESULTS PRINTER  (unchanged)
# ------------------------------------------------------------------
def print_trade_results(trades, equity_curve, max_drawdown):
    if not trades:
        print("No trades."); return
    print("="*100)
    print("TRADE RESULTS - 200-h vs 5-h SMA CROSSOVER STRATEGY")
    print("="*100)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    for t in trades:
        print(f"{t['type']:<12} | {t['entry_time']} -> {t['exit_time']} | "
              f"P&L: {t['pnl_pct']:+.2f}% | Equity: ${t['equity_after']:,.0f}")
    final = equity_curve[-1]
    print("="*100)
    print(f"Final equity: ${final:,.0f}  (Return: {(final/10_000-1)*100:+.2f}%)")
    print(f"Max draw-down: {max_drawdown:.2f}%")
    print(f"Win rate: {wins/len(trades)*100:.1f}%  ({wins}/{len(trades)} trades)")
    print("="*100)

# ------------------------------------------------------------------
# 5. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_process_data()
    df, bull, bear = calculate_sma_crossovers(df)
    trades, eq_curve, dd = simulate_trading(df, bull, bear)
    print_trade_results(trades, eq_curve, dd)
