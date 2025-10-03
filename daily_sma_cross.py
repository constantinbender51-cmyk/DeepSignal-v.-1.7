import pandas as pd
import numpy as np
from datetime import datetime
import time

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
# 2. 19-d vs 29-d SMA CROSSOVER  (DAILY)
# ------------------------------------------------------------------
def calculate_sma_crossovers(df):
    if df is None or len(df) < 29*24:  # at least 29 days of hourly data
        print(f"Need ≥29 days of hourly bars, got {len(df) if df is not None else 0}")
        return df, None, None

    # Resample to daily close
    daily = df['close'].resample('D').last().dropna()
    daily = daily.to_frame(name='close')

    daily['sma_19'] = daily['close'].rolling(window=19).mean()
    daily['sma_29'] = daily['close'].rolling(window=29).mean()
    daily['prev_sma_19'] = daily['sma_19'].shift(1)
    daily['prev_sma_29'] = daily['sma_29'].shift(1)

    bull = (daily['sma_19'] > daily['sma_29']) & (daily['prev_sma_19'] <= daily['prev_sma_29'])
    bear = (daily['sma_19'] < daily['sma_29']) & (daily['prev_sma_19'] >= daily['prev_sma_29'])

    return daily, bull, bear

# ------------------------------------------------------------------
# 3. BACK-TEST  (DAILY BARS)
# ------------------------------------------------------------------
def simulate_trading(daily, bull, bear):
    trades, equity = [], [10_000]
    pos, entry_p, entry_t = None, 0, None
    peak, max_dd = 10_000, 0

    # daily equity series
    daily_eq = daily['close'].copy()
    daily_eq[:] = 10_000

    for i in range(29, len(daily)):
        price = daily['close'].iloc[i]
        eq = equity[-1]

        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100)

        if bull.iloc[i] and pos != 'long':
            if pos == 'short':
                pnl = (entry_p - price) / entry_p
                eq += eq * pnl
                trades.append({'type':'short','side':'close','entry_time':entry_t,'exit_time':daily.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(daily.index[i]-entry_t).days,'equity_after':eq})
                equity.append(eq)
            pos, entry_p, entry_t = 'long', price, daily.index[i]

        elif bear.iloc[i] and pos != 'short':
            if pos == 'long':
                pnl = (price - entry_p) / entry_p
                eq += eq * pnl
                trades.append({'type':'long','side':'close','entry_time':entry_t,'exit_time':daily.index[i],
                               'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                               'duration_days':(daily.index[i]-entry_t).days,'equity_after':eq})
                equity.append(eq)
            pos, entry_p, entry_t = 'short', price, daily.index[i]
        else:
            equity.append(eq)

        daily_eq.iloc[i] = eq

    # final exit
    if pos:
        price = daily['close'].iloc[-1]
        pnl = (price - entry_p)/entry_p if pos == 'long' else (entry_p - price)/entry_p
        eq += eq * pnl; equity.append(eq)
        trades.append({'type':pos,'side':'close','entry_time':entry_t,'exit_time':daily.index[-1],
                       'entry_price':entry_p,'exit_price':price,'pnl_pct':pnl*100,
                       'duration_days':(daily.index[-1]-entry_t).days,'equity_after':eq})
        max_dd = max(max_dd, (peak-eq)/peak*100)

    return trades, equity, max_dd, daily_eq

# ------------------------------------------------------------------
# 4. ENHANCED RESULTS PRINTER
# ------------------------------------------------------------------
def print_trade_results(trades, equity_curve, max_drawdown, daily_equity):
    if not trades:
        print("No trades."); return

    # --- basic lists
    ret_pcts = np.array([t['pnl_pct'] for t in trades])
    winners  = ret_pcts[ret_pcts > 0]
    losers   = ret_pcts[ret_pcts <= 0]

    gross_win   = winners.sum() if len(winners) else 0
    gross_loss  = abs(losers.sum()) if len(losers) else 0
    profit_factor = gross_win / gross_loss if gross_loss else np.inf

    # --- annualised Sharpe & Sortino (assume 365 days, rf = 0)
    daily_ret = daily_equity.pct_change().dropna()
    sharpe   = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() else 0
    downside = daily_ret[daily_ret < 0].std()
    sortino  = (daily_ret.mean() / downside) * np.sqrt(365) if downside else np.inf

    # --- trade counts
    long_trades  = sum(1 for t in trades if t['type'] == 'long')
    short_trades = sum(1 for t in trades if t['type'] == 'short')

    # --- time in market
    total_mins = sum(t['duration_days']*24*60 for t in trades)
    first_day  = daily_equity.index[0]
    last_day   = daily_equity.index[-1]
    calendar_mins = (last_day - first_day).total_seconds() / 60 + 24*60
    time_in_mkt_pct = (total_mins / calendar_mins) * 100

    # --- print header
    print("="*100)
    print("TRADE RESULTS - 19-d vs 29-d SMA CROSSOVER  (ENHANCED METRICS)")
    print("="*100)

    # --- individual trades
    for t in trades:
        print(f"{t['type']:<7} {t['side']:<6} | {t['entry_time']} -> {t['exit_time']} | "
              f"P&L: {t['pnl_pct']:+.2f}% | Dur: {t['duration_days']:.1f}d | Eq: ${t['equity_after']:,.0f}")
        time.sleep(0.01)

    print("-"*100)

    # --- summary block
    final_eq = equity_curve[-1]
    total_ret = (final_eq / 10_000 - 1) * 100
    expectancy = ret_pcts.mean()
    print(f"Final equity:        ${final_eq:,.0f}")
    print(f"Total return:        {total_ret:+.2f}%")
    print(f"Max draw-down:       {max_drawdown:.2f}%")
    print(f"Sharpe ratio (365d): {sharpe:.2f}")
    print(f"Sortino ratio:       {sortino:.2f}")
    print(f"Profit factor:       {profit_factor:.2f}")
    print(f"Expectancy/trade:    {expectancy:+.2f}%")
    print(f"Win rate:            {len(winners)/len(trades)*100:.1f}%  ({len(winners)}/{len(trades)})")
    print(f"Avg trade:           {ret_pcts.mean():+.2f}%")
    print(f"Avg winner:          {winners.mean():+.2f}%")
    print(f"Avg loser:           {losers.mean():+.2f}%")
    print(f"Largest winner:      {winners.max():+.2f}%")
    print(f"Largest loser:       {losers.min():+.2f}%")
    print(f"Long trades:         {long_trades}")
    print(f"Short trades:        {short_trades}")
    print(f"Time in market:      {time_in_mkt_pct:.1f}%")
    print("="*100)

# ------------------------------------------------------------------
# 5. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_process_data()
    daily, bull, bear = calculate_sma_crossovers(df)
    trades, eq_curve, dd, daily_eq = simulate_trading(daily, bull, bear)
    print_trade_results(trades, eq_curve, dd, daily_eq)
