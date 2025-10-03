"""
Daily-bar SMA grid 1-50 / 2-200  (daily SMAs, daily signals)
10 k$ cash, 10× lev, liquidation ≤ 1 k$
Survivors-only csv + top-10 Sharpe
"""

import pandas as pd
import numpy as np
import itertools, time

CASH       = 10_000
LEV_MAX    = 10
LIQ_LEVEL  = 0.10 * CASH
FAST_RANGE = range(1, 51)
SLOW_RANGE = range(2, 201)
MIN_TRADES = 10
IN_FILE    = 'xbtusd_1h_8y.csv'
OUT_FILE   = 'daily_sma_survivors_10x.csv'

def load_daily():
    """Resample hourly → daily close."""
    df = pd.read_csv(IN_FILE)
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
    df.set_index('open_time', inplace=True)
    daily_close = df['close'].resample('D').last().dropna()
    return daily_close

def daily_backtest(close: pd.Series, f: int, s: int):
    """Return dict or None (liquidated / < MIN_TRADES)."""
    n = len(close)
    sma_f = close.rolling(f).mean()
    sma_s = close.rolling(s).mean()

    # 1 = long, -1 = short, 0 = flat
    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(1, n):
        if (sma_f.iloc[i] > sma_s.iloc[i]) and (sma_f.iloc[i-1] <= sma_s.iloc[i-1]):
            state = 1
        elif (sma_f.iloc[i] < sma_s.iloc[i]) and (sma_f.iloc[i-1] >= sma_s.iloc[i-1]):
            state = -1
        pos[i] = state

    # equity path
    equity = np.empty(n)
    equity[0] = CASH
    liquidated = False
    for i in range(1, n):
        ret = pos[i-1] * (close.iloc[i] / close.iloc[i-1] - 1)
        notional = min(LEV_MAX * equity[i-1], LEV_MAX * CASH)
        equity[i] = equity[i-1] + notional * ret
        if equity[i] <= LIQ_LEVEL:
            liquidated = True
            break
    if liquidated:
        return None

    trades = (np.diff(pos) != 0).sum() // 2
    if trades < MIN_TRADES:
        return None

    total_ret = (equity[-1] / CASH - 1)
    dd = ((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)).max()
    daily_ret = pd.Series(equity, index=close.index).pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() else 0
    return dict(fast=f, slow=s, trades=trades,
                total_ret=total_ret, max_dd=dd, sharpe=sharpe,
                final_eq=equity[-1])

def main():
    close = load_daily()
    pairs = [(f, s) for f in FAST_RANGE for s in SLOW_RANGE if f < s]
    print(f'Testing {len(pairs):,} daily-bar pairs …')
    t0 = time.time()
    survivors = [r for r in (daily_backtest(close, f, s) for f, s in pairs) if r is not None]
    df = pd.DataFrame(survivors).sort_values('sharpe', ascending=False)
    df.to_csv(OUT_FILE, index=False)
    print(f'Done – {len(survivors)} / {len(pairs)} survived.  Saved → {OUT_FILE}')
    print('\nTop 10 by Sharpe:')
    print(df.head(10).round(3))

if __name__ == '__main__':
    main()
