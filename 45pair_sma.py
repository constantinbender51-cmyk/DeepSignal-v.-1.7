"""
Brute-force 1-50 / 2-200 SMA grid
10 k$ cash, 10× leverage, instant liquidation ≤ 1 k$
Survivors-only output.
Vectorised per pair → ~ 1-2 min.
"""

import pandas as pd
import numpy as np
import itertools, time, os

CASH        = 10_000
LEV_MAX     = 10
LIQ_LEVEL   = 0.10 * CASH          # 1 000 $
FAST_RANGE  = range(1, 51)         # 1 … 50
SLOW_RANGE  = range(2, 201)        # 2 … 200
MIN_TRADES  = 10
OUT_CSV     = 'grid_survivors_10x_1_50_200.csv'

# ---------- I/O ----------
def load_close():
    df = pd.read_csv('xbtusd_1h_8y.csv')
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    return df['close']

# ---------- fast vectorised back-test ----------
def test_one(f: int, s: int, close: np.ndarray, index: pd.DatetimeIndex):
    """
    Return dict or None (liquidated / < MIN_TRADES).
    close : float ndarray length N
    """
    n = len(close)
    sma_f = pd.Series(close).rolling(f).mean().values
    sma_s = pd.Series(close).rolling(s).mean().values

    # 1 = long, -1 = short, 0 = flat
    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(1, n):
        if (sma_f[i] > sma_s[i]) and (sma_f[i-1] <= sma_s[i-1]):
            state = 1
        elif (sma_f[i] < sma_s[i]) and (sma_f[i-1] >= sma_s[i-1]):
            state = -1
        pos[i] = state

    # equity path with leverage + liquidation check
    equity = np.empty(n)
    equity[0] = CASH
    liquidated = False
    for i in range(1, n):
        ret = pos[i-1] * (close[i] / close[i-1] - 1)
        notional = min(LEV_MAX * equity[i-1], LEV_MAX * CASH)   # cap initial leverage
        dollar_ret = notional * ret
        equity[i] = equity[i-1] + dollar_ret
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
    daily_eq = pd.Series(equity, index=index).resample('D').last()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() else 0
    return dict(fast=f, slow=s, trades=trades,
                total_ret=total_ret, max_dd=dd, sharpe=sharpe,
                final_eq=equity[-1])

# ---------- main ----------
def main():
    close_sr = load_close()
    close = close_sr.values
    index = close_sr.index
    pairs = [(f, s) for f in FAST_RANGE for s in SLOW_RANGE if f < s]
    print(f'Testing {len(pairs):,} pairs with 10× leverage & liquidation …')
    t0 = time.time()
    survivors = []
    for f, s in pairs:
        res = test_one(f, s, close, index)
        if res is not None:
            survivors.append(res)
    df = pd.DataFrame(survivors).sort_values('sharpe', ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f'Done – {len(survivors)} / {len(pairs)} survived.  Saved → {OUT_CSV}')
    print('\nTop 10 by Sharpe:')
    print(df.head(10).round(3))

if __name__ == '__main__':
    main()
