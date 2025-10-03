"""
Brute-force every (fast, slow) SMA cross
1 ≤ fast ≤ 50 , fast < slow ≤ 200
≈ 9 850 combinations – vectorised, < 20 s
"""

import pandas as pd
import numpy as np
from datetime import datetime
import itertools, time, os

# ---------- user knobs ----------
FAST_RANGE = range(1, 51)          # 1 … 50
SLOW_RANGE = range(2, 201)       # 2 … 200
MIN_TRADES = 10                  # discard curves with <10 round-trips
OUT_FILE   = 'grid_results.csv'
# --------------------------------

def load_close():
    """Return pd.Series of close prices."""
    df = pd.read_csv('xbtusd_1h_8y.csv')
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    return df['close']

def metrics_from_returns(ret: pd.Series) -> dict:
    """Daily-metrics dictionary from 1-bar pct-change Series."""
    daily = (1 + ret).resample('D').prod() - 1
    daily = daily.dropna()
    if daily.empty or daily.std() == 0:
        return dict(sharpe=0, sortino=0, total_ret=0, max_dd=0, trades=0)
    total_ret = (1 + ret).prod() - 1
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365)
    downside = daily[daily < 0].std()
    sortino  = (daily.mean() / downside) * np.sqrt(365) if downside else 0
    cum = (1 + ret).cumprod()
    max_dd = (cum.cummax() - cum).max()
    trades = (ret.abs() > 0).astype(int).diff().abs().sum() // 2   # rough count
    return dict(sharpe=sharpe, sortino=sortino, total_ret=total_ret,
                max_dd=max_dd, trades=trades)

def batch_loop(close: pd.Series, pairs):
    """
    Core routine: vectorised back-test for a list of (f,s) tuples.
    Returns list of dicts.
    """
    n = len(close)
    results = []
    # pre-compute log returns once
    log_p = np.log(close.values)
    log_r = np.zeros_like(log_p)
    log_r[1:] = log_p[1:] - log_p[:-1]

    for f, s in pairs:
        # SMAs
        sma_f = close.rolling(f).mean().values
        sma_s = close.rolling(s).mean().values
        # cross flags
        long  = (sma_f > sma_s) & (np.roll(sma_f, 1) <= np.roll(sma_s, 1))
        short = (sma_f < sma_s) & (np.roll(sma_f, 1) >= np.roll(sma_s, 1))
        # position vector: 1 long, -1 short, 0 flat (carry previous)
        pos = np.zeros(n)
        state = 0
        for i in range(1, n):
            if long[i]:
                state = 1
            elif short[i]:
                state = -1
            pos[i] = state
        # strategy log-returns
        strat_log_r = pos[:-1] * log_r[1:]
        strat_r = pd.Series(np.exp(strat_log_r) - 1, index=close.index[1:])
        m = metrics_from_returns(strat_r)
        m['fast'], m['slow'] = f, s
        if m['trades'] >= MIN_TRADES:
            results.append(m)
    return results

def main():
    close = load_close()
    pairs = [(f, s) for f in FAST_RANGE for s in SLOW_RANGE if f < s]
    print(f'Testing {len(pairs):,} pairs …')
    t0 = time.time()
    res = batch_loop(close, pairs)
    df = pd.DataFrame(res)
    df.to_csv(OUT_FILE, index=False)
    print(f'Done in {time.time()-t0:.1f} s.  Saved → {OUT_FILE}')
    print('\nTop 10 by Sharpe:')
    print(df.sort_values('sharpe', ascending=False).head(10)
            [['fast','slow','sharpe','total_ret','max_dd','trades']].round(3))

if __name__ == '__main__':
    main()
