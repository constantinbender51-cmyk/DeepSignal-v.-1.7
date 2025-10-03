"""
45-pair power-of-two SMA scan on XBTUSD 1 h data
run time < 1 min on a laptop
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import itertools

# ---------- config ----------
FAST_MAX_LOG2 = 9          # 2^9 = 512
SLOW_MAX_LOG2 = 9          # 2^9 = 512
MIN_TRADES     = 20
RF             = 0.0       # risk-free rate
SLIPPAGE       = 0.0       # bps  (set to realistic value if wanted)
# ----------------------------

def load_data():
    df = pd.read_csv('xbtusd_1h_8y.csv')
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    return df[['close']]

def one_backtest(close: pd.Series, fast: int, slow: int):
    """Return dict of metrics for a single fast/slow pair."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    long  = (f > s) & (f.shift() <= s.shift())
    short = (f < s) & (f.shift() >= s.shift())
    # 1 = long, -1 = short, 0 = flat
    pos = pd.Series(0, index=close.index)
    state = 0
    for i in range(len(close)):
        if long.iloc[i]:
            state = 1
        elif short.iloc[i]:
            state = -1
        pos.iloc[i] = state
    # returns
    ret = pos.shift(1) * close.pct_change()
    daily_ret = (1 + ret).resample('1D').prod() - 1
    daily_ret = daily_ret.dropna()
    # metrics
    trades = (pos != pos.shift()).sum() // 2
    total_ret = (1 + ret).prod() - 1
    sharpe = (daily_ret.mean() - RF) / daily_ret.std() * np.sqrt(365) if daily_ret.std() else 0
    max_dd = ((1 + ret).cumprod().cummax() - (1 + ret).cumprod()).max()
    win = (ret > 0).sum()
    loss = (ret < 0).sum()
    pf = abs(ret[ret > 0].sum() / ret[ret < 0].sum()) if ret[ret < 0].sum() else np.inf
    return dict(fast=fast, slow=slow, trades=trades, total_ret=total_ret,
                sharpe=sharpe, max_dd=max_dd, win_rate=win/(win+loss) if (win+loss) else 0, pf=pf)

def main():
    close = load_data()['close']
    lengths = [2**i for i in range(FAST_MAX_LOG2 + 1)]          # 1,2,4…512
    results = []
    for fast, slow in itertools.combinations(lengths, 2):
        if fast >= slow:
            continue
        r = one_backtest(close, fast, slow)
        if r['trades'] >= MIN_TRADES:
            results.append(r)
    df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(df_res.head(10).round(3))

if __name__ == '__main__':
    main()
