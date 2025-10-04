import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 0. USER CONTROLS – change these four lines only
# ------------------------------------------------------------------
FAST_MA   = 35
SLOW_MA   = 190
LEVERAGE  = 4.0
STOP_FRAC = 0.045

# ------------------------------------------------------------------
# 1. load hourly csv → daily candles + fast-/slow-MA
# ------------------------------------------------------------------
def load_daily(path='xbtusd_1h_8y.csv'):
    df = pd.read_csv(path)
    time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    daily = (df.resample('D')
               .agg({'open': 'first',
                     'high': 'max',
                     'low':  'min',
                     'close':'last'})
               .dropna())

    daily[f'sma{FAST_MA}'] = daily['close'].rolling(FAST_MA).mean()
    daily[f'sma{SLOW_MA}'] = daily['close'].rolling(SLOW_MA).mean()
    return daily.dropna()

# ------------------------------------------------------------------
# 2. single engine – returns summary dict
# ------------------------------------------------------------------
def _engine(daily, lev=LEVERAGE, fee=0.0025, stop=STOP_FRAC, cash=100):
    pos = 0
    entry = None
    balance = cash
    trades = 0
    trade_log = []

    fast_col = f'sma{FAST_MA}'
    slow_col = f'sma{SLOW_MA}'
    prev_fast = daily[fast_col].shift(1)
    prev_slow = daily[slow_col].shift(1)

    for i, (date, r) in enumerate(daily.iterrows()):
        # 1. stop exit
        if pos:
            st = entry*(1 - stop) if pos > 0 else entry*(1 + stop)
            if (pos > 0 and r.low <= st) or (pos < 0 and r.high >= st):
                pnl = pos*(st - entry)
                balance += pnl - abs(pnl)*fee
                trade_log.append({'date': date, 'side': 'STOP',
                                  'price': st, 'pnl': pnl, 'balance': balance})
                pos = 0
                trades += 1

        # 2. cross signal
        cross_up = (r[fast_col] > r[slow_col]) and (prev_fast.iloc[i] <= prev_slow.iloc[i])
        cross_dn = (r[fast_col] < r[slow_col]) and (prev_fast.iloc[i] >= prev_slow.iloc[i])
        sig = 1 if cross_up else (-1 if cross_dn else 0)

        # 3. enter / flip – LEVERAGE-sized
        if sig and balance > 0:
            if pos:                         # close old
                pnl = pos*(r.close - entry)
                balance += pnl - abs(pnl)*fee
                trade_log.append({'date': date, 'side': 'EXIT',
                                  'price': r.close, 'pnl': pnl, 'balance': balance})
                trades += 1
            notional = balance * lev
            max_size = notional / (r.close * (1 + fee))
            pos = sig * max_size
            entry = r.close
            balance -= abs(pos * entry) * fee
            trade_log.append({'date': date, 'side': 'ENTRY',
                              'price': entry, 'pnl': 0, 'balance': balance})
            trades += 1

    # 4. final exit
    if pos:
        pnl = pos*(daily['close'].iloc[-1] - entry)
        balance += pnl - abs(pnl)*fee
        trade_log.append({'date': daily.index[-1], 'side': 'FINAL_EXIT',
                          'price': daily['close'].iloc[-1], 'pnl': pnl, 'balance': balance})
        trades += 1

    return {'final': balance,
            'return_%': (balance/cash - 1)*100,
            'trades': trades}

# ------------------------------------------------------------------
# 3. run one simulation
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()
    result = _engine(daily)
    print('One-run result:')
    for k, v in result.items():
        print(f'  {k}: {v}')
