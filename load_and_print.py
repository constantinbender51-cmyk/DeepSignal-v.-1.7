import pandas as pd

# ------------------------------------------------------------------
# 1. load hourly csv → daily candles + 50-/200-MA
# ------------------------------------------------------------------
def load_daily(path='xbtusd_1h_8y.csv'):
    df = pd.read_csv(path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)

    daily = (df.resample('D')
               .agg({'open': 'first',
                     'high': 'max',
                     'low':  'min',
                     'close':'last'})
               .dropna())
    daily['sma50']  = daily['close'].rolling(50).mean()
    daily['sma200'] = daily['close'].rolling(200).mean()
    return daily.dropna()

# ------------------------------------------------------------------
# 2. single 50/200 cross strategy with 1 % stop
# ------------------------------------------------------------------
def run(daily, lev=2, fee=0.0025, stop=0.01, cash=100):
    pos = 0
    entry = None
    balance = cash
    trades = 0

    # previous-bar MAs for true cross detection
    prev_fast = daily['sma50'].shift(1)
    prev_slow = daily['sma200'].shift(1)

    for i, (_, r) in enumerate(daily.iterrows()):
        # ---- 1. stop exit ----------------------------------------
        if pos:
            st = entry*(1 - stop) if pos > 0 else entry*(1 + stop)
            if (pos > 0 and r.low <= st) or (pos < 0 and r.high >= st):
                pnl = pos*(st - entry)
                balance += pnl - abs(pnl)*fee
                pos = 0
                trades += 1

        # ---- 2. cross signal (only first flip bar) ---------------
        cross_up = (r.sma50 > r.sma200) and (prev_fast.iloc[i] <= prev_slow.iloc[i])
        cross_dn = (r.sma50 < r.sma200) and (prev_fast.iloc[i] >= prev_slow.iloc[i])
        sig = 1 if cross_up else (-1 if cross_dn else 0)

        # ---- 3. enter / flip --------------------------------------
        if sig and sig != pos:
            if pos:                                 # close old
                pnl = pos*(r.close - entry)
                balance += pnl - abs(pnl)*fee
                trades += 1
            pos = sig                               # open new
            entry = r.close
            balance -= abs(pos*entry)*fee           # open fee
            trades += 1

    # ---- 4. final exit if still in market ------------------------
    if pos:
        pnl = pos*(daily['close'].iloc[-1] - entry)
        balance += pnl - abs(pnl)*fee
        trades += 1

    return {'final': balance,
            'return_%': (balance/cash - 1)*100,
            'trades': trades}

# ------------------------------------------------------------------
# 3. fire once
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()
    print(run(daily))
