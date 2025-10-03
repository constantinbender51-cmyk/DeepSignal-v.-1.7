import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 1. load hourly csv → daily candles + 50-/200-MA
# ------------------------------------------------------------------
def load_daily(path='xbtusd_1h_8y.csv'):
    df = pd.read_csv(path)
    # ---- accept either 'open_time' or 'timestamp' ----------------
    time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

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
# 2. 50/200 cross strategy with 1 % stop – cash-sized positions
# ------------------------------------------------------------------
def run(daily, lev=1, fee=0.0025, stop=0.01, cash=100):
    pos = 0
    entry = None
    balance = cash
    trades = 0

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

        # ---- 3. enter / flip – size by available cash -------------
        if sig and sig != pos and balance > 0:
            if pos:                                 # close old
                pnl = pos*(r.close - entry)
                balance += pnl - abs(pnl)*fee
                trades += 1
            # --- cash-sized position ------------------------------
            max_size = balance / (r.close * (1 + fee))  # coins we can afford
            pos = sig * max_size                        # long or short that amount
            entry = r.close
            balance -= abs(pos * entry) * fee           # opening fee
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
# 3. generator that yields every trade (entry / exit / stop)
# ------------------------------------------------------------------
def run_detailed(daily, lev=1, fee=0.0025, stop=0.01, cash=100):
    pos = 0
    entry = None
    balance = cash

    prev_fast = daily['sma50'].shift(1)
    prev_slow = daily['sma200'].shift(1)

    for i, (date, r) in enumerate(daily.iterrows()):
        # ---- 1. stop exit ----------------------------------------
        if pos:
            st = entry*(1 - stop) if pos > 0 else entry*(1 + stop)
            if (pos > 0 and r.low <= st) or (pos < 0 and r.high >= st):
                pnl = pos*(st - entry)
                balance += pnl - abs(pnl)*fee
                yield {'date': date, 'side': 'STOP',
                       'price': st, 'pnl': pnl, 'balance': balance}
                pos = 0

        # ---- 2. cross signal -------------------------------------
        cross_up = (r.sma50 > r.sma200) and (prev_fast.iloc[i] <= prev_slow.iloc[i])
        cross_dn = (r.sma50 < r.sma200) and (prev_fast.iloc[i] >= prev_slow.iloc[i])
        sig = 1 if cross_up else (-1 if cross_dn else 0)

        # ---- 3. enter / flip – size by available cash -------------
        if sig and sig != pos and balance > 0:
            if pos:                                 # close old
                pnl = pos*(r.close - entry)
                balance += pnl - abs(pnl)*fee
                yield {'date': date, 'side': 'EXIT',
                       'price': r.close, 'pnl': pnl, 'balance': balance}
            # --- cash-sized position ------------------------------
            max_size = balance / (r.close * (1 + fee))
            pos = sig * max_size
            entry = r.close
            balance -= abs(pos * entry) * fee
            yield {'date': date, 'side': 'ENTRY',
                   'price': entry, 'pnl': 0, 'balance': balance}

    # ---- 4. final exit if still in market ------------------------
    if pos:
        pnl = pos*(daily['close'].iloc[-1] - entry)
        balance += pnl - abs(pnl)*fee
        yield {'date': daily.index[-1], 'side': 'FINAL_EXIT',
               'price': daily['close'].iloc[-1], 'pnl': pnl, 'balance': balance}

# ------------------------------------------------------------------
# 4. buy-and-hold benchmark
# ------------------------------------------------------------------
def buy_and_hold(daily, fee=0.0025, cash=100):
    first_close = daily['close'].iloc[0]
    last_close  = daily['close'].iloc[-1]
    shares = cash / (first_close * (1 + fee))
    final  = shares * last_close * (1 - fee)
    return {'final': final,
            'return_%': (final/cash - 1)*100,
            'trades': 2}

# ------------------------------------------------------------------
# 5. helper: every 50/200 cross
# ------------------------------------------------------------------
def crosses(daily):
    prev_fast = daily['sma50'].shift(1)
    prev_slow = daily['sma200'].shift(1)
    for date, r in daily.iterrows():
        cross_up = (r.sma50 > r.sma200) and (prev_fast.loc[date] <= prev_slow.loc[date])
        cross_dn = (r.sma50 < r.sma200) and (prev_fast.loc[date] >= prev_slow.loc[date])
        if cross_up or cross_dn:
            yield {'date': date,
                   'type': 'CROSS_UP' if cross_up else 'CROSS_DN',
                   'sma50': r.sma50,
                   'sma200': r.sma200,
                   'close': r.close}

# ------------------------------------------------------------------
# 6. run everything
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()

    # ---- summary -----------------------------------------------
    ma_result = run(daily)
    bh_result = buy_and_hold(daily)
    print('50-/200-MA cross:', ma_result)
    print('Buy-and-hold:   ', bh_result)

    # ---- every cross -------------------------------------------
    print('\n--- every 50/200 cross ---')
    for c in crosses(daily):
        print(f"{c['date'].date()}  {c['type']:<9}  "
              f"sma50={c['sma50']:.2f}  sma200={c['sma200']:.2f}  close={c['close']:.2f}")

    # ---- every trade -------------------------------------------
    print('\n--- every cross-trade ---')
    for t in run_detailed(daily):
        print(f"{t['date'].date()}  {t['side']:<10} "
              f"price={t['price']:.2f}  pnl={t['pnl']:.2f}  balance={t['balance']:.2f}")
