import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 1. load daily candles (unchanged)
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
# 2. original run() – unchanged
# ------------------------------------------------------------------
def run(daily, lev=2, fee=0.0025, stop=0.01, cash=100):
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
# 3. NEW: generator that yields every trade with P&L
# ------------------------------------------------------------------
def run_detailed(daily, lev=2, fee=0.0025, stop=0.01, cash=100):
    """
    Yields one dict per trade (entry or exit) with keys:
    date, side, price, pnl, balance
    """
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

        # ---- 2. cross signal (only first flip bar) ---------------
        cross_up = (r.sma50 > r.sma200) and (prev_fast.iloc[i] <= prev_slow.iloc[i])
        cross_dn = (r.sma50 < r.sma200) and (prev_fast.iloc[i] >= prev_slow.iloc[i])
        sig = 1 if cross_up else (-1 if cross_dn else 0)

        # ---- 3. enter / flip --------------------------------------
        if sig and sig != pos:
            if pos:                                 # close old
                pnl = pos*(r.close - entry)
                balance += pnl - abs(pnl)*fee
                yield {'date': date, 'side': 'EXIT',
                       'price': r.close, 'pnl': pnl, 'balance': balance}
            pos = sig                               # open new
            entry = r.close
            balance -= abs(pos*entry)*fee           # open fee
            yield {'date': date, 'side': 'ENTRY',
                   'price': entry, 'pnl': 0, 'balance': balance}

    # ---- 4. final exit if still in market ------------------------
    if pos:
        pnl = pos*(daily['close'].iloc[-1] - entry)
        balance += pnl - abs(pnl)*fee
        yield {'date': daily.index[-1], 'side': 'FINAL_EXIT',
               'price': daily['close'].iloc[-1], 'pnl': pnl, 'balance': balance}

# ------------------------------------------------------------------
# 4. buy-and-hold benchmark (unchanged)
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
# 5. run everything and print every cross-trade
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()

    # original summary
    ma_result = run(daily)
    bh_result = buy_and_hold(daily)
    print('50-/200-MA cross:', ma_result)
    print('Buy-and-hold:   ', bh_result)

    # NEW: print every single trade
    print('\n--- every cross-trade ---')
    for t in run_detailed(daily):
        print(f"{t['date'].date()}  {t['side']:<10} "
              f"price={t['price']:.2f}  pnl={t['pnl']:.2f}  balance={t['balance']:.2f}")
