import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 0. USER CONTROLS – change these four lines only
# ------------------------------------------------------------------
FAST_MA = 200         # short / fast window
SLOW_MA = 50          # long  / slow window
LEVERAGE = 1.0        # desired leverage (e.g. 3×)
STOP_FRAC = 1      # stop-loss fraction (e.g. 2 %  → 0.02)

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
# 2. single engine – returns (summary_dict, trade_list)
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

    summary = {'final': balance,
               'return_%': (balance/cash - 1)*100,
               'trades': trades}
    return summary, trade_log

# ------------------------------------------------------------------
# 3. thin wrappers
# ------------------------------------------------------------------
def run(daily, **kw):
    summary, _ = _engine(daily, **kw)
    return summary

def run_detailed(daily, **kw):
    _, trade_log = _engine(daily, **kw)
    yield from trade_log

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
# 5. helper: every fast/slow cross
# ------------------------------------------------------------------
def crosses(daily):
    fast_col = f'sma{FAST_MA}'
    slow_col = f'sma{SLOW_MA}'
    prev_fast = daily[fast_col].shift(1)
    prev_slow = daily[slow_col].shift(1)

    for date, r in daily.iterrows():
        cross_up = (r[fast_col] > r[slow_col]) and (prev_fast.loc[date] <= prev_slow.loc[date])
        cross_dn = (r[fast_col] < r[slow_col]) and (prev_fast.loc[date] >= prev_slow.loc[date])
        if cross_up or cross_dn:
            yield {'date': date,
                   'type': 'CROSS_UP' if cross_up else 'CROSS_DN',
                   fast_col: r[fast_col],
                   slow_col: r[slow_col],
                   'close': r.close}

# ------------------------------------------------------------------
# 6. run everything
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()

    # ---- summary -----------------------------------------------
    ma_result = run(daily)
    bh_result = buy_and_hold(daily)
    label = f'{FAST_MA}-/{SLOW_MA}-MA (×{LEVERAGE}, {STOP_FRAC*100:.0f}% stop):'
    print(f'{label:<35}', ma_result)
    print('Buy-and-hold:                      ', bh_result)

    # ---- every cross -------------------------------------------
    print(f'\n--- every {FAST_MA}/{SLOW_MA} cross ---')
    for c in crosses(daily):
        print(f"{c['date'].date()}  {c['type']:<9}  "
              f"sma{FAST_MA}={c[f'sma{FAST_MA}']:.2f}  "
              f"sma{SLOW_MA}={c[f'sma{SLOW_MA}']:.2f}  "
              f"close={c['close']:.2f}")

    # ---- every trade -------------------------------------------
    print(f'\n--- every cross-trade ---')
    for t in run_detailed(daily):
        print(f"{t['date'].date()}  {t['side']:<10} "
              f"price={t['price']:.2f}  pnl={t['pnl']:.2f}  balance={t['balance']:.2f}")
