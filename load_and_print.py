import pandas as pd
from datetime import datetime
import itertools, csv, os, time

# ------------------------------------------------------------------
# 0. USER CONTROLS – change these four lines only
# ------------------------------------------------------------------
FAST_MA = 200         # short / fast window
SLOW_MA = 50          # long  / slow window
LEVERAGE = 1.0        # desired leverage (e.g. 3×)
STOP_FRAC = 1         # stop-loss fraction (e.g. 2 %  → 0.02)

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
# 6. exhaustive scan – FAST 200→50, SLOW 45→5, stop 0.01→0.20, lev 1→5
# ------------------------------------------------------------------
if __name__ == '__main__':
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **__: x          # fallback if tqdm not installed

    daily = load_daily()

    # --- parameter grids -------------------------------------------------------
    fast_grid = range(200, 45, -5)          # 200, 195, 190 … 50
    slow_grid = range( 45,  0, -5)          #  45, 40, 35 …  5
    stop_grid = [round(x,3) for x in
                 [x * 0.005 for x in range(2, 41)]]  # 0.01 … 0.20 step 0.005
    lev_grid  = range(1, 6)                 # 1, 2, 3, 4, 5

    total = len(fast_grid) * len(slow_grid) * len(stop_grid) * len(lev_grid)

    # --- output files ----------------------------------------------------------
    SUM_CSV = 'scan_summary.csv'
    TRD_CSV = 'scan_trades.csv'

    with open(SUM_CSV, 'w', newline='') as fs, \
         open(TRD_CSV, 'w', newline='') as ft:

        sum_writer = None
        trd_writer = None

        pbar = tqdm(itertools.product(fast_grid, slow_grid, stop_grid, lev_grid),
                    total=total, desc='scanning')

        for fast, slow, stop, lev in pbar:
            if fast <= slow:                # meaningless MA pair
                continue

            # override the global constants inside the engine
            FAST_MA, SLOW_MA, STOP_FRAC, LEVERAGE = fast, slow, stop, lev

            # recompute MAs on the daily frame
            daily_copy = daily.copy()
            daily_copy[f'sma{fast}'] = daily_copy['close'].rolling(fast).mean()
            daily_copy[f'sma{slow}'] = daily_copy['close'].rolling(slow).mean()
            daily_copy = daily_copy.dropna()

            summary, trade_log = _engine(daily_copy, lev=lev, stop=stop)

            # --- write summary row -------------------------------------------
            summary.update({'fast': fast, 'slow': slow,
                            'stop': stop, 'leverage': lev})
            if sum_writer is None:
                sum_writer = csv.DictWriter(fs, fieldnames=list(summary.keys()))
                sum_writer.writeheader()
            sum_writer.writerow(summary)

            # --- write individual trades -------------------------------------
            for tr in trade_log:
                tr.update({'fast': fast, 'slow': slow,
                           'stop': stop, 'leverage': lev})
                if trd_writer is None:
                    trd_writer = csv.DictWriter(ft, fieldnames=list(tr.keys()))
                    trd_writer.writeheader()
                trd_writer.writerow(tr)

    print('Scan finished →', os.path.abspath(SUM_CSV),
          '&', os.path.abspath(TRD_CSV))

    # ------------------------------------------------------------------
    # 7. slow-print the files (0.1 s between lines)
    # ------------------------------------------------------------------
    def _slow_print(path):
        with open(path) as f:
            for ln in f:
                print(ln.rstrip())
                time.sleep(0.1)

    _slow_print(SUM_CSV)
    print('\n' + '='*80 + '\n')
    _slow_print(TRD_CSV)
