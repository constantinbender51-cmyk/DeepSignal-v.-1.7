import pandas as pd

def load_daily(path='xbtusd_1h_8y.csv'):
    df = pd.read_csv(path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    daily = df.resample('D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    daily['sma50']  = daily['close'].rolling(50).mean()
    daily['sma200'] = daily['close'].rolling(200).mean()
    return daily.dropna()

def run(daily, lev=2, fee=0.0025, stop=0.01, cash=100):
    pos = 0
    entry = None
    balance = cash
    trades = 0

    for _, r in daily.iterrows():
        # 1. stop exit
        if pos:
            st = entry*(1-stop) if pos>0 else entry*(1+stop)
            if (pos>0 and r.low<=st) or (pos<0 and r.high>=st):
                pnl = pos*(st-entry)
                balance += pnl - abs(pnl)*fee
                pos = 0; trades += 1

        # 2. cross signal
        sig = 0
        if r.sma50 > r.sma200: sig = 1
        if r.sma50 < r.sma200: sig =-1

        # 3. enter/flip
        if sig and sig != pos:
            if pos:                                 # close old
                pnl = pos*(r.close-entry)
                balance += pnl - abs(pnl)*fee
                trades += 1
            pos = sig                               # open new
            entry = r.close
            balance -= abs(pos*entry)*fee           # open fee
            trades += 1

    # final exit
    if pos:
        pnl = pos*(r.close-entry)
        balance += pnl - abs(pnl)*fee
        trades += 1

    return {'final': balance, 'return_%': (balance/cash-1)*100, 'trades': trades}

if __name__ == '__main__':
    daily = load_daily()
    print(run(daily))
