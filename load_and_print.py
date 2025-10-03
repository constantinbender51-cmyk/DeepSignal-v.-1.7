import pandas as pd
import time

# ------------------------------------------------------------------
# 1. LOAD CSV & CONVERT TO DAILY CANDLES
# ------------------------------------------------------------------
def load_and_convert_to_daily(path='xbtusd_1h_8y.csv'):
    try:
        df = pd.read_csv(path)
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)

        # Resample to daily candles
        daily = df.resample('D').agg({
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return daily
    except FileNotFoundError:
        print("Error: 'xbtusd_1h_8y.csv' not found."); return None
    except Exception as e:
        print(f"Data load error: {e}"); return None

# ------------------------------------------------------------------
# 2. PRINT DAILY CANDLES WITH 0.01 sec DELAY
# ------------------------------------------------------------------
def print_daily_candles(daily_df):
    if daily_df is None or daily_df.empty:
        print("No daily candles to display.")
        return

    for ts, row in daily_df.iterrows():
        print(f"{ts.date()} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.0f}")
        time.sleep(0.01)

# ------------------------------------------------------------------
# 3. 200-SMA CROSS STRATEGY (5× lev, 100 USD margin)
# ------------------------------------------------------------------
def run_sma_cross(daily_df, leverage=5, initial_margin=100, fee=0.0005):
    """Simulates a 200-SMA cross strategy on the daily candles."""
    if daily_df is None or daily_df.empty:
        print("No data – nothing to back-test.")
        return

    # --- prepare series -------------------------------------------------
    close = daily_df['close']
    sma200 = close.rolling(200).mean()
    daily_df = daily_df.copy()
    daily_df['sma200'] = sma200
    daily_df.dropna(inplace=True)          # need 200 pts before first trade

    # --- state variables ------------------------------------------------
    position = 0.0          # number of contracts / coins held
    balance = initial_margin
    max_bal = balance
    max_dd = 0.0
    side = 0                # 1 long, -1 short, 0 flat

    # --- helper ---------------------------------------------------------
    def pos_size(price):
        """How many coins we can buy/sell with full leverage."""
        return balance * leverage / price

    # --- main loop ------------------------------------------------------
    for ts, row in daily_df.iterrows():
        price = row['close']
        sma = row['sma200']

        # signal
        prev_side = side
        if price > sma and side != 1:      # cross up → go long
            side = 1
        elif price < sma and side != -1:   # cross down → go short
            side = -1
        else:                              # no signal
            side = side

        # execute trade only on side change
        if side != prev_side:
            # close old position
            if position != 0:
                pnl = position * (price - entry_price)
                balance += pnl - abs(pnl) * fee       # pay fee on exit
                position = 0.0
            # open new position
            if side != 0:
                entry_price = price
                position = pos_size(price) * side
                balance -= abs(position * price) * fee  # pay fee on entry

        # mark-to-market PnL while in trade
        mtm_pnl = 0.0
        if position != 0:
            mtm_pnl = position * (price - entry_price)

        # draw-down
        eq = balance + mtm_pnl
        max_bal = max(max_bal, eq)
        dd = (max_bal - eq) / max_bal
        max_dd = max(max_dd, dd)

        # pretty print
        print(f"{ts.date()} | "
              f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} "
              f"V:{row['volume']:.0f} | SMA:{sma:.2f} | "
              f"{'LONG' if side==1 else 'SHORT' if side==-1 else 'FLAT'} | "
              f"Eq:{eq:.2f} USD")

        time.sleep(0.01)

    # --- summary --------------------------------------------------------
    print("\n===== STRATEGY SUMMARY =====")
    print(f"Initial margin : {initial_margin:,.2f} USD")
    print(f"Final equity   : {balance + (position*(daily_df['close'].iloc[-1] - entry_price) if position else 0):,.2f} USD")
    print(f"Total return   : {((balance/initial_margin - 1)*100):,.2f} %")
    print(f"Max draw-down  : {max_dd*100:.2f} %")
    print("=============================")

# ------------------------------------------------------------------
# 4. ONE-CLICK RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    daily = load_and_convert_to_daily()
    run_sma_cross(daily)
