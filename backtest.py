import pandas as pd
import json
import random
import os
from datetime import timedelta
from deepseek_signal import get_signal

# --- Global Constants ---
CSV_PATH = "xbtusd_1h_8y.csv"   # 70k candles
OUT_PATH = "backtest_trades.csv"
SIZE_BTC = 0.0001
FEE      = 0.0007          # 0.02 % maker + 0.05 % taker + buffer
SLIP     = 0.0001          # 1 bp slippage

def bar_exit(bar: pd.Series, side: str, stop: float, target: float, entry_px: float):
    """
    Determines if a trade would have been closed within the same bar
    due to a stop-loss or take-profit.
    
    Args:
        bar (pd.Series): The current price bar (candle).
        side (str): The trade side ("long" or "short").
        stop (float): The stop-loss percentage.
        target (float): The take-profit percentage.
        entry_px (float): The price at which the trade was entered.

    Returns:
        tuple: (exit_price, exit_time, hit_stop_bool)
    """
    # Calculate the exact stop and target prices
    stop_px  = entry_px * (1 + stop / 100) if side == "long" else entry_px * (1 - stop / 100)
    tgt_px   = entry_px * (1 + target / 100) if side == "long" else entry_px * (1 - target / 100)

    # Log the prices for clarity
    print(f"    [LOG] Calculated Stop-Loss Price: {stop_px:.2f} | Take-Profit Price: {tgt_px:.2f}")

    if side == "long":
        # Check for stop-loss first
        if bar["low"] <= stop_px:
            print(f"    [LOG] Bar low ({bar['low']:.2f}) hit stop-loss ({stop_px:.2f}). Exiting trade.")
            return stop_px - SLIP, bar["time"], True
        # Check for take-profit
        if bar["high"] >= tgt_px:
            print(f"    [LOG] Bar high ({bar['high']:.2f}) hit take-profit ({tgt_px:.2f}). Exiting trade.")
            return tgt_px + SLIP, bar["time"], False
    else:  # short
        # Check for stop-loss first
        if bar["high"] >= stop_px:
            print(f"    [LOG] Bar high ({bar['high']:.2f}) hit stop-loss ({stop_px:.2f}). Exiting trade.")
            return stop_px + SLIP, bar["time"], True
        # Check for take-profit
        if bar["low"] <= tgt_px:
            print(f"    [LOG] Bar low ({bar['low']:.2f}) hit take-profit ({tgt_px:.2f}). Exiting trade.")
            return tgt_px - SLIP, bar["time"], False
    
    print("    [LOG] Trade remains open at the end of the bar.")
    return None, None, None  # still open

def run():
    """Main backtesting loop with rigorous logging."""
    print("=" * 80)
    print(" " * 25 + "STARTING BACKTEST PROCESS" + " " * 26)
    print("=" * 80)

    # Step 1: Data Loading
    print(f"[STEP 1] Loading historical data from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["open_time"])
        df = df.rename(columns={"open_time": "time"})
        candles = df[["time", "open", "high", "low", "close", "volume"]].to_dict("records")
        print(f"[LOG] Successfully loaded {len(candles)} candles.")
    except FileNotFoundError:
        print(f"[ERROR] The file '{CSV_PATH}' was not found. Please ensure it is in the same directory.")
        return
    
    # Step 2: Data Shuffling
    print(f"[STEP 2] Shuffling candles for randomized testing.")
    random.shuffle(candles)
    print(f"[LOG] Candles have been shuffled. Backtest will run on a random sample.")

    # Step 3: Main Backtest Loop
    print(f"[STEP 3] Entering main backtest loop. Simulating trades...")
    trades = []
    trade_count = 0
    for i, cand in enumerate(candles):
        # Warm-up period
        if i < 50:
            if i == 0:
                print(f"[LOG] Skipping first 50 candles for warm-up period...")
            continue
        
        # Log the current candle being processed
        print("-" * 50)
        print(f"[LOG] Processing candle {i} at time: {cand['time']}")

        # Get signal from the trading algorithm
        last_50 = [dict(time=c["time"].timestamp(), o=c["open"], h=c["high"],
                        l=c["low"], c=c["close"], v=c["volume"])
                   for c in candles[i-50:i]]
        print(f"  [LOG] Calling deepseek_signal with previous 50 candles...")
        action, stop, target = get_signal(last_50)
        print(f"  [LOG] Signal received: action='{action}', stop={stop}%, target={target}%")

        # Check if a trade should be opened
        if action == "FLAT":
            print(f"  [LOG] Signal is 'FLAT'. No trade will be entered for this bar.")
            continue

        # Trade entry logic
        trade_count += 1
        entry_px = cand["open"] * (1 + 5/3600/100)
        print(f"  [LOG] Trade #{trade_count} opened at: {cand['time']}")
        print(f"  [LOG] Side: {action} | Entry Price: {entry_px:.2f} | Stop: {stop}% | Target: {target}%")
        
        # Check for in-bar exit
        print(f"  [LOG] Checking for in-bar exit on the same candle...")
        exit_px, exit_time, hit_stop = bar_exit(cand, action.lower(), stop, target, entry_px)

        if exit_px:  # closed in the same bar
            pnl_pct = (exit_px - entry_px) / entry_px * (1 if action == "BUY" else -1) - FEE
            print(f"  [LOG] Trade closed in the same bar.")
            print(f"  [LOG] Exit Price: {exit_px:.2f} | Close Time: {exit_time}")
            print(f"  [LOG] Result: {'STOP-LOSS' if hit_stop else 'TAKE-PROFIT'}")
            print(f"  [LOG] PnL %: {pnl_pct*100:.4f}% (including fees)")
            
            trades.append({
                "open_time": cand["time"], "close_time": exit_time,
                "side": action, "entry": entry_px, "exit": exit_px,
                "stop": stop, "target": target, "hit_stop": hit_stop,
                "pnl_pct": pnl_pct
            })
        
        # Stop condition
        if len(trades) >= 10:
            print("\n[STEP 4] Trade limit reached. Stopping backtest.")
            break

    # Step 5: Save Results
    print(f"\n[STEP 5] Backtest complete. Saving {len(trades)} trades to: {OUT_PATH}")
    pd.DataFrame(trades).to_csv(OUT_PATH, index=False)
    
    # Step 6: Print Final Results
    print("\n[STEP 6] Final Results:")
    print("=" * 80)
    if os.path.exists(OUT_PATH):
        result_df = pd.read_csv(OUT_PATH)
        print(result_df.to_string(index=False))
    else:
        print("No trades were generated in the backtest")
    print("=" * 80)

if __name__ == "__main__":
    run()
