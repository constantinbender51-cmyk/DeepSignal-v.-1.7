# forward_compound.py
import pandas as pd, json, random, time, uuid, os
from datetime import datetime, timedelta
from deepseek_signal import get_signal
from typing import List, Dict

CSV   = "xbtusd_1h_8y.csv"
OUT   = "compound_trades.csv"
SIZE  = 0.0001
FEE   = 0.00035          # half round-trip per side
SLIP  = 0.0001
MAX_HOLD_HOURS = 24      # 24-hour auto close

class Slice:
    __slots__ = ("uid", "side", "entry_px", "stop", "target", "birth_idx", "birth_time")
    def __init__(self, side: str, entry: float, stop: float, target: float, idx: int, birth_time):
        self.uid, self.side, self.entry_px, self.stop, self.target, self.birth_idx, self.birth_time = \
            str(uuid.uuid4())[:6], side, entry, stop, target, idx, birth_time

def bar_exit(bar: Dict, slc: Slice) -> tuple:
    """Return (exit_px, hit_stop_bool) if tripped, else (None,None)"""
    stop_px  = slc.entry_px * (1 - slc.stop/100) if slc.side=="buy" else slc.entry_px * (1 + slc.stop/100)
    tgt_px   = slc.entry_px * (1 + slc.target/100) if slc.side=="buy" else slc.entry_px * (1 - slc.target/100)
    if slc.side=="buy":
        if bar["low"] <= stop_px:  return stop_px - SLIP, True
        if bar["high"]>= tgt_px:   return tgt_px + SLIP, False
    else:
        if bar["high"]>= stop_px:  return stop_px + SLIP, True
        if bar["low"] <= tgt_px:   return tgt_px - SLIP, False
    return None, None

def check_24h_exit(bar: Dict, slc: Slice) -> tuple:
    """Check if 24 hours have passed since slice creation"""
    current_time = bar["time"]
    if isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)
    
    time_diff = current_time - slc.birth_time
    if time_diff.total_seconds() >= MAX_HOLD_HOURS * 3600:
        # Auto-close at current bar's open price with slip
        exit_px = bar["open"] * (1 + SLIP if slc.side == "buy" else 1 - SLIP)
        return exit_px, False  # False means it's not a stop loss exit
    return None, None

def net_pos(slices: List[Slice]) -> float:
    return sum(SIZE * (1 if s.side=="buy" else -1) for s in slices)

def run():
    df = pd.read_csv(CSV, parse_dates=["open_time"]).rename(columns={"open_time":"time"})
    candles = df.to_dict("records")
    start = random.randint(50, len(candles)-5000)   # leave runway
    slices: List[Slice] = []
    trades: List[Dict]  = []
    closed_cnt = 0

    for idx in range(start, len(candles)):
        bar = candles[idx]

        # ---------- exit checks (lose-first) ----------
        exits = []
        for slc in slices:
            # Check regular exit conditions first
            exit_px, hit_stop = bar_exit(bar, slc)
            
            # If no regular exit, check 24-hour auto-close
            if exit_px is None:
                exit_px, hit_stop = check_24h_exit(bar, slc)
            
            if exit_px is not None:
                pnl_pct = (exit_px - slc.entry_px)/slc.entry_px * (1 if slc.side=="buy" else -1) - FEE*2
                exit_reason = "stop" if hit_stop else ("target" if not hit_stop and exit_px != bar["open"] * (1 + SLIP if slc.side == "buy" else 1 - SLIP) else "24h_auto")
                
                trades.append({
                    "uid": slc.uid, "open_t": candles[slc.birth_idx]["time"], "close_t": bar["time"],
                    "side": slc.side, "entry": slc.entry_px, "exit": exit_px, 
                    "hit_stop": hit_stop, "exit_reason": exit_reason, "pnl_pct": pnl_pct,
                    "hold_hours": (bar["time"] - slc.birth_time).total_seconds() / 3600 if hasattr(bar["time"], 'timestamp') else 0
                })
                exits.append(slc)
        
        for slc in exits:  
            slices.remove(slc)
        closed_cnt += len(exits)

        # ---------- new signal (ONLY if flat) ----------
        if not slices:                       # <── single-position guard
            last50 = [dict(time=c["time"].timestamp(),o=c["open"],h=c["high"],l=c["low"],c=c["close"],v=c["volume"])
                      for c in candles[idx-50:idx]]
            action, stop, target, reason = get_signal(last50)    
            print("Action: ", action, "stop: ", stop, "target:", target, "reason: ", reason)
            
            if action != "FLAT":
                entry = bar["open"] * (1 + 5/3600/100)   # 5-sec slip
                slices.append(Slice(action.lower(), entry, stop, target, idx, bar["time"]))

        # ---------- logging ----------
        if idx % 100 == 0 or exits:
            print(f"[{bar['time']}] bar {idx}  net {net_pos(slices):+.4f}  slices {len(slices)}  "
                  f"new {action if not slices else 'WAIT'}  exits {len(exits)}")
            
            # Log slices approaching 24-hour limit
            for slc in slices:
                current_time = bar["time"]
                if isinstance(current_time, str):
                    current_time = pd.to_datetime(current_time)
                
                time_diff = current_time - slc.birth_time
                hours_held = time_diff.total_seconds() / 3600
                if hours_held > 20:  # Warn when approaching 24 hours
                    print(f"  WARNING: Slice {slc.uid} has been held for {hours_held:.1f} hours")
        
        if closed_cnt and closed_cnt % 50 == 0:
            avg = sum(t["pnl_pct"] for t in trades) / closed_cnt
            print(f"---- expectancy after {closed_cnt} slices: {avg:.3%} ----")

        if closed_cnt >= 20:  # full target
            break

    pd.DataFrame(trades).to_csv(OUT, index=False)
    print("Compound back-test done:", OUT)
    
    # Print the contents of the generated CSV for verification
    if os.path.exists(OUT):
        print("\nContents of the generated CSV:")
        print(pd.read_csv(OUT))

    df_trades = pd.DataFrame(trades)

    # Additional analysis for 24-hour exits
    auto_exits = df_trades[df_trades["exit_reason"] == "24h_auto"]
    regular_exits = df_trades[df_trades["exit_reason"] != "24h_auto"]
    
    pnl = df_trades["pnl_pct"]
    win      = pnl[pnl > 0]
    loss     = pnl[pnl <= 0]
    wr       = len(win) / len(pnl)
    exp      = wr * win.mean() + (1 - wr) * loss.mean()
    median   = pnl.median()
    shar     = pnl.mean() / pnl.std() if pnl.std() else 0
    max_up   = pnl.max()
    max_down = pnl.min()

    print("\n==========  BACK-TEST SUMMARY  ==========")
    print(f"trades     : {len(df_trades)}")
    print(f"24h auto exits: {len(auto_exits)} ({len(auto_exits)/len(df_trades):.1%})")
    print(f"win rate   : {wr:.1%}")
    print(f"avg win    : {win.mean():.2%}")
    print(f"avg loss   : {loss.mean():.2%}")
    print(f"expectancy : {exp:.2%}")
    print(f"median pnl : {median:.2%}")
    print(f"shar (trades): {shar:.2f}")
    print(f"biggest +  : {max_up:.2%}")
    print(f"biggest –  : {max_down:.2%}")
    
    if len(auto_exits) > 0:
        auto_pnl = auto_exits["pnl_pct"]
        auto_wr = len(auto_pnl[auto_pnl > 0]) / len(auto_pnl)
        print(f"\n24h Auto-Close Performance:")
        print(f"  Win rate: {auto_wr:.1%}")
        print(f"  Avg PnL: {auto_pnl.mean():.2%}")
    
    print("=========================================")

if __name__ == "__main__":
    run()
