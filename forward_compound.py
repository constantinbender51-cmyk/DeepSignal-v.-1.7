# forward_compound.py
import pandas as pd, json, random, time, uuid, os
from deepseek_signal import get_signal
from typing import List, Dict

CSV   = "xbtusd_1h_8y.csv"
OUT   = "compound_trades.csv"
SIZE  = 0.0001
FEE   = 0.00035          # half round-trip per side
SLIP  = 0.0001

class Slice:
    __slots__ = ("uid", "side", "entry_px", "stop", "target", "birth_idx", "entry_time")
    def __init__(self, side: str, entry: float, stop: float, target: float, idx: int, entry_time):
        self.uid, self.side, self.entry_px, self.stop, self.target, self.birth_idx, self.entry_time = \
            str(uuid.uuid4())[:6], side, entry, stop, target, idx, entry_time

def time_exit(bar: Dict, slc: Slice, max_hours: int = 24) -> bool:
    """True -> close the slice because 24 h have passed"""
    # bar['time'] and slc.entry_time are pandas.Timestamp objects
    return (bar['time'] - slc.entry_time).total_seconds() / 3600 >= max_hours

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

    # ---------- exit checks ----------
    exits = []
    for slc in slices:
        # 3a. normal price exit
        exit_px, hit_stop = bar_exit(bar, slc)
        # 3b. time exit
        if exit_px is None and time_exit(bar, slc):
            # close at open of this bar (or mid, or whatever you prefer)
            exit_px = bar['open']
            hit_stop = None          # mark as time-out
        if exit_px:                  # either price or time closed the slice
            pnl_pct = (exit_px - slc.entry_px)/slc.entry_px * (1 if slc.side=="buy" else -1) - FEE*2
            trades.append({
                "uid":slc.uid,
                "open_t":candles[slc.birth_idx]["time"],
                "close_t":bar["time"],
                "side":slc.side,
                "entry":slc.entry_px,
                "exit":exit_px,
                "hit_stop":hit_stop if hit_stop is not None else "time",
                "pnl_pct":pnl_pct
            })
            exits.append(slc)
    for slc in exits:
        slices.remove(slc)
    closed_cnt += len(exits)

    # ---------- new signal ----------
    last50 = [dict(time=c["time"].timestamp(),o=c["open"],h=c["high"],l=c["low"],c=c["close"],v=c["volume"])
              for c in candles[idx-50:idx]]
    action, stop, target = get_signal(last50)
    if action != "FLAT":
        entry = bar["open"] * (1 + 5/3600/100)
        slices.append(Slice(action.lower(), entry, stop, target, idx, bar["time"]))

        # ---------- logging ----------
        if idx % 100 == 0 or exits:
            print(f"[{bar['time']}] bar {idx}  net {net_pos(slices):+.4f}  slices {len(slices)}  "
                  f"new {action}  exits {len(exits)}")
        if closed_cnt and closed_cnt % 50 == 0:
            avg = sum(t["pnl_pct"] for t in trades) / closed_cnt
            print(f"---- expectancy after {closed_cnt} slices: {avg:.3%} ----")

        if closed_cnt >= 10:  # full target
            break

    pd.DataFrame(trades).to_csv(OUT, index=False)
    print("Compound back-test done:", OUT)
    # Print the contents of the generated CSV for verification
    if os.path.exists(OUT):
        print("\nContents of the generated CSV:")
        print(pd.read_csv(OUT))

    df_trades = pd.DataFrame(trades)

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
    print(f"win rate   : {wr:.1%}")
    print(f"avg win    : {win.mean():.2%}")
    print(f"avg loss   : {loss.mean():.2%}")
    print(f"expectancy : {exp:.2%}")
    print(f"median pnl : {median:.2%}")
    print(f"shar (trades): {shar:.2f}")
    print(f"biggest +  : {max_up:.2%}")
    print(f"biggest –  : {max_down:.2%}")
    print("=========================================")

if __name__ == "__main__":
    run()
