def run():
    df = pd.read_csv(CSV, parse_dates=["open_time"]).rename(columns={"open_time":"time"})
    candles = df.to_dict("records")
    start = random.randint(50, len(candles)-5000)
    slices: List[Slice] = []
    trades: List[Dict]  = []
    closed_cnt = 0

    for idx in range(start, len(candles)):
        bar = candles[idx]

        # ---------- exit checks (lose-first) ----------
        exits = []
        for slc in slices:
            # 1. regular stop/target
            exit_px, hit_stop = bar_exit(bar, slc)
            # 2. 24-hour auto-close
            if exit_px is None:
                exit_px, hit_stop = check_24h_exit(bar, slc)

            if exit_px is not None:
                pnl_pct = (exit_px - slc.entry_px)/slc.entry_px * (1 if slc.side=="buy" else -1) - FEE*2
                exit_reason = ("stop" if hit_stop else
                              ("target" if not hit_stop and exit_px != bar["open"] * (1 + SLIP if slc.side == "buy" else 1 - SLIP)
                               else "24h_auto"))

                trades.append({
                    "uid": slc.uid, "open_t": candles[slc.birth_idx]["time"], "close_t": bar["time"],
                    "side": slc.side, "entry": slc.entry_px, "exit": exit_px,
                    "hit_stop": hit_stop, "exit_reason": exit_reason, "pnl_pct": pnl_pct,
                    "hold_hours": (bar["time"] - slc.birth_time).total_seconds() / 3600
                })
                exits.append(slc)

        for slc in exits:
            slices.remove(slc)
        closed_cnt += len(exits)

        # ---------- new signal (ONLY if flat) ----------
        if not slices:                       # <── single-position guard
            last50 = [dict(time=c["time"].timestamp(), o=c["open"], h=c["high"],
                           l=c["low"], c=c["close"], v=c["volume"])
                      for c in candles[idx-50:idx]]
            action, stop, target, reason = get_signal(last50)
            print("Action:", action, "stop:", stop, "target:", target, "reason:", reason)

            if action != "FLAT":
                entry = bar["open"] * (1 + 5/3600/100)   # 5-sec slip
                slices.append(Slice(action.lower(), entry, stop, target, idx, bar["time"]))

        # ---------- logging ----------
        if idx % 100 == 0 or exits:
            print(f"[{bar['time']}] bar {idx}  net {net_pos(slices):+.4f}  slices {len(slices)}  "
                  f"exits {len(exits)}")
            for slc in slices:
                hours_held = (bar["time"] - slc.birth_time).total_seconds() / 3600
                if hours_held > 20:
                    print(f"  WARNING: Slice {slc.uid} has been held for {hours_held:.1f} hours")

        if closed_cnt and closed_cnt % 50 == 0:
            avg = sum(t["pnl_pct"] for t in trades) / closed_cnt
            print(f"---- expectancy after {closed_cnt} slices: {avg:.3%} ----")

        if closed_cnt >= 20:
            break

    # ---------- summary ----------
    pd.DataFrame(trades).to_csv(OUT, index=False)
    print("Compound back-test done:", OUT)
    if os.path.exists(OUT):
        print("\nContents of the generated CSV:")
        print(pd.read_csv(OUT))

    df_trades = pd.DataFrame(trades)
    ...  # (rest of your stats block remains unchanged)
