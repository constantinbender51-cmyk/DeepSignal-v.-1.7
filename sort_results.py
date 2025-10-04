#!/usr/bin/env python3
import sys, re
from pathlib import Path

inf_re = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+ $$inf$$ \s+')

def iter_results(path: Path):
    with path.open() as fh:
        # --- 1. look for the header ----------------------------------------
        for line in fh:
            if "final,return_%,trades,fast,slow,stop,leverage" in line:
                print("DEBUG: header found")          # <──
                break
        else:                                         # header never seen
            print("DEBUG: header line missing – no data will be read")
            return

        # --- 2. process every subsequent line ------------------------------
        for lineno, line in enumerate(fh, start=1):
            line = inf_re.sub("", line.strip())
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                print(f"DEBUG: line {lineno} has {len(parts)} fields (need 7)")
                continue
            try:
                float(parts[0])
            except ValueError:
                print(f"DEBUG: line {lineno} first field not a float: {parts[0]!r}")
                continue
            yield parts

log = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("x.txt")
rows = sorted(iter_results(log), key=lambda r: float(r[0]), reverse=True)

print("final  return_%  trades  fast  slow  stop  leverage")
for r in rows[:50]:
    print(" ".join(f"{v:>8}" for v in r))
