#!/usr/bin/env python3
import sys, re
from pathlib import Path

inf_re = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+ $$inf$$ \s+')

def iter_results(path: Path):
    with path.open() as fh:
        for line in fh:
            if "final,return_%,trades,fast,slow,stop,leverage" in line:
                break
        for line in fh:
            line = inf_re.sub("", line.strip())
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                continue
            try:
                float(parts[0])
            except ValueError:
                continue
            yield parts

log = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("x.txt")
rows = sorted(iter_results(log), key=lambda r: float(r[0]), reverse=True)

print("final  return_%  trades  fast  slow  stop  leverage")
for r in rows[:50]:
    print(" ".join(f"{v:>8}" for v in r))
