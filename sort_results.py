#!/usr/bin/env python3
import sys, csv, re
from pathlib import Path

inf_re = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+
 $$inf$$ \s+')

def iter_results(path: Path):
    with path.open() as fh:
        # fast-forward until we see the header
        for line in fh:
            if "final,return_%,trades,fast,slow,stop,leverage" in line:
                break

        for line in fh:
            line = inf_re.sub("", line.strip())   # drop timestamp/[inf]
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                continue
            try:
                final = float(parts[0])
            except ValueError:
                continue
            yield parts

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python sort_results.py <logfile>")
    log = Path(sys.argv[1])
    if not log.exists():
        sys.exit(f"file not found: {log}")

    rows = list(iter_results(log))
    if not rows:
        sys.exit("no result rows found.")

    rows.sort(key=lambda r: float(r[0]), reverse=True)

    writer = csv.writer(sys.stdout)
    writer.writerow(["final", "return_%", "trades", "fast", "slow", "stop", "leverage"])
    writer.writerows(rows[:50])

if __name__ == "__main__":
    main()
