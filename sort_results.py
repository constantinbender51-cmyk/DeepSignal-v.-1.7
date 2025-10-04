#!/usr/bin/env python3
"""
Sort the log lines that contain the back-test results by the
“final” equity value (first numeric field) in descending order
and print the top-50 rows in a nice table.

Usage:
    python sort_results.py x.txt
"""

import sys
import csv
from pathlib import Path

def parse_lines(path: Path):
    """
    Yield the CSV-like rows that appear *after* the line that starts
    with 'final,return_%...'.  We simply look for lines that contain
    six commas and can be parsed as floats.
    """
    with path.open() as fh:
        # fast-forward until we see the header line
        for line in fh:
            if line.strip().startswith("final,return_%"):
                break

        # now every subsequent line that looks like CSV is a result
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:        # expect 7 columns
                continue
            try:
                final = float(parts[0])
            except ValueError:
                continue
            yield parts                 # list of strings

def main():
    if len(sys.argv) != 2:
        print("Usage: python sort_results.py <logfile>")
        sys.exit(1)

    logfile = Path(sys.argv[1])
    if not logfile.exists():
        print(f"File not found: {logfile}")
        sys.exit(1)

    rows = list(parse_lines(logfile))
    if not rows:
        print("No result rows found.")
        sys.exit(0)

    # sort by the first column (final equity) descending
    rows.sort(key=lambda r: float(r[0]), reverse=True)

    # pretty print top 50
    writer = csv.writer(sys.stdout)
    writer.writerow(["final", "return_%", "trades", "fast", "slow", "stop", "leverage"])
    for row in rows[:50]:
        writer.writerow(row)

if __name__ == "__main__":
    main()
