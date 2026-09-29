#!/usr/bin/env python3
"""Reproduce the paper results by claim from final processed data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
POINTS = {
    "atomic": ("table2", "table4", "table9", "table10", "figure6", "tables12_15"),
    "nlsearch": ("figure3", "table5", "table7", "tables12_15"),
    "python": ("figure4", "table8", "tables12_15"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path,
                        default=ROOT / "data/processed")
    parser.add_argument("--point", choices=("all", *POINTS), default="all",
                        help="Claim to reproduce (default: all)")
    args = parser.parse_args()
    data = args.processed_root.resolve()
    runs = (tuple(dict.fromkeys(name for group in POINTS.values() for name in group))
            if args.point == "all" else POINTS[args.point])
    for name in runs:
        command = [sys.executable, str(HERE / f"reproduce_{name}.py"),
                   "--data-root", str(data)]
        print(f"Reproducing {name}", flush=True)
        subprocess.run(command, check=True)
    print(f"All {len(runs)} reproduction scripts passed using {data}")


if __name__ == "__main__":
    main()
