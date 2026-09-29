#!/usr/bin/env python3
"""Recompute paper Table 2 from the final processed task outcomes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACTORS = {
    "qwen3-coder-30b": ("Qwen3Coder-30B", "q3c30b"),
    "kimi-k2.5": ("Kimi-K2.5", "kimi-k2.5"),
    "sonnet4.5": ("Sonnet-4.5", "sonnet-4.5"),
}
SETUPS = {
    "bashonly": "BashOnly",
    "atomic-common": "Atomic",
    "bash-search": "NLSearch",
    "bash-hypo": "HypoTrack",
    "bash-scratchpad": "Scratchpad",
    "py-interface": "Python",
}
KS = (5, 7, 9)
REPEATS = 10
INSTANCES = 65


def consistency(n: int, passed: int, k: int) -> float:
    return math.comb(passed, k) / math.comb(n, k) if passed >= k else 0.0


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["actor"], row["setup"])].append(row)
    summary = {}
    for actor in ACTORS:
        summary[actor] = {}
        for setup in SETUPS:
            group = groups[(actor, setup)]
            if len(group) != INSTANCES:
                raise ValueError(f"Expected {INSTANCES} tasks for {actor}/{setup}; found {len(group)}")
            summary[actor][setup] = {
                str(k): sum(consistency(r["n_repeats"], r["n_pass"], k) for r in group) / len(group)
                for k in KS
            }
    return summary


def markdown_table(summary: dict) -> str:
    lines = [
        "# Reproduced Table 2: Consistency across actor models",
        "",
        "Each value averages C(passes, k) / C(repeats, k) over all 65 tasks.",
        "Parentheses give the difference from the same actor's BashOnly value.",
        "",
        "| Actor | Setup | c@5 | c@7 | c@9 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for actor, (display, _) in ACTORS.items():
        baseline = summary[actor]["bashonly"]
        for setup, setup_display in SETUPS.items():
            cells = []
            for k in KS:
                value = summary[actor][setup][str(k)]
                cell = f"{value:.3f}"
                if setup != "bashonly":
                    cell += f" ({value - baseline[str(k)]:+.3f})"
                cells.append(cell)
            lines.append(f"| {display} | {setup_display} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output_dir or ROOT / "results/table2").resolve()
    curated = data_root / "table2/inputs.json"
    rows = json.loads(curated.read_text())["per_instance"]
    expected_groups = {(actor, setup, task["instance"])
                       for actor in ACTORS for setup in SETUPS for task in rows
                       if task["actor"] == actor and task["setup"] == setup}
    if len(rows) != len(expected_groups) or any(
        row["n_repeats"] != REPEATS or len(row["resolved_by_repeat"]) != REPEATS
        or row["n_pass"] != sum(row["resolved_by_repeat"])
        for row in rows
    ):
        raise ValueError("Invalid or duplicate Table 2 processed outcome rows")
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_instance.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "table2.md").write_text(markdown_table(summary))
    (output / "inputs_used.json").write_text(json.dumps(["table2/inputs.json"], indent=2) + "\n")
    print(f"Read {len(rows)} task outcomes; wrote Table 2 to {output}")


if __name__ == "__main__":
    main()
