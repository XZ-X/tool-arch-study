#!/usr/bin/env python3
"""Recompute Table 8 BashOnly/Python efficiency means."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {"qwen": "Qwen3Coder-30B", "kimi": "Kimi-K2.5", "sonnet": "Sonnet-4.5"}
SETUPS = {"bashonly": "BashOnly", "py_interface": "Python"}
METRICS = ("n_steps", "total_output", "total_observations", "output_per_step", "obs_per_step")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        if row["actor"] not in ACTORS or row["setup"] not in SETUPS:
            raise ValueError(f"Unexpected actor/setup: {row}")
        grouped[row["actor"], row["setup"]].append(row)
    summary = {}
    for actor in ACTORS:
        summary[actor] = {}
        for setup in SETUPS:
            group = grouped[actor, setup]
            if len(group) != 650 or len({(x["instance"], x["repeat"]) for x in group}) != 650:
                raise ValueError(f"Expected 650 distinct attempts for {actor}/{setup}")
            summary[actor][setup] = {
                "n_trajectories": len(group),
                **{name: sum(row[name] for row in group) / len(group) for name in METRICS},
            }
    return summary


def markdown(summary: dict) -> str:
    lines = [
        "# Reproduced Table 8: BashOnly versus Python efficiency",
        "",
        "Each cell is the rounded mean of a trajectory-level metric across 650 attempts.",
        "",
        "| Actor | Setup | Steps | Total-Out | Total-Obs | Out/Step | Obs/Step |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for actor, label in ACTORS.items():
        for setup, setup_label in SETUPS.items():
            values = summary[actor][setup]
            cells = [f"{round(values[name]):,}" for name in METRICS]
            lines.append(f"| {label} | {setup_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table8")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    curated = data_root / "table8/inputs.json"
    output = (args.output_dir or ROOT / "results/table8").resolve()
    summary = compute(json.loads(curated.read_text())["trajectories"])
    write_json(output / "summary.json", summary)
    (output / "table8.md").write_text(markdown(summary))
    print(f"Wrote {output / 'table8.md'}")


if __name__ == "__main__":
    main()
