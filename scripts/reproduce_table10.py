#!/usr/bin/env python3
"""Recompute Table 10 BashOnly-to-Atomic efficiency medians."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {"qwen3-coder-30b": "Qwen3Coder-30B", "kimi-k2.5": "Kimi-K2.5", "sonnet4.5": "Sonnet-4.5"}
SHORT = {"qwen": "qwen3-coder-30b", "kimi": "kimi-k2.5", "sonnet": "sonnet4.5"}
SETUPS = ("bashonly", "atomic-common")
METRICS = ("num_steps", "input_tokens", "output_tokens")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(trajectories: list[dict], atomicity: list[dict]) -> dict:
    groups = defaultdict(list)
    ratios = defaultdict(list)
    for row in trajectories:
        if row["actor"] not in ACTORS or row["setup"] not in SETUPS:
            raise ValueError(f"Unexpected actor/setup: {row}")
        if any(type(row[name]) is not int or row[name] < 0 for name in METRICS):
            raise ValueError(f"Invalid trajectory metric: {row}")
        groups[row["actor"], row["setup"]].append(row)
    for row in atomicity:
        if row["actor"] not in ACTORS or row["ops_per_step_no_cd"] < 0:
            raise ValueError(f"Invalid atomicity record: {row}")
        ratios[row["actor"]].append(row)
    expected_ratio_counts = {"qwen3-coder-30b": 621, "kimi-k2.5": 630, "sonnet4.5": 645}
    summary = {}
    for actor in ACTORS:
        ratio_rows = ratios[actor]
        if len(ratio_rows) != expected_ratio_counts[actor]:
            raise ValueError(f"Unexpected atomicity count for {actor}")
        summary[actor] = {
            "n_atomicity_trajectories": len(ratio_rows),
            "ops_per_step_median": statistics.median(
                row["ops_per_step_no_cd"] for row in ratio_rows
            ),
            "setups": {},
        }
        for setup in SETUPS:
            group = groups[actor, setup]
            if len(group) != 650 or len({(x["instance"], x["repeat"]) for x in group}) != 650:
                raise ValueError(f"Expected 650 distinct trajectories for {actor}/{setup}")
            summary[actor]["setups"][setup] = {
                "n_trajectories": len(group),
                **{name: statistics.median(row[name] for row in group) for name in METRICS},
            }
    return summary


def display(summary: dict, actor: str) -> tuple[list[float], list[str]]:
    item = summary[actor]
    bash, atomic = (item["setups"][setup] for setup in SETUPS)
    pairs = [
        (round(bash["num_steps"]), round(atomic["num_steps"])),
        (round(bash["input_tokens"] / 1000), round(atomic["input_tokens"] / 1000)),
        (round(bash["output_tokens"]), round(atomic["output_tokens"])),
    ]
    nums = [round(item["ops_per_step_median"], 2)]
    directions = []
    for old, new in pairs:
        nums += [old, new, round(abs(new - old) / old * 100)]
        directions.append("up" if new > old else "down")
    return nums, directions


def markdown(summary: dict) -> str:
    lines = [
        "# Reproduced Table 10: BashOnly to Atomic efficiency",
        "",
        "Steps and tokens are per-trajectory medians across 650 attempts per setup.",
        "Ops/Step is the BashOnly median from parsed trajectories after excluding navigation.",
        "Percentage changes use displayed rounded medians.",
        "",
        "| Actor | Ops/Step | Steps | Input tokens | Output tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for actor, label in ACTORS.items():
        nums, directions = display(summary, actor)
        cells = [f"{nums[0]:.2f}"]
        for index, suffix in enumerate(("", "K", "")):
            base = 1 + 3 * index
            old, new, percent = (int(value) for value in nums[base:base + 3])
            sign = "+" if directions[index] == "up" else "-"
            cells.append(f"{old}{suffix} → {new}{suffix} ({sign}{percent}%)")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table10")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output_dir or ROOT / "results/table10").resolve()
    efficiency = json.loads((data_root / "figure4/inputs.json").read_text())["trajectories"]
    trajectories = [{"actor": row["actor"], "setup": row["setup"],
                     "instance": row["instance"], "repeat": row["repeat"],
                     "num_steps": row["steps"], "input_tokens": row["input_tokens"],
                     "output_tokens": row["output_tokens"]}
                    for row in efficiency if row["setup"] in SETUPS]
    short = {"qwen": "qwen3-coder-30b", "kimi": "kimi-k2.5", "sonnet": "sonnet4.5"}
    ratios = json.loads((data_root / "table9/inputs.json").read_text())["trajectories"]
    atomicity = [{"actor": short[row["actor"]], "instance": row["instance"],
                  "repeat": row["repeat"], "ops_per_step_no_cd": row["ops_per_step_no_cd"]}
                 for row in ratios]
    summary = compute(trajectories, atomicity)
    write_json(output / "summary.json", summary)
    (output / "table10.md").write_text(markdown(summary))
    print(f"Wrote {output / 'table10.md'}")


if __name__ == "__main__":
    main()
