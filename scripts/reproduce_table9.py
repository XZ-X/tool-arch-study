#!/usr/bin/env python3
"""Recompute Table 9 BashOnly operation counts and compounding."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {"qwen": "Qwen3Coder-30B", "kimi": "Kimi-K2.5", "sonnet": "Sonnet-4.5"}
CATEGORIES = ("n_read", "n_search", "n_edit", "n_write", "n_run", "n_git", "n_misc")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        if row["actor"] not in ACTORS:
            raise ValueError(f"Unexpected actor: {row['actor']}")
        if row["n_bash_steps"] <= 0:
            raise ValueError(f"Invalid Bash step count: {row}")
        expected_ops = sum(row[name] for name in CATEGORIES)
        if expected_ops != row["n_atomic_ops"] - row["n_cd_ops"]:
            raise ValueError(f"Operation total differs for {row['actor']}/{row['instance']}/{row['repeat']}")
        expected_ratio = expected_ops / row["n_bash_steps"]
        if abs(expected_ratio - row["ops_per_step_no_cd"]) > 0.000501:
            raise ValueError(f"Ops/Step differs for {row['actor']}/{row['instance']}/{row['repeat']}")
        grouped[row["actor"]].append(row)
    summary = {}
    counts = {"qwen": 621, "kimi": 630, "sonnet": 645}
    for actor in ACTORS:
        group = grouped[actor]
        if len(group) != counts[actor] or len({(x["instance"], x["repeat"]) for x in group}) != counts[actor]:
            raise ValueError(f"Expected {counts[actor]} distinct parsed trajectories for {actor}")
        summary[actor] = {
            "n_trajectories": len(group),
            "category_means": {
                name: statistics.mean(row[name] for row in group) for name in CATEGORIES
            },
            "ops_per_step_median": statistics.median(
                row["ops_per_step_no_cd"] for row in group
            ),
        }
    return summary


def markdown(summary: dict) -> str:
    lines = [
        "# Reproduced Table 9: BashOnly operation compounding",
        "",
        "Operation columns are per-trajectory means. Ops/Step is the median per-trajectory ratio after excluding navigation operations.",
        "",
        "| Actor | READ | SEARCH | EDIT | WRITE | RUN | GIT | MISC | Ops/Step | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for actor, label in ACTORS.items():
        values = summary[actor]
        cells = [f"{values['category_means'][name]:.1f}" for name in CATEGORIES]
        cells += [f"{values['ops_per_step_median']:.3f}", str(values["n_trajectories"])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table9")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    curated = data_root / "table9/inputs.json"
    output = (args.output_dir or ROOT / "results/table9").resolve()
    summary = compute(json.loads(curated.read_text())["trajectories"])
    write_json(output / "summary.json", summary)
    (output / "table9.md").write_text(markdown(summary))
    print(f"Wrote {output / 'table9.md'}")


if __name__ == "__main__":
    main()
