#!/usr/bin/env python3
"""Recompute Table 5 from final processed early-action distances."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {
    "qwen3-coder-30b": "Qwen3Coder-30B",
    "kimi-k2.5": "Kimi-K2.5",
    "sonnet4.5": "Sonnet-4.5",
}
SETUPS = {"bash-search": "NLSearch", "bashonly": "BashOnly"}
METRICS = ("jaccard_distance", "levenshtein_distance")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        key = row["actor"], row["setup"]
        if key[0] not in ACTORS or key[1] not in SETUPS:
            raise ValueError(f"Unexpected actor/setup: {key}")
        if row["n_repeats"] != 10:
            raise ValueError(f"Expected ten repeats for {key}/{row['instance']}")
        if any(not 0 <= row[metric] <= 1 for metric in METRICS):
            raise ValueError(f"Distance out of range for {key}/{row['instance']}")
        groups[key].append(row)
    summary = {}
    for actor in ACTORS:
        summary[actor] = {}
        for setup in SETUPS:
            group = groups[actor, setup]
            if len(group) != 65 or len({row["instance"] for row in group}) != 65:
                raise ValueError(f"Expected 65 distinct tasks for {actor}/{setup}")
            summary[actor][setup] = {
                "n_tasks": len(group),
                **{metric: sum(row[metric] for row in group) / len(group) for metric in METRICS},
            }
    return summary


def markdown(summary: dict) -> str:
    lines = [
        "# Reproduced Table 5: Diversity of early search actions",
        "",
        "Each value is the arithmetic mean of 65 task-level distances.",
        "",
        "| Actor | Setup | Jaccard | Levenshtein |",
        "| --- | --- | ---: | ---: |",
    ]
    for actor, label in ACTORS.items():
        for setup, setup_label in SETUPS.items():
            values = summary[actor][setup]
            lines.append(
                f"| {label} | {setup_label} | {values['jaccard_distance']:.3f} | "
                f"{values['levenshtein_distance']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table5")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    curated = data_root / "table5/inputs.json"
    output = (args.output_dir or ROOT / "results/table5").resolve()
    summary = compute(json.loads(curated.read_text())["task_scores"])
    write_json(output / "summary.json", summary)
    (output / "table5.md").write_text(markdown(summary))
    print(f"Wrote {output / 'table5.md'}")


if __name__ == "__main__":
    main()
