#!/usr/bin/env python3
"""Recompute Table 7 precision and recall from final processed file sets."""

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
SETUPS = {
    "bashonly": "BashOnly",
    "atomic-common": "Atomic",
    "bash-search": "NLSearch",
    "bash-hypo": "HypoTrack",
    "bash-scratchpad": "ScratchPad",
    "py-interface": "Python",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(relevant: dict[str, list[str]], trajectories: list[dict]) -> tuple[list[dict], dict]:
    if len(relevant) != 38 or sum(map(len, relevant.values())) != 405:
        raise ValueError("Expected 38 tasks and 405 core files")
    reference = {instance: set(files) for instance, files in relevant.items()}
    if any(not files for files in reference.values()):
        raise ValueError("Core file set is empty for a selected task")
    grouped = defaultdict(list)
    per_trajectory = []
    seen = set()
    for row in trajectories:
        actor, setup, instance, repeat = (
            row["actor"], row["setup"], row["instance"], row["repeat"]
        )
        if actor not in ACTORS or setup not in SETUPS or instance not in reference:
            raise ValueError(f"Unexpected trajectory: {actor}/{setup}/{instance}")
        if type(repeat) is not int or not 0 <= repeat < 10:
            raise ValueError(f"Invalid repeat: {row}")
        key = actor, setup, instance, repeat
        if key in seen:
            raise ValueError(f"Duplicate trajectory: {key}")
        seen.add(key)
        read = set(row["files_read"])
        hit = len(read & reference[instance])
        record = {
            "actor": actor,
            "setup": setup,
            "instance": instance,
            "repeat": repeat,
            "n_files_read": len(read),
            "n_relevant_files": len(reference[instance]),
            "n_relevant_read": hit,
            "precision": hit / len(read) if read else 0.0,
            "recall": hit / len(reference[instance]),
        }
        per_trajectory.append(record)
        grouped[actor, setup].append(record)
    summary = {}
    for actor in ACTORS:
        summary[actor] = {}
        for setup in SETUPS:
            records = grouped[actor, setup]
            expected = 379 if (actor, setup) == ("kimi-k2.5", "bash-hypo") else 380
            if len(records) != expected:
                raise ValueError(f"Expected {expected} trajectories for {actor}/{setup}, got {len(records)}")
            summary[actor][setup] = {
                "n_trajectories": len(records),
                "precision": sum(row["precision"] for row in records) / len(records),
                "recall": sum(row["recall"] for row in records) / len(records),
            }
    return per_trajectory, summary


def markdown(summary: dict) -> str:
    lines = [
        "# Reproduced Table 7: Quality of explored context",
        "",
        "Each metric is the arithmetic mean of per-trajectory precision or recall.",
        "Parentheses give the difference from BashOnly before rounding.",
        "",
        "| Actor | Setup | Precision | Recall | N |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for actor, actor_label in ACTORS.items():
        baseline = summary[actor]["bashonly"]
        for setup, setup_label in SETUPS.items():
            row = summary[actor][setup]
            cells = []
            for metric in ("precision", "recall"):
                cell = f"{row[metric]:.3f}"
                if setup != "bashonly":
                    cell += f" ({row[metric] - baseline[metric]:+.3f})"
                cells.append(cell)
            lines.append(
                f"| {actor_label} | {setup_label} | {cells[0]} | {cells[1]} | {row['n_trajectories']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table7")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output_dir or ROOT / "results/table7").resolve()
    relevant = json.loads((data_root / "high_relevant_files/task_reference_sets.json").read_text())["instances"]
    read_sets = json.loads((data_root / "figure3/inputs.json").read_text())["read_sets"]
    trajectories = []
    for key, files in read_sets.items():
        actor, setup, instance, repeat = key.split("|")
        if instance in relevant:
            trajectories.append({"actor": actor, "setup": setup, "instance": instance,
                                 "repeat": int(repeat), "files_read": files})
    per_trajectory, summary = compute(relevant, trajectories)
    write_json(output / "per_trajectory.json", per_trajectory)
    write_json(output / "summary.json", summary)
    (output / "table7.md").write_text(markdown(summary))
    print(f"Wrote {len(per_trajectory)} trajectory rows and {output / 'table7.md'}")


if __name__ == "__main__":
    main()
