#!/usr/bin/env python3
"""Recompute Table 4 consistency from final processed evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {"qwen3-coder-30b": "Qwen3Coder-30B", "kimi-k2.5": "Kimi-K2.5"}
SETUPS = {"bashonly": "BashOnly", "atomic-common": "Atomic"}
KS = (5, 7, 9)
TASKS = 65


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def consistency(n: int, passed: int, k: int) -> float:
    return math.comb(passed, k) / math.comb(n, k) if n >= k and passed >= k else 0.0


def compute(runs: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for run in runs:
        key = (run["actor"], run["setup"], run["target_repeats"])
        if run["actor"] not in ACTORS or run["setup"] not in SETUPS or run["target_repeats"] not in (10, 30):
            raise ValueError(f"Unexpected run: {key}")
        grouped[key].append(run)

    rows = []
    summary = {actor: {} for actor in ACTORS}
    for actor in ACTORS:
        for repeats in (10, 30):
            summary[actor][str(repeats)] = {}
            for setup in SETUPS:
                group = sorted(grouped[(actor, setup, repeats)], key=lambda run: run["repeat"])
                if [run["repeat"] for run in group] != list(range(repeats)):
                    raise ValueError(f"Missing or duplicate repeat in {actor}/{setup}/{repeats}")
                instances = set(group[0]["submitted_ids"])
                if len(instances) != TASKS:
                    raise ValueError(f"Expected {TASKS} tasks in {actor}/{setup}/{repeats}")
                if any(set(run["submitted_ids"]) != instances for run in group):
                    raise ValueError(f"Task set differs across repeats in {actor}/{setup}/{repeats}")

                group_rows = []
                for instance in sorted(instances):
                    outcomes = [instance in run["resolved_ids"] for run in group]
                    n = repeats
                    passed = sum(outcomes)
                    scores = {str(k): consistency(n, passed, k) for k in KS}
                    record = {
                        "actor": actor,
                        "setup": setup,
                        "target_repeats": repeats,
                        "instance": instance,
                        "outcomes_by_repeat": outcomes,
                        "n_counted": n,
                        "n_pass": passed,
                        "consistency_at_k": scores,
                    }
                    rows.append(record)
                    group_rows.append(record)
                summary[actor][str(repeats)][setup] = {
                    "n_instances": len(group_rows),
                    "n_counted_attempts": sum(row["n_counted"] for row in group_rows),
                    "consistency_at_k": {
                        str(k): sum(row["consistency_at_k"][str(k)] for row in group_rows) / len(group_rows)
                        for k in KS
                    },
                }
    return rows, summary


def markdown_table(summary: dict) -> str:
    lines = [
        "# Reproduced Table 4: Consistency robustness under additional repeats",
        "",
        "Values average C(passes,k) / C(submitted attempts,k) across 65 tasks; unresolved attempts are failures.",
        "Deltas use unrounded Atomic and BashOnly values.",
        "",
        "| Actor | Repeats | c@5 Bash | Atomic | Δ | c@7 Bash | Atomic | Δ | c@9 Bash | Atomic | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for actor, label in ACTORS.items():
        for repeats in (10, 30):
            bash = summary[actor][str(repeats)]["bashonly"]["consistency_at_k"]
            atomic = summary[actor][str(repeats)]["atomic-common"]["consistency_at_k"]
            cells = []
            for k in KS:
                b, a = bash[str(k)], atomic[str(k)]
                cells += [f"{b:.3f}", f"{a:.3f}", f"{a - b:+.3f}"]
            lines.append(f"| {label} | {repeats} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/table4")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    curated = data_root / "table4/inputs.json"
    output = (args.output_dir or ROOT / "results/table4").resolve()
    rows, summary = compute(json.loads(curated.read_text())["runs"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "per_instance.json", rows)
    write_json(output / "summary.json", summary)
    (output / "table4.md").write_text(markdown_table(summary))
    print(f"Wrote {len(rows)} task rows and {output / 'table4.md'}")


if __name__ == "__main__":
    main()
