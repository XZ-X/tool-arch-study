#!/usr/bin/env python3
"""Reproduce appendix Tables 12–15 from final processed inputs."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTORS = {"qwen3-coder-30b": ("Qwen3Coder-30B", "Qwen3 Coder"),
          "kimi-k2.5": ("Kimi-K2.5", "Kimi K2.5")}
DATASETS = ("SWE-bench Live", "SWE-bench Verified", "Feature Impl.", "Debugging")
ADDITIONAL_DATASETS = ("SWE-bench Verified tiny", "SWE-bench Pro tiny", "SWE-bench debug/stacktrace")
LABELS = {"12": "tab:generalization-pass5", "13": "tab:generalization-read-diversity",
          "14": "tab:generalization-input-tokens", "15": "tab:generalization-steps"}
PUBLIC_MANIFESTS = {
    "SWE-bench Verified": ("Verified", "tasks_verified.txt"),
    "Feature Impl.": ("Pro", "tasks_pro.txt"),
    "Debugging": ("Debugging", "tasks_debugging.txt"),
}
PUBLIC_SETUPS = {"bashonly": "BashOnly", "atomic-common": "Atomic"}


def read_json(path: Path):
    return json.loads(path.read_text())


def one_decimal(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def display(value: Decimal, percent: bool = False) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}" + ("%" if percent else "")


def consistency_percent(group: dict) -> float:
    n = group["n_repeats"]
    if n < 5:
        raise ValueError(f"Consistency@5 needs at least five repeats, got {n}")
    counts = group["success_counts"]
    return 100 * sum(math.comb(c, 5) / math.comb(n, 5) if c >= 5 else 0
                     for c in counts.values()) / len(counts)


def processed_consistency_inputs(data_root: Path) -> dict:
    """Count successes from normalized selected outcome files."""
    base = data_root / "tables12_15"
    result = {}
    for actor, (model, _) in ACTORS.items():
        result[actor] = {}
        for dataset, (folder, manifest) in PUBLIC_MANIFESTS.items():
            ids = [line.strip() for line in (base / "manifests" / manifest).read_text().splitlines()
                   if line.strip()]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate task in {manifest}")
            result[actor][dataset] = {}
            repeats = 10 if actor == "qwen3-coder-30b" and dataset == "SWE-bench Verified" else 5
            for setup, setup_folder in PUBLIC_SETUPS.items():
                group = base / "evaluation_outcomes" / folder / model / setup_folder
                files = sorted(group.glob("repeat_*.json"))
                expected = [group / f"repeat_{i}.json" for i in range(repeats)]
                if files != expected:
                    raise ValueError(f"Expected {repeats} numbered outcomes in {group}")
                counts = dict.fromkeys(ids, 0)
                for path in files:
                    report = read_json(path)
                    submitted = set(report["submitted_tasks"])
                    resolved = set(report["resolved_tasks"])
                    if not resolved <= submitted or not submitted <= set(ids):
                        raise ValueError(f"Invalid submitted/resolved tasks in {path}")
                    for task in resolved:
                        counts[task] += 1
                result[actor][dataset][setup] = {"n_repeats": repeats,
                                                  "success_counts": counts}
    return result


def compute(inputs: dict) -> tuple[dict, dict]:
    tables = {number: {} for number in LABELS}
    details = {number: {} for number in LABELS}
    for actor, (paper_name, generated_name) in ACTORS.items():
        base_read = inputs["live_read_diversity"][actor]["bashonly"]
        search_read = inputs["live_read_diversity"][actor]["bash-search"]
        # Figure 3 computes its Live bar after rounding each setup mean to 3 decimals.
        live_read_change = 100 * (round(search_read, 3) - round(base_read, 3)) / round(base_read, 3)
        read_deltas = [one_decimal(live_read_change)]
        for dataset in ADDITIONAL_DATASETS:
            base = inputs["additional_read_diversity"][f"{generated_name}|{dataset}|bashonly"]["mean"]
            search = inputs["additional_read_diversity"][f"{generated_name}|{dataset}|NL search"]["mean"]
            read_deltas.append(one_decimal(100 * (search - base) / base))

        live_base = inputs["live_consistency"][actor]["bashonly"]
        live_atomic = inputs["live_consistency"][actor]["atomic-common"]
        consistency_deltas = [one_decimal(100 * (live_atomic - live_base))]
        details["12"][paper_name] = {}
        for dataset in PUBLIC_MANIFESTS:
            groups = inputs["additional_consistency"][actor][dataset]
            base = consistency_percent(groups["bashonly"])
            atomic = consistency_percent(groups["atomic-common"])
            # The rebuttal's Pro cells subtract setup means displayed to one decimal.
            delta = one_decimal(float(one_decimal(atomic) - one_decimal(base)))
            consistency_deltas.append(delta)
            details["12"][paper_name][dataset] = {
                "repeats": groups["bashonly"]["n_repeats"],
                "tasks": len(groups["bashonly"]["success_counts"]),
                "bashonly_percent": base,
                "atomic_percent": atomic,
                "bashonly_display": str(one_decimal(base)),
                "atomic_display": str(one_decimal(atomic)),
                "delta_display": display(delta),
            }
        consistency_deltas.append(one_decimal(sum(consistency_deltas) / 4))
        tables["12"][paper_name] = [display(v) for v in consistency_deltas]

        read_deltas.append(one_decimal(sum(read_deltas) / 4))
        tables["13"][paper_name] = [display(v, True) for v in read_deltas]
        details["13"][paper_name] = [str(v) for v in read_deltas]

        for number, metric, scale in (("14", "input_tokens", 1_000_000), ("15", "steps", 1)):
            live = inputs["live_efficiency"][actor]
            deltas = [one_decimal((live["py-interface"][metric] - live["bashonly"][metric]) / scale)]
            for dataset in ADDITIONAL_DATASETS:
                base = inputs["additional_efficiency"][f"{generated_name}|{dataset}|bashonly"][metric]
                python = inputs["additional_efficiency"][f"{generated_name}|{dataset}|python interface"][metric]
                deltas.append(one_decimal((python - base) / scale))
            deltas.append(one_decimal(sum(deltas) / 4))
            tables[number][paper_name] = [display(v) for v in deltas]
            details[number][paper_name] = [str(v) for v in deltas]
    return tables, details


def markdown(tables: dict, details: dict) -> str:
    lines = ["# Appendix Tables 12–15", "",
             "Table 12 is calculated from per-task success counts for the selected evaluated attempts.", "",
             "Tables 13–15 use processed read-diversity and efficiency summaries. Overall is the equal-weight mean of the four displayed task deltas, rounded to one decimal.", ""]
    for number, label in LABELS.items():
        lines.extend([f"## Table {number} ({label})", "",
                      "| Actor | " + " | ".join(DATASETS) + " | Overall |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for actor in ACTORS:
            name = ACTORS[actor][0]
            lines.append("| " + name + " | " + " | ".join(tables[number][name]) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    input_path = data_root / "tables12_15/inputs.json"
    output_dir = (args.output_dir or ROOT / "results/tables12_15").resolve()
    inputs = read_json(input_path)
    inputs["additional_consistency"] = processed_consistency_inputs(data_root)
    tables, details = compute(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"tables": tables, "details": details}, indent=2) + "\n")
    (output_dir / "tables12_15.md").write_text(markdown(tables, details))
    print(f"Wrote {output_dir / 'tables12_15.md'}")


if __name__ == "__main__":
    main()
