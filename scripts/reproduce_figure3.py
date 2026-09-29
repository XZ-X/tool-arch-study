#!/usr/bin/env python3
"""Reproduce Figure 3 from final processed file-read and solution scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = ("qwen3-coder-30b", "kimi-k2.5", "sonnet4.5")
ACTOR_LABELS = ("Q3C", "Kimi", "Sonnet")
SETUPS = ("bashonly", "atomic-common", "bash-search", "bash-hypo", "bash-scratchpad", "py-interface")
SETUP_LABELS = ("BashOnly", "Atomic", "NLSearch", "HypoTrack", "ScratchPad", "Python")
METRICS = ("read_diversity", "solution_diversity")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)


def compute_read_diversity(read_sets: dict) -> tuple[list[dict], list[str]]:
    grouped = defaultdict(dict)
    for key, files in read_sets.items():
        actor, setup, instance, repeat_text = key.split("|")
        repeat = int(repeat_text)
        if actor not in ACTORS or setup not in SETUPS or repeat not in range(10):
            raise ValueError(f"Unexpected trajectory key: {key}")
        grouped[(actor, setup, instance)][repeat] = set(files)

    records = []
    skipped = []
    for (actor, setup, instance), repeats in sorted(grouped.items()):
        if set(repeats) != set(range(10)):
            skipped.append(f"{actor}/{setup}/{instance}: {len(repeats)} repeats")
            continue
        distances = [
            jaccard_distance(repeats[i], repeats[j])
            for i, j in combinations(range(10), 2)
        ]
        records.append({
            "actor": actor,
            "setup": setup,
            "instance": instance,
            "distance": sum(distances) / len(distances),
            "n_pairs": len(distances),
        })
    return records, skipped


def load_solution_diversity(codebleu: dict) -> tuple[list[dict], list[str]]:
    records = []
    skipped = []
    for key, value in sorted(codebleu.items()):
        actor, setup, instance = key.split("/", 2)
        if actor not in ACTORS or setup not in SETUPS:
            raise ValueError(f"Unexpected CodeBLEU key: {key}")
        distance = value["distance"]
        if distance is None:
            skipped.append(key)
            continue
        records.append({
            "actor": actor,
            "setup": setup,
            "instance": instance,
            "distance": distance,
            "n_pairs": value["n_pairs"],
        })
    return records, skipped


def aggregate(read_records: list[dict], solution_records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for metric, records in (("read_diversity", read_records), ("solution_diversity", solution_records)):
        for record in records:
            grouped[(record["actor"], record["setup"], metric)].append(record["distance"])
    result = {}
    for actor in ACTORS:
        result[actor] = {}
        for metric in METRICS:
            result[actor][metric] = {}
            for setup in SETUPS:
                values = grouped[(actor, setup, metric)]
                if not values:
                    raise ValueError(f"No data for {actor}/{setup}/{metric}")
                mean = sum(values) / len(values)
                result[actor][metric][setup] = {
                    "mean": mean,
                    "plot_mean": round(mean, 3),
                    "n_instances": len(values),
                }
            baseline = result[actor][metric]["bashonly"]["plot_mean"]
            for setup in SETUPS[1:]:
                item = result[actor][metric][setup]
                item["percent_from_bashonly"] = 100 * (item["plot_mean"] - baseline) / baseline
    return result


def plot_pdf(summary: dict, output: Path) -> None:
    """Draw a vector PDF with the original figure's two-row, three-column layout."""
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    width, height = 1152, 576
    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    positive = HexColor("#1A7C8E")
    negative = HexColor("#E88AA3")
    x0, col_width, gap = 116, 313, 40
    panel_height = 176
    row_bottoms = (311, 112)
    low, high = -20, 35

    pdf.setTitle("Figure 3: Exploration and final-solution diversity")
    pdf.setFont("Helvetica-Bold", 16)
    for col, label in enumerate(ACTOR_LABELS):
        pdf.drawCentredString(x0 + col * (col_width + gap) + col_width / 2, 520, label)

    for row, metric in enumerate(METRICS):
        bottom = row_bottoms[row]
        pdf.saveState()
        pdf.translate(45, bottom + panel_height / 2)
        pdf.rotate(90)
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawCentredString(0, 0, "Read Diversity" if row == 0 else "Solution Diversity")
        pdf.restoreState()

        for col, actor in enumerate(ACTORS):
            left = x0 + col * (col_width + gap)

            def y(value: float) -> float:
                return bottom + (value - low) * panel_height / (high - low)

            for tick in (-20, -10, 0, 10, 20, 30):
                pdf.setStrokeColor(HexColor("#dddddd") if tick else HexColor("#222222"))
                pdf.setLineWidth(0.5 if tick else 1.2)
                pdf.line(left, y(tick), left + col_width, y(tick))
                if col == 0:
                    pdf.setFillColor(HexColor("#333333"))
                    pdf.setFont("Helvetica", 9)
                    pdf.drawRightString(left - 7, y(tick) - 3, str(tick))

            for index, setup in enumerate(SETUPS[1:]):
                pct = summary[actor][metric][setup]["percent_from_bashonly"]
                center = left + (index + 0.5) * col_width / 5
                bar_width = 36
                zero = y(0)
                edge = y(pct)
                pdf.setFillColor(positive if pct >= 0 else negative)
                pdf.setStrokeColor(HexColor("#111111"))
                pdf.setLineWidth(0.6)
                pdf.rect(center - bar_width / 2, min(zero, edge), bar_width, abs(edge - zero), fill=1, stroke=1)
                pdf.setFillColor(HexColor("#111111"))
                pdf.setFont("Helvetica-Bold", 10)
                label = f"{pct:+.1f}%"
                if pct >= 0:
                    pdf.drawCentredString(center, edge + 7, label)
                else:
                    pdf.drawCentredString(center, edge - 15, label)
                if row == 1:
                    pdf.saveState()
                    pdf.translate(center + 8, bottom - 13)
                    pdf.rotate(30)
                    pdf.setFont("Helvetica", 11)
                    pdf.drawRightString(0, 0, SETUP_LABELS[index + 1])
                    pdf.restoreState()

    pdf.saveState()
    pdf.translate(19, height / 2)
    pdf.rotate(90)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawCentredString(0, 0, "% Difference from BashOnly")
    pdf.restoreState()
    pdf.setFillColor(positive)
    pdf.rect(347, 27, 14, 10, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(368, 27, "Higher than BashOnly")
    pdf.setFillColor(negative)
    pdf.rect(594, 27, 14, 10, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#111111"))
    pdf.drawString(615, 27, "Lower than BashOnly")
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/figure3")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    curated = data_root / "figure3/inputs.json"
    output = (args.output_dir or ROOT / "results/figure3").resolve()
    source = json.loads(curated.read_text())
    read_records, skipped_read = compute_read_diversity(source["read_sets"])
    solution_records, skipped_solution = load_solution_diversity(source["codebleu"])
    summary = aggregate(read_records, solution_records)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "per_instance.json", {
        "read_diversity": read_records,
        "solution_diversity": solution_records,
        "skipped_read": skipped_read,
        "skipped_solution": skipped_solution,
    })
    write_json(output / "plot_data.json", summary)
    plot_pdf(summary, output / "figure3.pdf")
    print(f"Read: {len(read_records)} tasks ({len(skipped_read)} skipped); "
          f"solution: {len(solution_records)} tasks ({len(skipped_solution)} skipped)")
    print(f"Wrote {output / 'figure3.pdf'}")


if __name__ == "__main__":
    main()
