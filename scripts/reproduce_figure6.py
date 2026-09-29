#!/usr/bin/env python3
"""Recompute Figure 6 error means and render a vector PDF."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = {
    "qwen3-coder-30b": ("qwen", "Q3C"),
    "kimi-k2.5": ("kimi", "Kimi"),
    "sonnet4.5": ("sonnet", "Sonnet"),
}
SETUPS = ("bashonly", "atomic-common")
CODES = ("S3", "E2", "E6", "E7", "E3", "S4")
CATEGORIES = {
    "misaligned_param": ("S3", "E2", "E6"),
    "mis_edit": ("E7",),
    "wrong_syntax": ("E3", "S4"),
}
METRICS = ("total_errors", *CATEGORIES)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def compute(trajectories: list[dict]) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for row in trajectories:
        actor, setup = row["actor"], row["setup"]
        if actor not in ACTORS or setup not in SETUPS:
            raise ValueError(f"Unexpected actor/setup: {actor}/{setup}")
        if type(row["repeat"]) is not int or not 0 <= row["repeat"] < 10:
            raise ValueError(f"Invalid repeat: {row}")
        for code in CODES:
            if type(row[code]) is not int or row[code] < 0:
                raise ValueError(f"Invalid {code} count: {row}")
        groups[(actor, setup, row["instance"])].append(row)

    per_instance = []
    summary = {}
    for actor, (short, label) in ACTORS.items():
        instance_sets = [{key[2] for key in groups if key[:2] == (actor, setup)}
                         for setup in SETUPS]
        if instance_sets[0] != instance_sets[1] or len(instance_sets[0]) != 65:
            raise ValueError(f"Expected 65 matched tasks for {actor}")
        actor_rows = []
        for instance in sorted(instance_sets[0]):
            record = {"actor": short, "instance": instance}
            for setup in SETUPS:
                rows = groups[(actor, setup, instance)]
                if sorted(row["repeat"] for row in rows) != list(range(10)):
                    raise ValueError(f"Expected 10 distinct repeats for {actor}/{setup}/{instance}")
                means = {
                    category: sum(sum(row[code] for code in codes) for row in rows) / 10
                    for category, codes in CATEGORIES.items()
                }
                means["total_errors"] = sum(means.values())
                record["atomic" if setup == "atomic-common" else "bashonly"] = means
            record["diff"] = {
                metric: record["atomic"][metric] - record["bashonly"][metric]
                for metric in METRICS
            }
            actor_rows.append(record)
            per_instance.append(record)
        summary[short] = {
            "actor": actor,
            "label": label,
            "n_instances": len(actor_rows),
            "n_trajectories_per_setup": len(actor_rows) * 10,
            "bashonly": {
                metric: sum(row["bashonly"][metric] for row in actor_rows) / len(actor_rows)
                for metric in METRICS
            },
            "atomic": {
                metric: sum(row["atomic"][metric] for row in actor_rows) / len(actor_rows)
                for metric in METRICS
            },
        }
    return per_instance, summary


def numbers_markdown(summary: dict) -> str:
    lines = [
        "# Figure 6: Mean environment-interaction errors per trajectory",
        "",
        "Each value averages ten repeats within each task, then averages the 65 task means.",
        "Total is the sum of the three displayed categories.",
        "",
        "| Actor | Setup | Total | Misaligned param | Mis-edit | Wrong syntax |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for actor in ("qwen", "kimi", "sonnet"):
        for setup, label in (("bashonly", "BashOnly"), ("atomic", "Atomic")):
            values = summary[actor][setup]
            lines.append(
                f"| {summary[actor]['label']} | {label} | "
                + " | ".join(f"{values[metric]:.2f}" for metric in METRICS)
                + " |"
            )
    return "\n".join(lines) + "\n"


def render_pdf(summary: dict, output: Path) -> None:
    from reportlab.lib.colors import HexColor, Color
    from reportlab.pdfgen import canvas

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(1152, 360))
    pdf.setTitle("Figure 6: Environment-interaction errors")
    bash_color = HexColor("#1A7C8E")
    atomic_color = HexColor("#E88AA3")
    grid = Color(0.75, 0.75, 0.75)
    bottom, top = 86, 307
    actors = ("qwen", "kimi", "sonnet")
    setups = (("bashonly", bash_color), ("atomic", atomic_color))

    def axis(left: float, right: float, maximum: float, tick: float, title: str) -> None:
        pdf.setFont("Helvetica-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawCentredString((left + right) / 2, 332, title)
        pdf.setFont("Helvetica", 10)
        n_ticks = int(maximum / tick)
        for index in range(n_ticks + 1):
            value = index * tick
            y = bottom + (top - bottom) * value / maximum
            pdf.setStrokeColor(grid)
            pdf.setDash(2, 3)
            pdf.setLineWidth(0.35)
            pdf.line(left, y, right, y)
            pdf.setDash()
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawRightString(left - 8, y - 3, f"{value:g}")
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setLineWidth(0.8)
        pdf.rect(left, bottom, right - left, top - bottom, fill=0, stroke=1)
        pdf.saveState()
        pdf.translate(left - 46, (bottom + top) / 2)
        pdf.rotate(90)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(0, 0, "Avg Errors / Trajectory")
        pdf.restoreState()

    def bar(x: float, value: float, maximum: float, color: Color, width: float = 22) -> None:
        height = (top - bottom) * value / maximum
        pdf.setFillColor(color)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setLineWidth(0.45)
        pdf.rect(x, bottom, width, height, fill=1, stroke=1)
        if value > 0.01:
            label = f"{value:.2f}"
            if label.startswith("0."):
                label = label[1:]
            pdf.setFillColorRGB(0, 0, 0)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawCentredString(x + width / 2, min(bottom + height + 7, top + 3), label)

    axis(64, 333, 3.7, 0.5, "Total Errors")
    for index, actor in enumerate(actors):
        start = 91 + 81 * index
        for offset, (setup, color) in zip((0, 22), setups):
            bar(start + offset, summary[actor][setup]["total_errors"], 3.7, color)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawCentredString(start + 22, 65, summary[actor]["label"])
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(198, 29, "Total Errors")

    axis(443, 1109, 1.9, 0.25, "Error Categories")
    detail = (
        ("misaligned_param", "Misaligned Param"),
        ("mis_edit", "Mis-Edit"),
        ("wrong_syntax", "Wrong Syntax"),
    )
    for category_index, (category, label) in enumerate(detail):
        base = 459 + 218 * category_index
        for actor_index, actor in enumerate(actors):
            start = base + 64 * actor_index
            for offset, (setup, color) in zip((0, 22), setups):
                bar(start + offset, summary[actor][setup][category], 1.9, color)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawCentredString(start + 22, 65, summary[actor]["label"])
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(base + 86, 29, label)

    pdf.setFillColorRGB(1, 1, 1)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.rect(1002, 273, 98, 31, fill=1, stroke=1)
    for index, (setup, color) in enumerate(setups):
        y = 290 - 13 * index
        pdf.setFillColor(color)
        pdf.rect(1008, y, 15, 8, fill=1, stroke=1)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(1028, y, "BashOnly" if setup == "bashonly" else "Atomic")
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/figure6")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output_dir or ROOT / "results/figure6").resolve()
    labels = json.loads((data_root / "error_labels/per_trajectory_error_counts.json").read_text())
    trajectories = [{"actor": row["actor"], "setup": row["setup_type"],
                     "instance": row["instance"], "repeat": row["repeat"],
                     **{code: row[code] for code in CODES}}
                    for row in labels if row["actor"] in ACTORS and row["setup_type"] in SETUPS]
    paired, summary = compute(trajectories)
    write_json(output / "per_instance.json", paired)
    write_json(output / "summary.json", summary)
    (output / "figure6-numbers.md").write_text(numbers_markdown(summary))
    render_pdf(summary, output / "figure6.pdf")
    print(f"Wrote {len(paired)} paired tasks and {output / 'figure6.pdf'}")


if __name__ == "__main__":
    main()
