#!/usr/bin/env python3
"""Reproduce Figure 4 efficiency scatter from final processed metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTORS = ("qwen3-coder-30b", "kimi-k2.5", "sonnet4.5")
ACTOR_LABELS = ("Q3C", "Kimi", "Sonnet")
SETUPS = ("bashonly", "atomic-common", "bash-search", "bash-hypo", "bash-scratchpad", "py-interface")
SETUP_LABELS = ("BashOnly", "Atomic", "NLSearch", "HypoTrack", "Scratchpad", "Python")
COLORS = ("#FF9800", "#2196F3", "#4CAF50")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def aggregate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for item in records:
        actor, setup = item["actor"], item["setup"]
        if actor not in ACTORS or setup not in SETUPS:
            raise ValueError(f"Unexpected actor/setup: {actor}/{setup}")
        if item["resolved"] not in (True, False, None):
            raise ValueError(f"Invalid resolved status: {item['resolved']}")
        groups[(actor, setup)].append(item)

    points = []
    for actor in ACTORS:
        for setup in SETUPS:
            group = groups[(actor, setup)]
            if len(group) != 650:
                raise ValueError(f"Expected 650 trajectories for {actor}/{setup}, got {len(group)}")
            ids = {(x["instance"], x["repeat"]) for x in group}
            by_instance = defaultdict(set)
            for item in group:
                by_instance[item["instance"]].add(item["repeat"])
            if (len(ids) != 650 or len(by_instance) != 65
                    or any(repeats != set(range(10)) for repeats in by_instance.values())):
                raise ValueError(f"Duplicate or missing task repeats in {actor}/{setup}")
            resolved = sum(x["resolved"] is True for x in group)
            unresolved = sum(x["resolved"] is False for x in group)
            unknown = len(group) - resolved - unresolved
            evaluated = resolved + unresolved
            if not evaluated:
                raise ValueError(f"No evaluated attempts in {actor}/{setup}")
            points.append({
                "actor": actor,
                "setup": setup,
                "n_trajectories": len(group),
                "n_resolved": resolved,
                "n_unresolved": unresolved,
                "n_unknown": unknown,
                "n_evaluated": evaluated,
                "input_tokens": sum(x["input_tokens"] for x in group) / len(group),
                "output_tokens": sum(x["output_tokens"] for x in group) / len(group),
                "steps": sum(x["steps"] for x in group) / len(group),
                "resolve_rate_percent": 100 * resolved / evaluated,
            })
    return points


def write_markdown(points: list[dict], output: Path) -> None:
    actor_names = dict(zip(ACTORS, ACTOR_LABELS))
    setup_names = dict(zip(SETUPS, SETUP_LABELS))
    lines = [
        "# Reproduced Figure 4 efficiency points", "",
        "Cost means use all 650 trajectories per actor and setup. Resolve rate excludes unknown outcomes.", "",
        "| Actor | Setup | Avg input tokens | Avg output tokens | Avg steps | Resolve rate | Evaluated / all |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in points:
        lines.append(
            f"| {actor_names[p['actor']]} | {setup_names[p['setup']]} | "
            f"{p['input_tokens']:,.0f} | {p['output_tokens']:,.0f} | "
            f"{p['steps']:.1f} | {p['resolve_rate_percent']:.1f}% | "
            f"{p['n_evaluated']} / {p['n_trajectories']} |"
        )
    output.write_text("\n".join(lines) + "\n")


def draw_marker(pdf, x: float, y: float, setup: str, color) -> None:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(color)
    pdf.setStrokeColor(HexColor("#222222"))
    pdf.setLineWidth(1.0)
    size = 6.5
    if setup == "bashonly":
        pdf.setLineWidth(4.0)
        pdf.setStrokeColor(HexColor("#111111"))
        for sign in (-1, 1):
            pdf.line(x - size, y + sign * size, x + size, y - sign * size)
        pdf.setLineWidth(2.3)
        pdf.setStrokeColor(color)
        for sign in (-1, 1):
            pdf.line(x - size, y + sign * size, x + size, y - sign * size)
    elif setup == "atomic-common":
        pdf.rect(x - size, y - size, 2 * size, 2 * size, fill=1, stroke=0)
    elif setup == "bash-search":
        path = pdf.beginPath()
        path.moveTo(x, y + size * 1.25)
        path.lineTo(x - size * 1.1, y - size)
        path.lineTo(x + size * 1.1, y - size)
        path.close()
        pdf.drawPath(path, fill=1, stroke=0)
    elif setup == "bash-hypo":
        pdf.circle(x, y, size, fill=1, stroke=0)
    elif setup == "bash-scratchpad":
        pdf.rect(x - 3, y - size, 6, 2 * size, fill=1, stroke=0)
        pdf.rect(x - size, y - 3, 2 * size, 6, fill=1, stroke=0)
    else:
        path = pdf.beginPath()
        path.moveTo(x, y + size * 1.3)
        path.lineTo(x + size * 1.3, y)
        path.lineTo(x, y - size * 1.3)
        path.lineTo(x - size * 1.3, y)
        path.close()
        pdf.drawPath(path, fill=1, stroke=0)


def plot_pdf(points: list[dict], output: Path) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    width, height = 1002, 367
    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    pdf.setTitle("Figure 4: Efficiency comparison across tool architectures")
    panels = (
        ("input_tokens", "Avg Input Tokens / Trajectory", 0.9e6, 3.25e6, [1e6, 1.5e6, 2e6, 2.5e6, 3e6]),
        ("output_tokens", "Avg Output Tokens / Trajectory", 10e3, 29e3, [10e3, 15e3, 20e3, 25e3]),
        ("steps", "Avg Steps / Trajectory", 43, 89, [50, 60, 70, 80]),
    )
    actor_color = dict(zip(ACTORS, (HexColor(c) for c in COLORS)))
    bottom, top = 64, 287
    lefts = (55, 382, 709)
    panel_width = 280
    for (metric, x_label, xmin, xmax, ticks), left in zip(panels, lefts):
        def px(value: float) -> float:
            return left + (value - xmin) * panel_width / (xmax - xmin)

        def py(value: float) -> float:
            return bottom + (value - 13) * (top - bottom) / 31

        pdf.setStrokeColor(HexColor("#333333"))
        pdf.setLineWidth(0.8)
        pdf.rect(left, bottom, panel_width, top - bottom, fill=0, stroke=1)
        for ytick in (15, 20, 25, 30, 35, 40):
            pdf.setStrokeColor(HexColor("#dddddd"))
            pdf.setDash(2, 2)
            pdf.line(left, py(ytick), left + panel_width, py(ytick))
            pdf.setDash()
            pdf.setFillColor(HexColor("#222222"))
            pdf.setFont("Helvetica", 10)
            pdf.drawRightString(left - 5, py(ytick) - 3, str(ytick))
        for tick in ticks:
            x = px(tick)
            pdf.setStrokeColor(HexColor("#e5e5e5"))
            pdf.setDash(2, 2)
            pdf.line(x, bottom, x, top)
            pdf.setDash()
            pdf.setFillColor(HexColor("#222222"))
            pdf.setFont("Helvetica", 10)
            if metric == "input_tokens":
                label = f"{tick / 1e6:.1f}M"
            elif metric == "output_tokens":
                label = f"{tick / 1e3:.0f}K"
            else:
                label = str(tick)
            pdf.drawCentredString(x, bottom - 16, label)

        for actor in ACTORS:
            for setup in SETUPS[1:] + ("bashonly",):
                point = next(p for p in points if p["actor"] == actor and p["setup"] == setup)
                draw_marker(pdf, px(point[metric]), py(point["resolve_rate_percent"]), setup, actor_color[actor])

        pdf.setFillColor(HexColor("#111111"))
        pdf.setFont("Helvetica", 13)
        pdf.drawCentredString(left + panel_width / 2, 26, x_label)

    pdf.saveState()
    pdf.translate(18, (bottom + top) / 2)
    pdf.rotate(90)
    pdf.setFont("Helvetica", 13)
    pdf.drawCentredString(0, 0, "Resolve Rate (%)")
    pdf.restoreState()

    pdf.setFont("Helvetica", 12)
    pdf.drawString(116, 335, "Actor")
    for i, actor in enumerate(ACTORS):
        x = 176 + i * 68
        draw_marker(pdf, x, 339, "bash-hypo", actor_color[actor])
        pdf.setFillColor(HexColor("#111111"))
        pdf.drawString(x + 13, 335, ACTOR_LABELS[i])
    pdf.drawString(447, 335, "Tool Architectures")
    x = 580
    for setup, label in zip(SETUPS, SETUP_LABELS):
        draw_marker(pdf, x, 339, setup, HexColor("#666666"))
        pdf.setFillColor(HexColor("#111111"))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(x + 11, 335, label)
        x += 11 + pdf.stringWidth(label, "Helvetica", 10) + 19

    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, help="Default: results/figure4")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    curated = data_root / "figure4/inputs.json"
    output = (args.output_dir or ROOT / "results/figure4").resolve()
    records = json.loads(curated.read_text())["trajectories"]
    points = aggregate(records)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plot_data.json", points)
    write_markdown(points, output / "figure4-numbers.md")
    plot_pdf(points, output / "figure4.pdf")
    print(f"Read {len(records)} trajectories; wrote {len(points)} points and {output / 'figure4.pdf'}")


if __name__ == "__main__":
    main()
