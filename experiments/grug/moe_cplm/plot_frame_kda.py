# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot unsmoothed native metrics, preserving each configuration and seed."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def observations(path, metric, until):
    points = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        step, value = row.get("step"), row.get("metrics", {}).get(metric)
        if step is not None and value is not None and (until is None or step <= until):
            points[int(step)] = float(value)
    return sorted(points.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Configuration and seed label=metrics.jsonl")
    parser.add_argument("--metric", default="train/loss")
    parser.add_argument("--until", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.linewidth": 1.2,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.set_axisbelow(True)
    ax.grid(color="#B0B0B0", linestyle="--", linewidth=0.85, alpha=0.65)
    colors = ["#5F6368", "#1F77B4"]
    summary = {}
    table = []
    for index, item in enumerate(args.run):
        label, path = item.split("=", 1)
        points = observations(Path(path), args.metric, args.until)
        if not points:
            raise ValueError(f"No {args.metric} observations for {label}")
        x, y = zip(*points, strict=True)
        ax.plot(
            x,
            y,
            color=colors[index % len(colors)],
            linewidth=1.5,
            marker="o" if len(points) <= 30 else None,
            markersize=3.5,
            label=label,
        )
        summary[label] = {"metric": args.metric, "points": len(points), "first": points[0], "last": points[-1]}
        table.extend((label, step, value) for step, value in points)
    ax.set_xlabel("Optimizer step")
    labels = {
        "train/loss": "Training NLL (nats/token)",
        "eval/paloma/macro_loss": "Paloma macro NLL\n(nats/token)",
        "eval/uncheatable_eval/macro_loss": "Uncheatable macro NLL\n(nats/token)",
    }
    ax.set_ylabel(labels.get(args.metric, args.metric))
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("configuration_and_seed", "optimizer_step", args.metric))
        writer.writerows(table)
    plt.close(fig)


if __name__ == "__main__":
    main()
