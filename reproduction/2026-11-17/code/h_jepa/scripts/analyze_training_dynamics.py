#!/usr/bin/env python3
"""Aggregate per-level training losses across seeds (paper Figure 10)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from analysis_utils import (
    accepted_run,
    add_common_output_args,
    default_data_root,
    find_files,
    group_rows,
    pyplot,
    run_metadata,
    save_figure,
    write_csv,
)


LEVEL_METRIC = re.compile(r"fit/(pred|sigreg|idm)_loss_level(\d+)$")
PAPER_SIMULATION_EPOCHS = {"fourroom": 7, "ant": 5, "pusht": 9, "cube": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Run directories or metrics.csv files.")
    parser.add_argument("--model", action="append", default=[], help="Short model-name filter.")
    parser.add_argument("--protocol-version", type=int, default=2, help="Only include this training protocol version (default: 2).")
    add_common_output_args(parser, "training_dynamics")
    return parser.parse_args()


def read_run(paths: list[Path], metadata: dict) -> list[dict]:
    # A resumed run can contain several CSV fragments. Later files overwrite the
    # same global step, preserving a single observation per optimizer step.
    step_rows: dict[int, tuple[int, dict[tuple[str, int], float]]] = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            metric_columns = []
            for name in reader.fieldnames or []:
                match = LEVEL_METRIC.fullmatch(name)
                if match:
                    metric_columns.append((name, match.group(1), int(match.group(2))))
                elif name == "fit/loss":
                    metric_columns.append((name, "total", 0))
            for source in reader:
                try:
                    step = int(float(source.get("step", "")))
                    epoch = int(float(source.get("epoch", "")))
                except (TypeError, ValueError):
                    continue
                values = {}
                for column, metric, level in metric_columns:
                    try:
                        value = float(source.get(column, ""))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and value >= 0:
                        values[(metric, level)] = value
                if values:
                    step_rows[step] = (epoch, values)

    totals: dict[tuple[int, str, int], list[float]] = {}
    for epoch, values in step_rows.values():
        for (metric, level), value in values.items():
            totals.setdefault((epoch, metric, level), []).append(value)
    rows = []
    for (epoch, metric, level), values in sorted(totals.items()):
        rows.append(
            {
                **metadata,
                "epoch": epoch + 1,
                "metric": metric,
                "level": level,
                "value": float(np.mean(values)),
                "n_steps": len(values),
            }
        )
    return rows


def collect(inputs: list[str], protocol_version: int = 2) -> list[dict]:
    if not inputs:
        inputs = [str(default_data_root() / "ckpts")]
    files = find_files(inputs, ["metrics.csv"])
    by_run: dict[str, list[Path]] = {}
    metadata_by_run = {}
    for path in files:
        metadata = run_metadata(path)
        if not accepted_run(metadata, protocol_version):
            continue
        run = str(metadata["run_dir"])
        by_run.setdefault(run, []).append(path)
        metadata_by_run[run] = metadata

    # Retries often coexist with an older directory carrying the same seed.
    # Figure 10 is a cross-seed summary, so retain only the most recently
    # written run for each logical (environment, model, seed) cell.
    latest: dict[tuple[object, object, object], tuple[float, str]] = {}
    for run, paths in by_run.items():
        metadata = metadata_by_run[run]
        key = (metadata["env"], metadata["model"], metadata["seed"])
        modified = max(path.stat().st_mtime for path in paths)
        if key not in latest or modified > latest[key][0]:
            latest[key] = (modified, run)

    rows = []
    for _, run in sorted(latest.values()):
        paths = by_run[run]
        rows.extend(read_run(paths, metadata_by_run[run]))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = ("env", "model", "epoch", "metric", "level")
    summary = []
    for key, values in sorted(group_rows(rows, keys).items()):
        array = np.asarray([row["value"] for row in values], dtype=float)
        summary.append(
            {
                **dict(zip(keys, key)),
                "mean": float(array.mean()),
                "min": float(array.min()),
                "max": float(array.max()),
                "n_seeds": len(array),
            }
        )
    return summary


def plot(summary: list[dict], output_dir: Path) -> None:
    plt = pyplot()
    panels = sorted({(row["env"], row["model"]) for row in summary})
    colors = ("#2166ac", "#b2182b", "#4d9221", "#7b3294")
    for env, model in panels:
        panel = [row for row in summary if row["env"] == env and row["model"] == model]
        levels_present = sorted({int(row["level"]) for row in panel if int(row["level"]) > 0})
        max_epoch = max(int(row["epoch"]) for row in panel)
        n_seeds = max(int(row["n_seeds"]) for row in panel)
        metrics = [metric for metric in ("pred", "sigreg", "idm", "total") if any(row["metric"] == metric for row in panel)]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.0), squeeze=False)
        for axis, metric in zip(axes[0], metrics):
            metric_rows = [row for row in panel if row["metric"] == metric]
            levels = sorted({row["level"] for row in metric_rows})
            for color, level in zip(colors, levels):
                if metric == "total":
                    color = "#333333"
                values = sorted((row for row in metric_rows if row["level"] == level), key=lambda row: row["epoch"])
                x = np.asarray([row["epoch"] for row in values])
                mean = np.asarray([row["mean"] for row in values])
                low = np.asarray([row["min"] for row in values])
                high = np.asarray([row["max"] for row in values])
                positive = np.concatenate((mean[mean > 0], low[low > 0], high[high > 0]))
                floor = float(positive.min() * 0.5) if len(positive) else np.finfo(float).tiny
                label = "total" if level == 0 else f"level {level}"
                axis.plot(x, np.maximum(mean, floor), color=color, label=label, marker="o" if len(x) == 1 else None)
                if any(int(row["n_seeds"]) > 1 for row in values):
                    axis.fill_between(x, np.maximum(low, floor), np.maximum(high, floor), color=color, alpha=0.17, linewidth=0)
            axis.set_yscale("log")
            axis.set(xlabel="Epoch", ylabel="Loss", title=metric.upper())
            axis.legend(frameon=False)
        seed_label = "seed" if n_seeds == 1 else "seeds"
        fig.suptitle(
            f"{env} / {model} (local partial: L{max(levels_present)}, "
            f"{n_seeds} {seed_label}, {max_epoch} epochs)"
        )
        fig.tight_layout()
        save_figure(fig, output_dir / f"training_dynamics_partial_{env}_{model}")
        plt.close(fig)


def plot_simulation_overview(summary: list[dict], output_dir: Path) -> None:
    env_order = ("fourroom", "ant", "pusht", "cube")
    environments = [env for env in env_order if any(row["env"] == env for row in summary)]
    metrics = [metric for metric in ("pred", "sigreg", "total") if any(row["metric"] == metric for row in summary)]
    if not environments or not metrics:
        return

    plt = pyplot()
    colors = ("#2166ac", "#b2182b", "#4d9221", "#7b3294")
    labels = {"fourroom": "FourRoom Distractors", "ant": "Visual AntMaze", "pusht": "Push-T", "cube": "OGBench Cube"}
    fig, axes = plt.subplots(
        len(metrics),
        len(environments),
        figsize=(3.25 * len(environments), 2.35 * len(metrics)),
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []
    for row_index, metric in enumerate(metrics):
        for column_index, env in enumerate(environments):
            axis = axes[row_index, column_index]
            panel = [row for row in summary if row["env"] == env and row["metric"] == metric]
            for color, level in zip(colors, sorted({int(row["level"]) for row in panel})):
                if metric == "total":
                    color = "#333333"
                values = sorted((row for row in panel if int(row["level"]) == level), key=lambda row: row["epoch"])
                x = np.asarray([row["epoch"] for row in values])
                y = np.asarray([row["mean"] for row in values])
                positive = y[y > 0]
                floor = float(positive.min() * 0.5) if len(positive) else np.finfo(float).tiny
                label = "total" if level == 0 else f"level {level}"
                (handle,) = axis.plot(
                    x,
                    np.maximum(y, floor),
                    color=color,
                    label=label,
                    marker="o" if len(x) == 1 else None,
                )
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
            axis.set_yscale("log")
            axis.set_xlabel("Epoch")
            paper_epochs = PAPER_SIMULATION_EPOCHS.get(env)
            if paper_epochs is not None:
                axis.set_xlim(0.8, paper_epochs + 0.2)
            if column_index == 0:
                axis.set_ylabel(f"{metric.upper()} loss")
            if row_index == 0:
                axis.set_title(labels.get(env, env))
    fig.suptitle("Local partial training trends: H-JEPA L2, seed 42 (not paper Figure 10)", y=0.995)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(legend_labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_dir / "training_dynamics_partial_simulation_overview")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect(args.inputs, args.protocol_version)
    if args.model:
        rows = [row for row in rows if row["model"] in set(args.model)]
    if not rows:
        raise SystemExit("No non-negative finite training-loss rows were found.")
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_dynamics_per_seed.csv", rows)
    write_csv(args.output_dir / "training_dynamics_summary.csv", summary)
    cells = []
    for env, model in sorted({(row["env"], row["model"]) for row in summary}):
        selected = [row for row in summary if row["env"] == env and row["model"] == model]
        observed_epochs = max(int(row["epoch"]) for row in selected)
        paper_epochs = PAPER_SIMULATION_EPOCHS.get(env)
        cells.append(
            {
                "env": env,
                "model": model,
                "levels": sorted({int(row["level"]) for row in selected if int(row["level"]) > 0}),
                "epochs": observed_epochs,
                "paper_simulation_epochs": paper_epochs,
                "schedule_complete": paper_epochs is not None and observed_epochs >= paper_epochs,
                "seeds": max(int(row["n_seeds"]) for row in selected),
            }
        )
    manifest = {
        "paper_reference": "Figure 10",
        "status": "local_partial_trend_only",
        "qualification": (
            "The simulation panels in the paper use four-level models for the Table 4 schedules "
            "(FourRoom 7, Ant 5, Push-T 9, Cube 8 epochs) and report the mean/min-max band over "
            "three seeds. DROID alone uses 100 epochs. These local runs are two-level seed-42 "
            "runs, several with incomplete schedules, so they show only analogous training trends."
        ),
        "cells": cells,
    }
    (args.output_dir / "training_dynamics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot(summary, args.output_dir)
    plot_simulation_overview(summary, args.output_dir)
    print(f"Aggregated {len(rows)} epoch-level loss values into {args.output_dir}")


if __name__ == "__main__":
    main()
