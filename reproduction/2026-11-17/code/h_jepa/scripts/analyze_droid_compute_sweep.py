#!/usr/bin/env python3
"""Aggregate DROID planner-budget sweeps into the Figure 9(c) Pareto plot."""

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
    load_json,
    load_yaml,
    mean_se,
    pareto_indices,
    pyplot,
    run_metadata,
    save_figure,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="DROID evaluation roots (default: $HJEPA_HOME/ckpts/droid).")
    parser.add_argument(
        "--compute-axis",
        choices=("optimizer_work", "wall_clock", "tflops"),
        default="optimizer_work",
        help="TFLOPs is available only when planning records contain measured flops/tflops.",
    )
    parser.add_argument("--require-seeds", type=int, default=0)
    parser.add_argument("--protocol-version", type=int, default=2, help="Only include this training protocol version (default: 2).")
    add_common_output_args(parser, "droid_compute_sweep")
    return parser.parse_args()


def _depth(model: str) -> int:
    match = re.search(r"_l(\d+)$", model)
    return int(match.group(1)) if match else 1


def _samples(config: dict, model: str) -> dict[str, int]:
    solver = config.get("solver", {}) or {}
    solvers = solver.get("solvers")
    if not solvers:
        return {"level1": int(solver.get("num_samples", 0))}
    depth = _depth(model)
    return {
        key: int(value.get("num_samples", 0))
        for key, value in solvers.items()
        if int(key.removeprefix("level")) <= depth
    }


def _compute(paths: list[Path], episodes: int) -> dict[str, float]:
    work = wall_clock = flops = 0.0
    has_flops = False
    for path in paths:
        for record in load_json(path):
            iterations = record.get("n_iters", [])
            if not isinstance(iterations, list):
                iterations = [iterations]
            work += float(record.get("num_samples", 0)) * float(record.get("horizon", 0)) * sum(map(float, iterations))
            per_env = record.get("wall_clock_per_env", [])
            wall_clock += sum(map(float, per_env)) if per_env else float(record.get("wall_clock", 0))
            if "flops" in record or "tflops" in record:
                has_flops = True
                flops += float(record.get("flops", 0)) + 1e12 * float(record.get("tflops", 0))
    divisor = max(episodes, 1)
    return {
        "optimizer_work": work / divisor,
        "wall_clock": wall_clock / divisor,
        "tflops": flops / divisor / 1e12 if has_flops else math.nan,
    }


def collect(inputs: list[str], protocol_version: int = 2) -> list[dict]:
    rows = []
    for eval_path in find_files(inputs, ["eval.csv"]):
        with eval_path.open(newline="") as handle:
            result = next(csv.DictReader(handle), None)
        if result is None or "frechet/skill_mean" not in result:
            continue
        config_path = eval_path.parent / "plan_config.yaml"
        if not config_path.exists():
            continue
        config = load_yaml(config_path)
        metadata = run_metadata(eval_path)
        if not accepted_run(metadata, protocol_version):
            continue
        planner_match = re.search(r"plan_seed(-?\d+)", str(eval_path))
        planner_seed = int(planner_match.group(1)) if planner_match else int(config.get("seed", 0))
        episodes = int(float(result.get("n_episodes", 0)))
        compute_paths = sorted(eval_path.parent.glob("ep_*/planning_compute.json"))
        rows.append(
            {
                **metadata,
                "planner_seed": planner_seed,
                "samples": json.dumps(_samples(config, str(metadata["model"])), sort_keys=True),
                "fidelity": 100.0 * float(result["frechet/skill_mean"]),
                "episodes": episodes,
                **_compute(compute_paths, episodes),
                "result_path": str(eval_path),
            }
        )
    return rows


def reduce(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per_seed = []
    keys = ("model", "seed", "samples")
    for key, values in sorted(group_rows(rows, keys).items()):
        record = dict(zip(keys, key))
        record["fidelity"] = float(np.mean([row["fidelity"] for row in values]))
        record["n_planner_seeds"] = len(values)
        for metric in ("optimizer_work", "wall_clock", "tflops"):
            finite = [row[metric] for row in values if np.isfinite(row[metric])]
            record[metric] = float(np.mean(finite)) if finite else math.nan
        per_seed.append(record)
    summary = []
    for key, values in sorted(group_rows(per_seed, ("model", "samples")).items()):
        mean, se, count = mean_se(row["fidelity"] for row in values)
        record = {"model": key[0], "samples": key[1], "fidelity_mean": mean, "fidelity_se": se, "n_seeds": count}
        for metric in ("optimizer_work", "wall_clock", "tflops"):
            finite = [row[metric] for row in values if np.isfinite(row[metric])]
            record[metric] = float(np.mean(finite)) if finite else math.nan
        summary.append(record)
    return per_seed, summary


def plot(summary: list[dict], axis_name: str, output_dir: Path) -> None:
    rows = [row for row in summary if np.isfinite(row[axis_name])]
    if not rows:
        raise SystemExit(f"No finite {axis_name} values were found in the planning records.")
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for (model,), values in sorted(group_rows(rows, ("model",)).items()):
        values = sorted(values, key=lambda row: row[axis_name])
        x = np.asarray([row[axis_name] for row in values])
        y = np.asarray([row["fidelity_mean"] for row in values])
        axis.errorbar(x, y, yerr=[row["fidelity_se"] for row in values], marker="o", alpha=0.4, label=model)
        frontier = pareto_indices(x, y)
        axis.plot(x[frontier], y[frontier], linewidth=2.5)
    if axis_name in ("optimizer_work", "tflops"):
        axis.set_xscale("log")
    labels = {"optimizer_work": "Optimizer work proxy", "wall_clock": "Planner seconds / episode", "tflops": "Planner TFLOPs / episode"}
    axis.set(xlabel=labels[axis_name], ylabel="Frechet fidelity (%)", title="DROID planner compute sweep")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_9c")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = args.inputs or [str(default_data_root() / "ckpts" / "droid")]
    raw = collect(inputs, args.protocol_version)
    if not raw:
        raise SystemExit("No DROID eval.csv outputs were found.")
    per_seed, summary = reduce(raw)
    if args.require_seeds and any(row["n_seeds"] < args.require_seeds for row in summary):
        raise SystemExit(f"At least one sweep cell has fewer than {args.require_seeds} training seeds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "figure_9c_per_eval.csv", raw)
    write_csv(args.output_dir / "figure_9c_per_seed.csv", per_seed)
    write_csv(args.output_dir / "figure_9c.csv", summary)
    plot(summary, args.compute_axis, args.output_dir)
    print(f"Aggregated {len(raw)} DROID sweep evaluations into {args.output_dir}")


if __name__ == "__main__":
    main()
