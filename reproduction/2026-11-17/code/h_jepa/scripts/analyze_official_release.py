#!/usr/bin/env python3
"""Validate and aggregate every experiment supported by the released checkpoints.

The script intentionally enumerates the paper protocol instead of recursively accepting
arbitrary result directories. Missing or partial cells are reported, never treated as zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_utils import (
    accepted_run,
    default_data_root,
    load_json,
    load_yaml,
    mean_se,
    pyplot,
    run_metadata,
    save_figure,
    write_csv,
)


SIM_ENVS = ("ant", "fourroom", "cube", "pusht")
SIM_SEEDS = (42, 43, 44)
DROID_SEEDS = (1, 1000, 10000)
PLAN_SEEDS = (1, 2, 3)
DEPTH_MODELS = (
    ("lewm", "flat", 1),
    ("hjepa_l2", "l2", 2),
    ("hjepa_l3", "l3", 3),
    ("hjepa_l4", "l4", 4),
    ("hwm_l2", "l2", 2),
    ("hwm_l3", "l3", 3),
    ("hwm_l4", "l4", 4),
)
DROID_MODELS = (("lewm", "flat"), ("hwm_l2", "l2"), ("hjepa_l2", "l2"))


def _verified_release(path: Path) -> None:
    metadata = run_metadata(path)
    if not accepted_run(metadata) or metadata["source"] != "official_release":
        raise ValueError(f"result is not rooted in the verified release manifest: {path}")


def _simulation_result(path: Path) -> tuple[float, int]:
    _verified_release(path)
    metrics = load_yaml(path)
    successes = metrics.get("episode_successes")
    if not isinstance(successes, list) or len(successes) != 50:
        raise ValueError(f"expected exactly 50 episode outcomes in {path}")
    measured = 100.0 * float(np.mean(np.asarray(successes, dtype=float)))
    reported = float(metrics["success_rate"])
    if not np.isclose(measured, reported, atol=1e-8):
        raise ValueError(f"success_rate disagrees with episode outcomes in {path}")
    return reported, len(successes)


def collect_depth(ckpt_root: Path) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    for env in SIM_ENVS:
        for model, planner, depth in DEPTH_MODELS:
            for seed in SIM_SEEDS:
                run = ckpt_root / env / f"{env}_{model}" / f"seed{seed}"
                path = run / f"eval_{planner}" / "metrics.yaml"
                cell = {"experiment": "depth", "env": env, "model": model, "planner": planner, "seed": seed}
                if not path.exists():
                    missing.append({**cell, "path": str(path)})
                    continue
                success_rate, episodes = _simulation_result(path)
                rows.append({**cell, "depth": depth, "success_rate": success_rate, "episodes": episodes, "path": str(path)})
    return rows, missing


def collect_cost(ckpt_root: Path) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    for env in SIM_ENVS:
        for depth in (2, 3, 4):
            model = f"hjepa_l{depth}"
            for cost_level in range(1, depth + 1):
                planner = "flat" if cost_level == 1 else f"l{cost_level}_project"
                for seed in SIM_SEEDS:
                    run = ckpt_root / env / f"{env}_{model}" / f"seed{seed}"
                    path = run / f"eval_{planner}" / "metrics.yaml"
                    cell = {
                        "experiment": "cost_ladder",
                        "env": env,
                        "model": model,
                        "depth": depth,
                        "cost_level": cost_level,
                        "planner": planner,
                        "seed": seed,
                    }
                    if not path.exists():
                        missing.append({**cell, "path": str(path)})
                        continue
                    success_rate, episodes = _simulation_result(path)
                    rows.append({**cell, "success_rate": success_rate, "episodes": episodes, "path": str(path)})
    return rows, missing


def _read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one aggregate row in {path}")
    return rows[0]


def collect_droid(ckpt_root: Path) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    for model, planner in DROID_MODELS:
        for seed in DROID_SEEDS:
            for plan_seed in PLAN_SEEDS:
                run = ckpt_root / "droid" / f"droid_{model}" / f"seed{seed}"
                path = run / f"eval_{planner}" / f"plan_seed{plan_seed}" / "eval.csv"
                cell = {
                    "experiment": "droid",
                    "model": model,
                    "planner": planner,
                    "seed": seed,
                    "plan_seed": plan_seed,
                }
                if not path.exists():
                    missing.append({**cell, "path": str(path)})
                    continue
                _verified_release(path)
                values = _read_one_csv(path)
                episodes = int(values["n_episodes"])
                if episodes != 16:
                    raise ValueError(f"expected exactly 16 DROID episodes in {path}")
                rows.append(
                    {
                        **cell,
                        "frechet_fidelity": 100.0 * float(values["frechet/skill_mean"]),
                        "ate_xyz": float(values["ate/end_distance_xyz"]),
                        "episodes": episodes,
                        "path": str(path),
                    }
                )
    return rows, missing


def _aggregate(rows: list[dict], group_keys: tuple[str, ...], value_key: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        mean, se, count = mean_se(row[value_key] for row in group)
        result.append({**dict(zip(group_keys, key)), "mean": mean, "se": se, "n": count})
    return result


def aggregate_droid(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per_training_seed = []
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["seed"]), []).append(row)
    for (model, seed), group in sorted(grouped.items()):
        if len(group) != len(PLAN_SEEDS):
            continue
        per_training_seed.append(
            {
                "model": model,
                "seed": seed,
                "frechet_fidelity": float(np.mean([row["frechet_fidelity"] for row in group])),
                "ate_xyz": float(np.mean([row["ate_xyz"] for row in group])),
                "n_plan_seeds": len(group),
            }
        )
    return per_training_seed, _aggregate(per_training_seed, ("model",), "frechet_fidelity")


def build_cost_tables(cost_rows: list[dict], depth_rows: list[dict]) -> list[dict]:
    """Lay out the cost ablation and its hierarchical/LeWM references as in Tables 1/9."""
    cost_lookup = {
        (row["env"], int(row["depth"]), int(row["cost_level"])): row
        for row in cost_rows
    }
    depth_lookup = {(row["env"], row["model"]): row for row in depth_rows}
    rows = []
    for env in SIM_ENVS:
        lewm = depth_lookup.get((env, "lewm"))
        if lewm is not None:
            rows.append(
                {
                    "env": env,
                    "model": "lewm",
                    "depth": 1,
                    "native_l1_mean": lewm["mean"],
                    "native_l1_se": lewm["se"],
                    "l2_projection_mean": None,
                    "l2_projection_se": None,
                    "l3_projection_mean": None,
                    "l3_projection_se": None,
                    "l4_projection_mean": None,
                    "l4_projection_se": None,
                    "hierarchical_mean": None,
                    "hierarchical_se": None,
                    "n_seeds": lewm["n"],
                }
            )
        for depth in (2, 3, 4):
            row = {"env": env, "model": f"hjepa_l{depth}", "depth": depth}
            for level, prefix in (
                (1, "native_l1"),
                (2, "l2_projection"),
                (3, "l3_projection"),
                (4, "l4_projection"),
            ):
                cell = cost_lookup.get((env, depth, level))
                row[f"{prefix}_mean"] = None if cell is None else cell["mean"]
                row[f"{prefix}_se"] = None if cell is None else cell["se"]
            hierarchy = depth_lookup.get((env, f"hjepa_l{depth}"))
            row["hierarchical_mean"] = None if hierarchy is None else hierarchy["mean"]
            row["hierarchical_se"] = None if hierarchy is None else hierarchy["se"]
            counts = [
                int(cell["n"])
                for level in range(1, depth + 1)
                if (cell := cost_lookup.get((env, depth, level))) is not None
            ]
            if hierarchy is not None:
                counts.append(int(hierarchy["n"]))
            row["n_seeds"] = min(counts) if counts else 0
            rows.append(row)
    return rows


def plot_depth(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["env"], row["model"]): row for row in rows}
    plt = pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), sharey=True)
    for axis, env in zip(axes, SIM_ENVS):
        lewm = lookup[(env, "lewm")]
        axis.errorbar(
            [1],
            [lewm["mean"]],
            yerr=[lewm["se"]],
            marker="o",
            linestyle="none",
            capsize=2,
            color="#777777",
            label="LeWM (flat)",
        )
        for prefix, color, label in (
            ("hjepa_l", "#d62728", "H-JEPA"),
            ("hwm_l", "#2171b5", "HWM"),
        ):
            depths = (2, 3, 4)
            values = [lookup[(env, f"{prefix}{depth}")] for depth in depths]
            axis.plot(
                [1, 2],
                [lewm["mean"], values[0]["mean"]],
                color="#888888",
                linestyle=":",
                linewidth=1.0,
                zorder=1,
            )
            axis.errorbar(
                depths,
                [row["mean"] for row in values],
                yerr=[row["se"] for row in values],
                marker="o",
                capsize=2,
                color=color,
                label=label,
                zorder=2,
            )
        axis.set(title=env, xlabel="Number of levels", xticks=(1, 2, 3, 4), xlim=(0.8, 4.2), ylim=(0, 105))
    axes[0].set_ylabel("Planning success (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, output_dir / "figure_6_bottom")
    plt.close(fig)


def plot_cost(rows: list[dict], depth_rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["env"], int(row["depth"]), int(row["cost_level"])): row for row in rows}
    hierarchy_lookup = {
        (row["env"], int(row["depth"])): row
        for row in depth_rows
        if str(row["model"]).startswith("hjepa_l")
    }
    plt = pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), sharey=True)
    colors = {2: "#2166ac", 3: "#4d9221", 4: "#b2182b"}
    for axis, env in zip(axes, SIM_ENVS):
        for depth in (2, 3, 4):
            values = [lookup[(env, depth, level)] for level in range(1, depth + 1)]
            axis.errorbar(
                range(1, depth + 1),
                [row["mean"] for row in values],
                yerr=[row["se"] for row in values],
                marker="o",
                capsize=2,
                color=colors[depth],
                label=f"{depth} levels",
            )
            hierarchy = hierarchy_lookup[(env, depth)]
            axis.errorbar(
                [depth + 0.18],
                [hierarchy["mean"]],
                yerr=[hierarchy["se"]],
                marker="*",
                markersize=9,
                linestyle="none",
                capsize=2,
                color=colors[depth],
            )
        axis.set(title=env, xlabel="Cost-space level", xticks=(1, 2, 3, 4), xlim=(0.75, 4.45), ylim=(0, 105))
    axes[0].set_ylabel("Planning success (%)")
    axes[-1].plot([], [], marker="*", color="#333333", linestyle="none", label="Hierarchical")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "tables_1_9_cost_ladder")
    plt.close(fig)


def plot_droid(rows: list[dict], output_dir: Path) -> None:
    order = ("lewm", "hwm_l2", "hjepa_l2")
    labels = ("LeWM + IDM", "HWM", "H-JEPA")
    lookup = {row["model"]: row for row in rows}
    values = [lookup[model] for model in order]
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(4.6, 3.4))
    axis.bar(labels, [row["mean"] for row in values], yerr=[row["se"] for row in values], color=("#666666", "#2166ac", "#b2182b"), capsize=3)
    axis.set(ylabel="Frechet fidelity (%)", title="DROID open-loop planning (released three-model subset)")
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_9b_released_subset")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, default=default_data_root() / "ckpts")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("depth", "cost", "droid"),
        default=("depth", "cost", "droid"),
        help="Aggregate only the selected released-checkpoint experiment groups.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.ckpt_root / "OFFICIAL_WEIGHTS_MANIFEST.json"
    manifest = load_json(manifest_path)
    revision = str(manifest["revision"])
    output_dir = args.output_dir or default_data_root() / "analyse" / f"official_release_{revision[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.experiments)
    depth, missing_depth = collect_depth(args.ckpt_root) if "depth" in selected else ([], [])
    cost, missing_cost = collect_cost(args.ckpt_root) if "cost" in selected else ([], [])
    droid, missing_droid = collect_droid(args.ckpt_root) if "droid" in selected else ([], [])
    depth_summary = _aggregate(depth, ("env", "model", "planner", "depth"), "success_rate")
    cost_summary = _aggregate(cost, ("env", "model", "depth", "cost_level"), "success_rate")
    droid_per_seed, droid_summary = aggregate_droid(droid)
    cost_tables = build_cost_tables(cost_summary, depth_summary)

    if "depth" in selected:
        write_csv(output_dir / "depth_raw.csv", depth)
        write_csv(output_dir / "depth_summary.csv", depth_summary)
    if "cost" in selected:
        write_csv(output_dir / "cost_ladder_raw.csv", cost)
        write_csv(output_dir / "cost_ladder_summary.csv", cost_summary)
        write_csv(output_dir / "table_1_ant.csv", [row for row in cost_tables if row["env"] == "ant" and row["model"] != "lewm"])
        write_csv(output_dir / "table_9_other_envs.csv", [row for row in cost_tables if row["env"] != "ant"])
    if "droid" in selected:
        write_csv(output_dir / "droid_raw.csv", droid)
        write_csv(output_dir / "droid_per_training_seed.csv", droid_per_seed)
        write_csv(output_dir / "droid_summary.csv", droid_summary)
    missing = missing_depth + missing_cost + missing_droid
    write_csv(output_dir / "missing_results.csv", missing)
    coverage = {
        "release_revision": revision,
        "experiments": list(args.experiments),
        "complete": not missing,
    }
    if "depth" in selected:
        coverage["depth"] = {"found": len(depth), "expected": 84}
    if "cost" in selected:
        coverage["cost_ladder"] = {"found": len(cost), "expected": 108}
    if "droid" in selected:
        coverage["droid"] = {"found": len(droid), "expected": 27}
    (output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if "depth" in selected and len(depth) == 84:
        plot_depth(depth_summary, output_dir)
    if "cost" in selected and len(cost) == 108:
        plot_cost(cost_summary, depth_summary, output_dir)
    if "droid" in selected and len(droid) == 27:
        plot_droid(droid_summary, output_dir)
    print(json.dumps(coverage, indent=2))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"{len(missing)} expected results are missing; see {output_dir / 'missing_results.csv'}")


if __name__ == "__main__":
    main()
