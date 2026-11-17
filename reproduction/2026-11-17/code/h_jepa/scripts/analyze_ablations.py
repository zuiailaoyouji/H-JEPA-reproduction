#!/usr/bin/env python3
"""Render unpublished paper ablations from explicit, auditable CSV manifests."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis_utils import add_common_output_args, group_rows, mean_se, pyplot, read_csv, save_figure, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("stagewise", "calibration", "anticollapse", "coefficients", "collapse_table"))
    parser.add_argument("input", type=Path, help="CSV manifest; required columns are documented in scripts/ANALYSIS.md.")
    parser.add_argument("--figure", type=int, choices=(14, 15), help="Calibration figure number.")
    add_common_output_args(parser, "ablations")
    return parser.parse_args()


def _float_rows(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    converted = []
    for row in rows:
        value = dict(row)
        for field in fields:
            if field not in value:
                continue
            raw = str(value[field]).strip()
            value[field] = math.nan if raw.lower() in {"", "none", "nan", "na", "n/a", "-"} else float(raw)
        converted.append(value)
    return converted


def stagewise(rows: list[dict], output_dir: Path) -> None:
    required = {"procedure", "level1_data", "level2_data", "seed", "success_rate"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"stagewise input needs columns {sorted(required)}")
    rows = _float_rows(rows, ("success_rate",))
    order = ("explore", "stitch", "union")
    summary = []
    for key, values in sorted(group_rows(rows, ("procedure", "level1_data", "level2_data")).items()):
        mean, se, count = mean_se(row["success_rate"] for row in values)
        summary.append({"procedure": key[0], "level1_data": key[1], "level2_data": key[2], "mean": mean, "se": se, "n_seeds": count})
    plt = pyplot()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": (0.45, 1, 1)})
    panels = (("flat", (1, 3)), ("stagewise", (3, 3)), ("end_to_end", (3, 3)))
    image = None
    for axis, (procedure, shape) in zip(axes, panels):
        matrix = np.full(shape, np.nan)
        for row in summary:
            if row["procedure"] != procedure:
                continue
            if procedure == "flat" and row["level1_data"] in order:
                matrix[0, order.index(row["level1_data"])] = row["mean"]
            elif row["level1_data"] in order and row["level2_data"] in order:
                matrix[order.index(row["level1_data"]), order.index(row["level2_data"])] = row["mean"]
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap=cmap)
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white" if matrix[i, j] < 55 else "black")
        axis.set(title=procedure.replace("_", " "), xticks=range(3), xticklabels=order)
        if procedure == "flat":
            axis.set(yticks=[0], yticklabels=["level 1"], xlabel="Training data")
        else:
            axis.set(yticks=range(3), yticklabels=order, xlabel="Level-2 data", ylabel="Level-1 data")
    fig.colorbar(image, ax=axes, label="Success rate (%)", fraction=0.025)
    fig.subplots_adjust(wspace=0.35, right=0.9)
    save_figure(fig, output_dir / "figure_13")
    plt.close(fig)
    write_csv(output_dir / "figure_13.csv", summary)


def calibration(rows: list[dict], output_dir: Path, figure: int | None) -> None:
    required = {"configuration", "episode", "fidelity", "outcome"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"calibration input needs columns {sorted(required)}")
    if figure is None:
        raise ValueError("calibration requires --figure 14 or --figure 15")
    rows = _float_rows(rows, ("fidelity", "outcome"))
    configurations = []
    for key, values in sorted(group_rows(rows, ("configuration",)).items()):
        configurations.append(
            {
                "configuration": key[0],
                "fidelity_median": float(np.median([row["fidelity"] for row in values])),
                "outcome_mean": float(np.mean([row["outcome"] for row in values])),
                "n_episodes": len(values),
            }
        )
    within = []
    for key, values in group_rows(rows, ("configuration",)).items():
        if len(values) < 3:
            continue
        statistic = spearmanr([row["fidelity"] for row in values], [row["outcome"] for row in values]).statistic
        if np.isfinite(statistic):
            within.append(float(statistic))
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].scatter([row["fidelity_median"] for row in configurations], [row["outcome_mean"] for row in configurations], color="#2166ac", s=28)
    axes[0].set(xlabel="Median Frechet fidelity (%)", ylabel="Mean outcome", title="Planner configurations")
    axes[1].scatter([row["fidelity"] for row in rows], [row["outcome"] for row in rows], alpha=0.2, s=8, color="#b2182b")
    deciles = []
    if figure == 15:
        ordered = sorted(rows, key=lambda row: row["fidelity"])
        for index, values in enumerate(np.array_split(np.asarray(ordered, dtype=object), 10), start=1):
            if not len(values):
                continue
            deciles.append(
                {
                    "decile": index,
                    "fidelity_mean": float(np.mean([row["fidelity"] for row in values])),
                    "outcome_mean": float(np.mean([row["outcome"] for row in values])),
                    "n_episodes": len(values),
                }
            )
        axes[1].plot(
            [row["fidelity_mean"] for row in deciles],
            [row["outcome_mean"] for row in deciles],
            color="#333333",
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    elif any(row["outcome"] > 0.5 for row in rows):
        threshold = min(row["fidelity"] for row in rows if row["outcome"] > 0.5)
        axes[1].axvline(threshold, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Frechet fidelity (%)", ylabel="Outcome", title=f"Episodes; within-config rho={np.mean(within):.2f}" if within else "Episodes")
    fig.tight_layout()
    save_figure(fig, output_dir / f"figure_{figure}")
    plt.close(fig)
    write_csv(output_dir / f"figure_{figure}.csv", configurations)
    if deciles:
        write_csv(output_dir / "figure_15_deciles.csv", deciles)


def anticollapse(rows: list[dict], output_dir: Path) -> None:
    required = {"sigreg", "idm", "detach", "planner_seed", "effective_rank", "fidelity"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"anticollapse input needs columns {sorted(required)}")
    rows = _float_rows(rows, ("sigreg", "idm", "effective_rank", "fidelity"))
    summary = []
    for key, values in sorted(group_rows(rows, ("sigreg", "idm", "detach")).items()):
        fidelity, fidelity_se, count = mean_se(row["fidelity"] for row in values)
        summary.append({"sigreg": key[0], "idm": key[1], "detach": key[2], "effective_rank": float(np.mean([row["effective_rank"] for row in values])), "fidelity": fidelity, "fidelity_se": fidelity_se, "n_planner_seeds": count})
    labels = [f"S={row['sigreg']:g}\nI={row['idm']:g}\nD={row['detach']}" for row in summary]
    x = np.arange(len(summary))
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, len(summary) * 0.7), 3.5))
    axes[0].bar(x, [row["effective_rank"] for row in summary], color="#2166ac")
    axes[0].set(ylabel="Effective rank", title="Representation rank")
    axes[1].bar(x, [row["fidelity"] for row in summary], yerr=[row["fidelity_se"] for row in summary], color="#b2182b", capsize=2)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(ylabel="Frechet fidelity (%)", title="Planning fidelity")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_17")
    plt.close(fig)
    write_csv(output_dir / "figure_17.csv", summary)


def coefficients(rows: list[dict], output_dir: Path) -> None:
    required = {"level", "sigreg", "idm", "seed", "fidelity"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"coefficients input needs columns {sorted(required)}")
    rows = _float_rows(rows, ("level", "sigreg", "idm", "fidelity"))
    summary = []
    for key, values in sorted(group_rows(rows, ("level", "sigreg", "idm")).items()):
        mean, se, count = mean_se(row["fidelity"] for row in values)
        selected = any(str(row.get("selected", "")).strip().lower() in {"1", "true", "yes"} for row in values)
        summary.append({"level": int(key[0]), "sigreg": key[1], "idm": key[2], "mean": mean, "se": se, "n_seeds": count, "selected": selected})
    levels = sorted({row["level"] for row in summary})
    plt = pyplot()
    fig, axes = plt.subplots(1, len(levels), figsize=(4.0 * len(levels), 3.2), squeeze=False)
    finite = [row["mean"] for row in summary if np.isfinite(row["mean"])]
    vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
    image = None
    for axis, level in zip(axes[0], levels):
        values = [row for row in summary if row["level"] == level]
        sigregs = sorted({row["sigreg"] for row in values})
        idms = sorted({row["idm"] for row in values})
        matrix = np.full((len(sigregs), len(idms)), np.nan)
        for row in values:
            matrix[sigregs.index(row["sigreg"]), idms.index(row["idm"])] = row["mean"]
        image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
        from matplotlib.patches import Rectangle

        for row in values:
            if row["selected"]:
                i, j = sigregs.index(row["sigreg"]), idms.index(row["idm"])
                axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
        axis.set(title=f"Level {level}", xlabel="IDM weight", ylabel="SIGReg weight", xticks=range(len(idms)), xticklabels=[f"{v:g}" for v in idms], yticks=range(len(sigregs)), yticklabels=[f"{v:g}" for v in sigregs])
    fig.colorbar(image, ax=axes[0].tolist(), label="Frechet fidelity (%)", fraction=0.035)
    fig.subplots_adjust(wspace=0.35, right=0.88, bottom=0.17)
    save_figure(fig, output_dir / "figure_18")
    plt.close(fig)
    write_csv(output_dir / "figure_18.csv", summary)


def collapse_table(rows: list[dict], output_dir: Path) -> None:
    numeric = tuple(name for name in ("sigreg", "time_repulsion", "seed", "onset", "sensitivity_epoch", "best_fidelity", "final_fidelity", "final_sensitivity", "chi") if rows and name in rows[0])
    rows = _float_rows(rows, numeric)
    group_fields = [name for name in ("label", "sigreg", "time_repulsion") if rows and name in rows[0]]
    summary = []
    metrics = [name for name in numeric if name not in ("sigreg", "time_repulsion", "seed")]
    for key, values in sorted(group_rows(rows, group_fields).items()):
        record = dict(zip(group_fields, key))
        record["n_seeds"] = len(values)
        for metric in metrics:
            present = [row[metric] for row in values if metric in row and math.isfinite(row[metric])]
            record[f"{metric}_mean"] = float(np.mean(present)) if present else math.nan
            record[f"{metric}_se"] = mean_se(present)[1] if present else math.nan
        summary.append(record)
    write_csv(output_dir / "table_12.csv", summary)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    if not rows:
        raise SystemExit(f"Input CSV is empty: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handlers = {
        "stagewise": lambda: stagewise(rows, args.output_dir),
        "calibration": lambda: calibration(rows, args.output_dir, args.figure),
        "anticollapse": lambda: anticollapse(rows, args.output_dir),
        "coefficients": lambda: coefficients(rows, args.output_dir),
        "collapse_table": lambda: collapse_table(rows, args.output_dir),
    }
    handlers[args.kind]()
    print(f"Wrote {args.kind} analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
