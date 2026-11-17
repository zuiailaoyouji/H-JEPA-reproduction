#!/usr/bin/env python3
"""Aggregate frozen-probe NMSE across seeds (paper Figures 4, 5, and 12)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from analysis_utils import (
    accepted_run,
    add_common_output_args,
    default_data_root,
    find_files,
    group_rows,
    load_yaml,
    mean_se,
    pyplot,
    run_metadata,
    save_figure,
    write_csv,
)


PROBE_RE = re.compile(r"(?:^|/)eval/level(\d+)_probe_(.+)_nmse$")
DEFAULT_TARGETS = {
    "ant": {"xy": "agent position", "proprio": "body state"},
    "fourroom": {"xy": "agent position", "distractor0_xy": "distractor position"},
    "pusht": {
        "agent_xy": "agent position",
        "block_xy": "block position",
        "block_ori": "block orientation",
        "agent_vel": "agent velocity",
    },
    "cube": {
        "block_pos": "block position",
        "block_ori": "block orientation",
        "effector_pos": "end-effector position",
        "effector_yaw": "end-effector yaw",
        "gripper": "gripper",
        "arm_joint_pos": "arm state",
        "arm_joint_vel": "arm velocity",
    },
}
FIGURE_4_TARGETS = {
    "ant": {"xy", "proprio"},
    "fourroom": {"xy", "distractor0_xy"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Run directories, metric files, or globs (default: $HJEPA_HOME/ckpts).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Keep a short model name such as hjepa_l4; repeat as needed.",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Plot every probe target instead of the paper entity subset.",
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=("eval", "train"),
        help="Metric split to aggregate (paper figures use eval).",
    )
    parser.add_argument(
        "--require-seeds",
        type=int,
        default=0,
        help="Fail unless every plotted cell has at least this many seeds.",
    )
    parser.add_argument("--protocol-version", type=int, default=2, help="Only include this training protocol version (default: 2).")
    parser.add_argument("--ckpt-root", type=Path, default=default_data_root() / "ckpts")
    parser.add_argument("--decoder-seed", type=int, default=42)
    add_common_output_args(parser, "probes")
    return parser.parse_args()


def collect(inputs: list[str], split: str, protocol_version: int = 2) -> list[dict]:
    if not inputs:
        inputs = [str(default_data_root() / "ckpts")]
    rows = []
    for path in find_files(inputs, ["metrics.yaml"]):
        metrics = load_yaml(path)
        metadata = run_metadata(path)
        if not accepted_run(metadata, protocol_version):
            continue
        matched = False
        for key, value in metrics.items():
            match = re.search(
                rf"(?:^|/){re.escape(split)}/level(\d+)_probe_(.+)_nmse$",
                str(key),
            )
            if match is None:
                continue
            matched = True
            rows.append(
                {
                    **metadata,
                    "level": int(match.group(1)),
                    "entity": match.group(2),
                    "nmse": float(value),
                    "metrics_path": str(path),
                }
            )
        if matched and metadata["seed"] is None:
            raise ValueError(f"Could not infer a seed for {path}")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = ("env", "model", "entity", "level")
    summary = []
    for group_key, values in sorted(group_rows(rows, keys).items()):
        mean, se, count = mean_se(row["nmse"] for row in values)
        summary.append(
            {
                **dict(zip(keys, group_key)),
                "nmse_mean": mean,
                "nmse_se": se,
                "n_seeds": count,
            }
        )
    return summary


def plot(summary: list[dict], output_dir: Path, all_targets: bool) -> None:
    selected = []
    for row in summary:
        targets = DEFAULT_TARGETS.get(row["env"], {})
        paper_target = row["entity"] in FIGURE_4_TARGETS.get(row["env"], set())
        if all_targets or (
            not all_targets and row["model"] == "hjepa_l4" and paper_target
        ):
            selected.append(row)
    if not selected:
        return

    panels = sorted({(row["env"], row["model"]) for row in selected})
    plt = pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False)
    colors = ("#2166ac", "#b2182b", "#4d9221", "#7b3294", "#666666")
    for axis, (env, model) in zip(axes[0], panels):
        panel_rows = [row for row in selected if row["env"] == env and row["model"] == model]
        entities = sorted({row["entity"] for row in panel_rows})
        for color, entity in zip(colors, entities):
            values = sorted((row for row in panel_rows if row["entity"] == entity), key=lambda r: r["level"])
            label = DEFAULT_TARGETS.get(env, {}).get(entity, entity.replace("_", " "))
            axis.errorbar(
                [row["level"] for row in values],
                [row["nmse_mean"] for row in values],
                yerr=[row["nmse_se"] for row in values],
                marker="o",
                capsize=2,
                color=color,
                label=label,
            )
        axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set(title=f"{env} / {model}", xlabel="Hierarchy level", ylabel="Probe NMSE")
        axis.set_xticks(sorted({row["level"] for row in panel_rows}))
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_4_left")
    plt.close(fig)


def plot_decodings(ckpt_root: Path, output_dir: Path, seed: int) -> None:
    sources = []
    for env, label in (("ant", "Visual AntMaze"), ("fourroom", "FourRoom Distractors")):
        path = (
            ckpt_root
            / env
            / f"{env}_hjepa_l4"
            / f"seed{seed}"
            / "final_probing_decoding_eval_v2"
            / "decodings"
            / "eval_level_decodings.png"
        )
        if not path.exists():
            return
        sources.append((label, path))

    plt = pyplot()
    fig, axes = plt.subplots(len(sources), 1, figsize=(9.0, 4.8), squeeze=False)
    for axis, (label, path) in zip(axes[:, 0], sources):
        axis.imshow(plt.imread(path))
        axis.set_title(label, loc="left", fontsize=10)
        axis.axis("off")
    fig.suptitle(f"Post-hoc decodings from one fixed observation (training seed {seed})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "figure_4_right")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect(args.inputs, args.split, args.protocol_version)
    if args.model:
        rows = [row for row in rows if row["model"] in set(args.model)]
    if not rows:
        raise SystemExit("No probe metrics matched the requested inputs and filters.")
    all_summary = summarize(rows)
    summary = [row for row in all_summary if row["entity"] in DEFAULT_TARGETS.get(row["env"], {})]
    if not summary:
        raise SystemExit("No paper Table 6 probe targets were found.")
    if args.require_seeds:
        short = [row for row in summary if row["n_seeds"] < args.require_seeds]
        if short:
            cells = ", ".join(
                f"{row['env']}/{row['model']}/{row['entity']}/L{row['level']}={row['n_seeds']}"
                for row in short[:8]
            )
            raise SystemExit(f"Insufficient seed coverage: {cells}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "probe_per_seed.csv", rows)
    write_csv(args.output_dir / "probe_all_targets_summary.csv", all_summary)
    write_csv(args.output_dir / "probe_summary.csv", summary)
    plot(all_summary if args.all_targets else summary, args.output_dir, args.all_targets)
    plot_decodings(args.ckpt_root, args.output_dir, args.decoder_seed)
    print(f"Wrote {len(rows)} per-seed probe values to {args.output_dir}")


if __name__ == "__main__":
    main()
