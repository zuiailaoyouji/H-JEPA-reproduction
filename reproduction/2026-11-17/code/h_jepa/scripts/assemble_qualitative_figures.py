#!/usr/bin/env python3
"""Assemble execution and decoded-plan frame strips (paper Figures 11 and 16)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from analysis_utils import add_common_output_args, pyplot, read_csv, save_figure


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=int, choices=(11, 16))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frames", type=int, default=8, help="Uniformly sampled execution/plan frames.")
    add_common_output_args(parser, "qualitative")
    return parser.parse_args()


def _paths(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser()
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _sample(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        raise ValueError("A frame directory is empty.")
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).round().astype(int)
    result = [paths[index] for index in indices]
    while len(result) < count:
        result.append(result[-1])
    return result


def _draw_image(axis, plt, path: Path, border: str | None = None) -> None:
    axis.imshow(plt.imread(path))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(border is not None)
        if border:
            spine.set_color(border)
            spine.set_linewidth(2.0)


def _step_labels(row: dict, frames: list[Path]) -> list[str]:
    supplied = [value.strip() for value in row.get("environment_steps", "").split(";") if value.strip()]
    if supplied:
        if len(supplied) != len(frames):
            raise ValueError("environment_steps must contain one semicolon-separated value per displayed frame.")
        return supplied
    labels = []
    for path in frames:
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        labels.append(match.group(1) if match else path.stem)
    return labels


def figure11(rows: list[dict], count: int, output_dir: Path) -> None:
    required = {"environment", "frames_dir", "goal_image"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"Figure 11 manifest needs columns {sorted(required)}")
    plt = pyplot()
    fig, axes = plt.subplots(len(rows), count + 1, figsize=(1.45 * (count + 1), 1.45 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        frames = _sample(_paths(row["frames_dir"]), count)
        labels = _step_labels(row, frames)
        for column, path in enumerate(frames):
            _draw_image(axes[row_index, column], plt, path)
            axes[row_index, column].set_title(f"t={labels[column]}", fontsize=7)
        _draw_image(axes[row_index, -1], plt, Path(row["goal_image"]), border="#4d9221")
        axes[row_index, 0].set_ylabel(row["environment"], rotation=0, ha="right", va="center")
    axes[0, -1].set_title("goal")
    fig.tight_layout(pad=0.35)
    save_figure(fig, output_dir / "figure_11")
    plt.close(fig)


def figure16(rows: list[dict], count: int, output_dir: Path) -> None:
    required = {"clip", "expert_dir", "plan_dir", "fidelity"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"Figure 16 manifest needs columns {sorted(required)}")
    rows = sorted(rows, key=lambda row: float(row["fidelity"]), reverse=True)
    blocks = 2 if len(rows) > 8 else 1
    per_block = int(np.ceil(len(rows) / blocks))
    spacer = 1
    cols = blocks * count + (blocks - 1) * spacer
    plt = pyplot()
    fig, axes = plt.subplots(2 * per_block, cols, figsize=(1.05 * cols, 1.0 * 2 * per_block), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for index, row in enumerate(rows):
        block = index // per_block
        within = index % per_block
        offset = block * (count + spacer)
        expert = _sample(_paths(row["expert_dir"]), count)
        plan = _sample(_paths(row["plan_dir"]), count)
        level2 = {int(value) for value in row.get("level2_indices", "").split(";") if value.strip()}
        for column, path in enumerate(expert):
            _draw_image(axes[2 * within, offset + column], plt, path)
        for column, path in enumerate(plan):
            border = "#4d9221" if column == 0 else ("#2166ac" if column in level2 else "#b2182b")
            _draw_image(axes[2 * within + 1, offset + column], plt, path, border=border)
        axes[2 * within, offset].set_ylabel(f"clip {row['clip']}\nexpert", rotation=0, ha="right", va="center", fontsize=7)
        axes[2 * within + 1, offset].set_ylabel(f"{float(row['fidelity']):+.0f}%\nplan", rotation=0, ha="right", va="center", fontsize=7)
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / "figure_16")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.manifest)
    if not rows:
        raise SystemExit(f"Manifest is empty: {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.figure == 11:
        figure11(rows, args.frames, args.output_dir)
    else:
        figure16(rows, args.frames, args.output_dir)
    print(f"Wrote figure_{args.figure} to {args.output_dir}")


if __name__ == "__main__":
    main()
