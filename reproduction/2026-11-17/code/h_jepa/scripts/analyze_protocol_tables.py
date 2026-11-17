#!/usr/bin/env python3
"""Recompute paper Tables 3 and 11 from datasets and the SIGReg formula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from analysis_utils import default_data_root, write_csv


DATASETS = {
    "fourroom": ("fourroom_tp35_d1.h5", 1_000_000, 0),
    "ant": ("visual_antmaze_medium_explore_stitch_train.h5", None, None),
    "pusht": ("pusht_expert_train.h5", None, None),
    "cube": ("cube_single_expert_train.h5", None, None),
}


def level1_frames(depth: int, upper_stride: int = 2, upper_window: int = 1) -> int:
    required = 4
    for _ in range(depth - 1):
        required = (required - 1) * upper_stride + upper_window
    return required


def select_transition_subset(
    lengths: np.ndarray, target: int, seed: int, minimum_span: int
) -> np.ndarray:
    valid = np.flatnonzero(lengths >= minimum_span)
    shuffled = np.random.default_rng(seed).permutation(valid)
    chosen: list[int] = []
    total = 0
    for episode in shuffled:
        previous = total
        chosen.append(int(episode))
        total += int(lengths[episode])
        if total >= target:
            if len(chosen) > 1 and abs(target - previous) < abs(target - total):
                chosen.pop()
            break
    return lengths[np.asarray(sorted(chosen), dtype=np.int64)]


def table3(data_root: Path) -> list[dict]:
    rows = []
    for env, (filename, transition_budget, subset_seed) in DATASETS.items():
        with h5py.File(data_root / filename, "r") as handle:
            lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        if transition_budget is not None:
            lengths = select_transition_subset(lengths, transition_budget, int(subset_seed), 20)
        baseline = None
        for depth in range(1, 5):
            frames = level1_frames(depth)
            # HDF5Dataset reserves frameskip * frames transitions. The paper reports
            # the endpoint-inclusive minimum, frameskip * (frames - 1) + 1.
            reserved_span = 5 * frames
            clips = int(np.maximum(lengths - reserved_span + 1, 0).sum())
            baseline = clips if baseline is None else baseline
            rows.append(
                {
                    "environment": env,
                    "depth": depth,
                    "model": "lewm" if depth == 1 else f"hjepa_l{depth}",
                    "level1_frames": frames,
                    "minimum_clip_length_env_steps": 5 * (frames - 1) + 1,
                    "loader_reserved_span": reserved_span,
                    "training_clips_per_epoch": clips,
                    "percent_of_lewm": 100.0 * clips / baseline,
                    "episodes": len(lengths),
                    "transitions": int(lengths.sum()),
                }
            )
    return rows


def _sigreg_setup(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn(256, 32, dtype=torch.float64, generator=generator, requires_grad=True)
    directions = torch.randn(32, 64, dtype=torch.float64, generator=generator)
    directions = directions / directions.norm(p=2, dim=0)
    knots = torch.linspace(0, 3, 17, dtype=torch.float64)
    step = 3 / 16
    weights = torch.full((17,), 2 * step, dtype=torch.float64)
    weights[[0, -1]] = step
    target = torch.exp(-knots.square() / 2)
    return samples, directions, knots, weights * target, target


def _ecf(
    samples: torch.Tensor, directions: torch.Tensor, knots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = (samples @ directions).unsqueeze(-1) * knots
    return projected.cos().mean(0), projected.sin().mean(0)


def _statistic(
    cosine: torch.Tensor,
    sine: torch.Tensor,
    sample_count: int,
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error = (cosine - target).square() + sine.square()
    return (error @ weights).mean() * sample_count


def table11(seed: int = 0) -> list[dict]:
    rows = []
    for shards in (1, 2, 4):
        samples, directions, knots, weights, target = _sigreg_setup(seed)
        pieces = samples.chunk(shards)
        components = [_ecf(piece, directions, knots) for piece in pieces]
        cosine = torch.stack([item[0] for item in components]).mean(0)
        sine = torch.stack([item[1] for item in components]).mean(0)
        global_value = _statistic(cosine, sine, len(samples), weights, target)
        global_gradient = torch.autograd.grad(global_value, samples, retain_graph=True)[0].norm()
        local_values = [
            _statistic(cos, sin, len(piece), weights, target)
            for piece, (cos, sin) in zip(pieces, components)
        ]
        reference = torch.stack(local_values).mean()
        rows.append(
            {
                "shards": shards,
                "samples_per_shard": len(samples) // shards,
                "gradient_norm_global": float(global_gradient),
                "tep_global": float(global_value),
                "tep_local_statistic_reference": float(reference),
            }
        )
    reference = rows[0]
    for row in rows:
        row["global_tep_delta_from_1_shard"] = row["tep_global"] - reference["tep_global"]
        row["global_gradient_delta_from_1_shard"] = (
            row["gradient_norm_global"] - reference["gradient_norm_global"]
        )
        row["local_reference_drift_percent"] = 100.0 * (
            row["tep_local_statistic_reference"] / reference["tep_local_statistic_reference"] - 1.0
        )
    if any(abs(row["global_tep_delta_from_1_shard"]) > 1e-12 for row in rows):
        raise AssertionError("global SIGReg value changed under equal sharding")
    if any(abs(row["global_gradient_delta_from_1_shard"]) > 1e-12 for row in rows):
        raise AssertionError("global SIGReg gradient changed under equal sharding")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sigreg-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_root / "analyse/protocol_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal = table3(args.data_root)
    sharding = table11(args.sigreg_seed)
    write_csv(output_dir / "table_3_temporal_layout.csv", temporal)
    write_csv(output_dir / "table_11_sigreg_sharding.csv", sharding)
    metadata = {
        "table_3": {
            "status": "exact_from_public_datasets_and_loader",
            "level1_frames": [level1_frames(depth) for depth in range(1, 5)],
        },
        "table_11": {
            "status": "exact_property_new_random_instance",
            "seed": args.sigreg_seed,
            "dtype": "float64",
            "samples": 256,
            "dimensions": 32,
            "directions": 64,
            "knots": 17,
            "qualification": (
                "The paper does not publish the seed for its fixed Gaussian samples and projection "
                "directions. Absolute values therefore differ, while global TEP and gradient invariance "
                "under 1/2/4 equal shards are reproduced exactly."
            ),
        },
    }
    (output_dir / "protocol_tables_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote Table 3 ({len(temporal)} rows) and Table 11 ({len(sharding)} rows) to {output_dir}")


if __name__ == "__main__":
    main()
