#!/usr/bin/env python3
"""Compute goal/subgoal cost monotonicity on held-out trajectories (Figure 8/Table 10)."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from analysis_utils import (
    accepted_run,
    add_common_output_args,
    default_data_root,
    group_rows,
    mean_se,
    pyplot,
    run_metadata,
    save_figure,
    write_csv,
)
from analyze_cost_geometry import _normalizer, _pixel_tensor, _selected_column
from data import load_normalizer_artifact
from final_probing_decoding_eval import _dense_encode_no_stride
from utils import load_training_config_for_checkpoint, resolve_model_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="Two-level HWM/H-JEPA checkpoints or run directories.")
    parser.add_argument("--dataset", type=Path, help="Override the validation HDF5 path for every checkpoint.")
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Episode sampling seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_output_args(parser, "cost_monotonicity")
    return parser.parse_args()


def negative_spearman(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.allclose(values, values[0]):
        return math.nan
    correlation = spearmanr(values, np.arange(values.size)).statistic
    return -float(correlation)


def _normalize_actions(actions: np.ndarray, artifact: dict) -> np.ndarray:
    stats = artifact["stats"]["action"]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
    return (actions - mean) / np.maximum(std, 1e-12)


@torch.no_grad()
def _project_level2(level2, level1_sequence: torch.Tensor) -> torch.Tensor:
    result = level2.encode({"embed_1": level1_sequence}, key="embed_1")
    return result["embed_0"]


@torch.no_grad()
def analyze_episode(model, pixels, proprio, actions, device: torch.device) -> dict:
    batch = {"pixels_level1": _pixel_tensor(pixels, device)[None]}
    if proprio is not None:
        batch["proprio_level1"] = torch.as_tensor(proprio, dtype=torch.float32, device=device)[None]
    encoded = _dense_encode_no_stride(model, batch)
    level1 = encoded["embed_1"][0].float()
    level2_dense = encoded["embed_2"][0].float()
    stride = int(getattr(model.get_level(2), "temporal_stride", 1))
    sampled = torch.arange(0, len(level1), stride, device=device)
    if len(sampled) < 2:
        raise ValueError("Episode is too short to contain two level-2 states.")

    level1_goal_cost = torch.square(level1 - level1[-1]).sum(-1).cpu().numpy()
    level2_values = level2_dense.index_select(0, sampled)
    level2_goal_cost = torch.square(level2_values - level2_values[-1]).sum(-1).cpu().numpy()

    level1_model = model.get_level(1)
    level2_model = model.get_level(2)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    segment_scores = []
    segment_costs = []
    for segment_start in sampled[:-1].tolist():
        segment_end = min(segment_start + stride, len(level1) - 1)
        state = level1[segment_start : segment_start + 1]
        predictions = []
        for step in range(segment_start, segment_end):
            primitive = action_tensor[step : step + 1].unsqueeze(0)
            latent_action = level1_model.action_encoder(primitive)
            action_embed = level1_model.action_embed(latent_action)
            state = level1_model.predict(state.unsqueeze(0), action_embed)[:, -1]
            predictions.append(state[0])
        if not predictions:
            continue
        predicted_level1 = torch.stack(predictions)[None]
        projected = _project_level2(level2_model, predicted_level1)[0]
        target = level2_dense[segment_end]
        costs = torch.square(projected - target).sum(-1).cpu().numpy()
        score = negative_spearman(costs)
        if np.isfinite(score):
            segment_scores.append(score)
        segment_costs.append(costs)
    return {
        "goal_level1": negative_spearman(level1_goal_cost),
        "goal_level2": negative_spearman(level2_goal_cost),
        "subgoal": float(np.mean(segment_scores)) if segment_scores else math.nan,
        "level1_goal_cost": level1_goal_cost,
        "level2_goal_cost": level2_goal_cost,
        "segment_costs": segment_costs,
        "stride": stride,
    }


def analyze_checkpoint(checkpoint: Path, dataset_override: Path | None, args) -> tuple[list[dict], dict | None]:
    cfg = load_training_config_for_checkpoint(checkpoint)
    if int(cfg.num_levels) < 2:
        raise ValueError(f"Monotonicity needs at least two levels: {checkpoint}")
    artifact = load_normalizer_artifact(checkpoint.parent / "normalizer.pt")
    metadata = run_metadata(checkpoint)
    if not accepted_run(metadata, 2):
        raise ValueError(
            f"Refusing a checkpoint that is neither protocol-v2 nor in the verified release manifest: {checkpoint} "
            f"(training_protocol_version={metadata['protocol_version']!r})"
        )
    dataset = dataset_override or default_data_root() / f"{cfg.data.dataset.val_name}.h5"
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval().to(args.device)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    frameskip = int(dataset_cfg["level1"]["frameskip"])
    rng = np.random.default_rng(args.seed)
    rows = []
    example = None

    with h5py.File(dataset, "r") as handle:
        lengths = np.asarray(handle["ep_len"][:], dtype=int)
        offsets = np.asarray(handle["ep_offset"][:], dtype=int)
        valid = np.flatnonzero(lengths >= 3 * frameskip)
        chosen = rng.choice(valid, size=min(args.num_episodes, len(valid)), replace=False)
        for episode in chosen:
            lo = int(offsets[episode])
            length = int(lengths[episode])
            n_states = length // frameskip
            state_indices = lo + np.arange(n_states) * frameskip
            pixels = _selected_column(handle, "pixels", state_indices)
            proprio = None
            if bool(cfg.level1.wm.get("use_proprio", False)):
                source = dataset_cfg.get("keys_to_merge", {}).get("proprio", "proprio")
                proprio = _normalizer(_selected_column(handle, source, state_indices), artifact, "proprio")
            raw_actions = np.asarray(handle["action"][lo : lo + n_states * frameskip], dtype=np.float32)
            raw_actions = raw_actions.reshape(n_states, frameskip, -1)
            actions = _normalize_actions(raw_actions, artifact).reshape(n_states, -1)
            result = analyze_episode(model, pixels, proprio, actions, torch.device(args.device))
            rows.append(
                {
                    **metadata,
                    "episode": int(episode),
                    "dataset": str(dataset),
                    "goal_level1": result["goal_level1"],
                    "goal_level2": result["goal_level2"],
                    "subgoal": result["subgoal"],
                    "stride": result["stride"],
                }
            )
            if example is None:
                example = {**result, **metadata, "episode": int(episode)}
    return rows, example


def plot_example(example: dict, output_dir: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.8))
    axes[0].plot(example["level1_goal_cost"], color="#2166ac", label="level 1")
    x2 = np.arange(len(example["level2_goal_cost"])) * example["stride"]
    axes[0].plot(x2, example["level2_goal_cost"], color="#b2182b", marker="o", label="level 2")
    axes[0].set(ylabel="Squared goal cost", title=f"{example['env']} / {example['model']} / episode {example['episode']}")
    axes[0].legend(frameon=False)
    cursor = 0
    for costs in example["segment_costs"]:
        x = np.arange(len(costs)) + cursor
        axes[1].plot(x, costs, color="#4d9221", marker="o", linewidth=1)
        cursor += len(costs)
    axes[1].set(xlabel="Step within concatenated subgoal segments", ylabel="Squared subgoal cost")
    fig.tight_layout()
    save_figure(fig, output_dir / f"figure_8_{example['env']}_{example['model']}_seed{example['seed']}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoints = [resolve_model_checkpoint_path(value, os.environ.get("HJEPA_HOME")) for value in args.checkpoints]
    per_episode = []
    examples = []
    for checkpoint in checkpoints:
        rows, example = analyze_checkpoint(checkpoint, args.dataset, args)
        per_episode.extend(rows)
        if example:
            examples.append(example)
    per_seed = []
    keys = ("env", "model", "seed")
    for group_key, rows in sorted(group_rows(per_episode, keys).items()):
        record = dict(zip(keys, group_key))
        record["n_episodes"] = len(rows)
        for metric in ("goal_level1", "goal_level2", "subgoal"):
            finite = [row[metric] for row in rows if np.isfinite(row[metric])]
            record[metric] = float(np.mean(finite)) if finite else math.nan
        per_seed.append(record)
    summary = []
    for group_key, rows in sorted(group_rows(per_seed, ("env", "model")).items()):
        record = {"env": group_key[0], "model": group_key[1]}
        for metric in ("goal_level1", "goal_level2", "subgoal"):
            mean, se, count = mean_se(row[metric] for row in rows)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_se"] = se
            record["n_seeds"] = count
        summary.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "monotonicity_per_episode.csv", per_episode)
    write_csv(args.output_dir / "monotonicity_per_seed.csv", per_seed)
    write_csv(args.output_dir / "table_10.csv", summary)
    for example in examples:
        plot_example(example, args.output_dir)
    print(f"Analyzed {len(per_episode)} held-out episodes; outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
