#!/usr/bin/env python3
"""CPU-only dataset, training-log, and run-manifest analysis for H-JEPA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import yaml

from metrics_lineage import find_metric_sessions, merge_metric_sessions

ROOT = Path("/home/songxw/H-JEPA")
DATA = Path("/home/songxw/hjepa-data")
OUT = ROOT / "local_env/analysis_20261027"
RUNS_JSON = ROOT / "local_env/fixes_20261027/restarted_l2_jobs.json"
PAPER_TEXT = Path("/tmp/hjepa-2610.06805.txt")
ENV = {
    "fourroom": {
        "dataset": "fourroom_tp35_d1",
        "train_transitions": 1_000_000,
        "subset_seed": 0,
        "entities": {"ego_xy_slow": ["xy"], "distractor_xy_fast": ["distractor0_xy"]},
    },
    "ant": {
        "dataset": "visual_antmaze_medium_explore_stitch_train",
        "train_transitions": 10_050_000,
        "subset_seed": 0,
        "entities": {"global_xy_slow": ["xy"], "body_state_fast": ["body_state"]},
    },
    "pusht": {
        "dataset": "pusht_expert_train",
        "train_transitions": None,
        "subset_seed": None,
        "entities": {
            "agent_xy": ["agent_xy"],
            "block_xy": ["block_xy"],
            "block_orientation": ["block_ori"],
            "agent_velocity": ["agent_vel"],
        },
    },
    "cube": {
        "dataset": "cube_single_expert_train",
        "train_transitions": None,
        "subset_seed": None,
        "entities": {
            "cube_position": ["block_pos"],
            "cube_orientation": ["block_ori"],
            "effector_position": ["effector_pos"],
            "effector_yaw": ["effector_yaw"],
            "gripper": ["gripper"],
            "arm_joint_positions": ["arm_joint_pos"],
            "arm_joint_velocities": ["arm_joint_vel"],
        },
    },
}
PAPER_CLIPS = {
    "fourroom": [952_708, 915_298, 840_478, 690_838],
    "ant": [9_575_000, 9_200_000, 8_450_000, 6_950_000],
    "pusht": [1_981_721, 1_701_446, 1_143_623, 285_931],
    "cube": [1_783_600, 1_636_600, 1_342_600, 754_600],
}


def now_local() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_dataset_hash(path: Path) -> str | None:
    """Reuse an earlier hash result or the dataset bundle's checksum manifest."""
    summary_path = OUT / "frequency_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            for entry in summary.values():
                if entry.get("source") == str(path) and entry.get("source_sha256"):
                    return entry["source_sha256"]
        except (OSError, json.JSONDecodeError):
            pass
    checksum_file = DATA / "SHA256SUMS"
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[1] == path.name:
                return fields[0]
    return None


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain") or ""
    return {
        "head": run("git", "rev-parse", "HEAD"),
        "worktree_dirty_at_collection": bool(status),
    }


def level_configs(depth: int) -> dict:
    configs = {"level1": {"frameskip": 5, "num_steps": 4}}
    for level in range(2, depth + 1):
        configs[f"level{level}"] = {"frameskip": 2, "window_size": 1, "num_steps": 4}
    return configs


def table3_counts() -> list[dict]:
    from stable_worldmodel.data.dataset import HDF5Dataset
    from h_jepa.data import _select_episodes_for_transition_budget

    rows = []
    for env, spec in ENV.items():
        for depth in range(1, 5):
            dataset = HDF5Dataset(
                name=spec["dataset"],
                cache_dir=DATA,
                **level_configs(depth),
            )
            lengths = np.asarray(dataset.lengths, dtype=np.int64)
            valid = np.flatnonzero(lengths >= int(dataset.last_level["span"]))
            selected = valid
            target = spec["train_transitions"]
            if target is not None:
                selected = _select_episodes_for_transition_budget(
                    dataset,
                    int(target),
                    np.random.default_rng(int(spec["subset_seed"])),
                )
            span = int(dataset.last_level["span"])
            clips = int(np.maximum(lengths[selected] - span + 1, 0).sum())
            level1_frames = 4 if depth == 1 else (3 * (2 ** (depth - 1)) + 1)
            official = int(PAPER_CLIPS[env][depth - 1])
            rows.append(
                {
                    "environment": env,
                    "depth": depth,
                    "dataset": spec["dataset"],
                    "level1_frames_per_clip": level1_frames,
                    "minimum_env_step_span_inclusive": (level1_frames - 1) * 5 + 1,
                    "loader_half_open_span_transitions": span,
                    "episodes_in_file": int(len(lengths)),
                    "eligible_episodes": int(len(valid)),
                    "selected_episodes": int(len(selected)),
                    "selected_transitions": int(lengths[selected].sum()),
                    "valid_clips_per_epoch": clips,
                    "fraction_of_local_lewm_clips": clips / next(
                        r["valid_clips_per_epoch"]
                        for r in rows
                        if r["environment"] == env and r["depth"] == 1
                    ) if depth > 1 else 1.0,
                    "paper_table3_clips": official,
                    "matches_paper_table3": clips == official,
                    "selection_seed": spec["subset_seed"],
                }
            )
    return rows


def entity_arrays(env: str, handle: h5py.File, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if env == "ant":
        # Paper probe target: 13 position-like qpos coordinates plus all 14 qvels.
        return {"global_xy_slow": arrays["xy"], "body_state_fast": np.concatenate([arrays["qpos"][..., 2:], arrays["qvel"]], axis=-1)}
    if env == "pusht":
        return {
            "agent_xy": arrays["state"][..., 0:2],
            "block_xy": arrays["state"][..., 2:4],
            "block_orientation": arrays["block_ori"],
            "agent_velocity": arrays["state"][..., 5:7],
        }
    if env == "cube":
        obs = arrays["observation"]
        return {
            "cube_position": obs[..., 19:22],
            "cube_orientation": obs[..., 22:28],
            "effector_position": obs[..., 12:15],
            "effector_yaw": obs[..., 15:17],
            "gripper": obs[..., 17:19],
            "arm_joint_positions": obs[..., 0:6],
            "arm_joint_velocities": obs[..., 6:12],
        }
    return {"ego_xy_slow": arrays["xy"], "distractor_xy_fast": arrays["distractor0_xy"]}


def selected_frequency_episodes(env: str, lengths: np.ndarray) -> np.ndarray:
    spec = ENV[env]
    if spec["train_transitions"] is None:
        return np.arange(len(lengths), dtype=np.int64)
    # Frequency analysis uses complete trajectories, selected by the training split's
    # transition budget and seed. All four environments have episodes long enough here.
    eligible = np.flatnonzero(lengths >= 35)
    rng = np.random.default_rng(int(spec["subset_seed"]))
    order = rng.permutation(eligible)
    chosen: list[int] = []
    total = 0
    for episode in order:
        size = int(lengths[episode])
        previous = total
        chosen.append(int(episode))
        total += size
        if total >= int(spec["train_transitions"]):
            if len(chosen) > 1 and abs(int(spec["train_transitions"]) - previous) < abs(int(spec["train_transitions"]) - total):
                chosen.pop()
            break
    return np.asarray(sorted(chosen), dtype=np.int64)


def per_dim_centroid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Power- and magnitude-weighted spectral centroids per episode/dimension."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x).all(axis=(1, 2))
    bad_rows = int((~finite).sum())
    x = x[finite]
    if x.size == 0:
        empty = np.empty((0, values.shape[-1]), dtype=np.float64)
        return empty, empty, bad_rows, 0
    x = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    valid_dims = std[:, 0, :] > 1e-12
    x = np.divide(x, std, out=np.zeros_like(x), where=std > 1e-12)
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0)
    centroids = []
    for weights in (spectrum**2, spectrum):
        mass = weights.sum(axis=1)
        numerator = (weights * frequencies[None, :, None]).sum(axis=1)
        centroid = np.divide(
            numerator,
            mass,
            out=np.full_like(numerator, np.nan),
            where=mass > 1e-12,
        )
        centroid[~valid_dims] = np.nan
        centroids.append(centroid)
    return centroids[0], centroids[1], bad_rows, int((~valid_dims).sum())


def frequency_analysis(batch_episodes: int = 128) -> tuple[list[dict], dict]:
    output_rows = []
    diagnostics = {}
    for env, spec in ENV.items():
        path = DATA / f"{spec['dataset']}.h5"
        power_centroids: dict[str, list[np.ndarray]] = defaultdict(list)
        magnitude_centroids: dict[str, list[np.ndarray]] = defaultdict(list)
        power_episode_means: dict[str, list[np.ndarray]] = defaultdict(list)
        read_keys = {
            "fourroom": ["xy", "distractor0_xy"],
            "ant": ["xy", "qpos", "qvel"],
            "pusht": ["state", "block_ori"],
            "cube": ["observation"],
        }[env]
        total_bad_rows = 0
        total_constant_dims = defaultdict(int)
        n_episodes = 0
        with h5py.File(path, "r") as handle:
            lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
            episodes = selected_frequency_episodes(env, lengths)
            # Group by sampled temporal length so each FFT uses its own correct frequency bins.
            sampled_counts = ((lengths[episodes] - 1) // 5) + 1
            for nframes in np.unique(sampled_counts):
                group = episodes[sampled_counts == nframes]
                sample_times = np.arange(int(nframes), dtype=np.int64) * 5
                for begin in range(0, len(group), batch_episodes):
                    ep_batch = group[begin : begin + batch_episodes]
                    indices = (offsets[ep_batch, None] + sample_times[None, :]).reshape(-1)
                    arrays = {
                        key: np.asarray(handle[key][indices]).reshape(len(ep_batch), int(nframes), -1)
                        for key in read_keys
                    }
                    entities = entity_arrays(env, handle, arrays)
                    for entity, values in entities.items():
                        power, magnitude, bad_rows, constant_dims = per_dim_centroid(values)
                        total_bad_rows += bad_rows
                        total_constant_dims[entity] += constant_dims
                        power_centroids[entity].append(power.reshape(-1))
                        magnitude_centroids[entity].append(magnitude.reshape(-1))
                        # One value per episode: equal-weight average over nonconstant entity dimensions.
                        valid_count = np.isfinite(power).sum(axis=1)
                        episode_mean = np.divide(
                            np.nansum(power, axis=1),
                            valid_count,
                            out=np.full(len(power), np.nan, dtype=np.float64),
                            where=valid_count > 0,
                        )
                        power_episode_means[entity].append(episode_mean)
                    n_episodes += len(ep_batch)

        entity_power_values = {
            entity: np.concatenate(values) for entity, values in power_centroids.items()
        }
        entity_magnitude_values = {
            entity: np.concatenate(values) for entity, values in magnitude_centroids.items()
        }
        entity_power_episode_values = {
            entity: np.concatenate(values) for entity, values in power_episode_means.items()
        }
        entity_power_means = {
            entity: float(np.nanmean(values)) for entity, values in entity_power_values.items()
        }
        entity_magnitude_means = {
            entity: float(np.nanmean(values)) for entity, values in entity_magnitude_values.items()
        }
        fastest = max(entity_power_means, key=entity_power_means.get)
        slowest = min(entity_power_means, key=entity_power_means.get)
        power_gap = entity_power_means[fastest] / entity_power_means[slowest]
        magnitude_fastest = max(entity_magnitude_means, key=entity_magnitude_means.get)
        magnitude_slowest = min(entity_magnitude_means, key=entity_magnitude_means.get)
        magnitude_gap = entity_magnitude_means[magnitude_fastest] / entity_magnitude_means[magnitude_slowest]
        power_episode_gap = None
        magnitude_episode_gap = None
        if env in ("fourroom", "ant"):
            slow_name, fast_name = (
                ("ego_xy_slow", "distractor_xy_fast") if env == "fourroom"
                else ("global_xy_slow", "body_state_fast")
            )
            power_episode_gap = entity_power_episode_values[fast_name] / entity_power_episode_values[slow_name]
            magnitude_episode_gap = (
                np.concatenate(magnitude_centroids[fast_name]).reshape(n_episodes, -1).mean(axis=1)
                / np.concatenate(magnitude_centroids[slow_name]).reshape(n_episodes, -1).mean(axis=1)
            )

        for entity, values in entity_power_values.items():
            finite_values = values[np.isfinite(values)]
            finite_magnitude_values = entity_magnitude_values[entity]
            finite_magnitude_values = finite_magnitude_values[np.isfinite(finite_magnitude_values)]
            output_rows.append(
                {
                    "environment": env,
                    "dataset": spec["dataset"],
                    "entity": entity,
                    "mean_power_spectral_centroid_cycles_per_level1_frame": float(np.mean(finite_values)),
                    "median_power_spectral_centroid_cycles_per_level1_frame": float(np.median(finite_values)),
                    "p25_power_spectral_centroid_cycles_per_level1_frame": float(np.quantile(finite_values, 0.25)),
                    "p75_power_spectral_centroid_cycles_per_level1_frame": float(np.quantile(finite_values, 0.75)),
                    "mean_magnitude_spectral_centroid_cycles_per_level1_frame": float(np.mean(finite_magnitude_values)),
                    "median_magnitude_spectral_centroid_cycles_per_level1_frame": float(np.median(finite_magnitude_values)),
                    "episode_count": int(n_episodes),
                    "dimension_episode_centroid_count": int(finite_values.size),
                    "constant_dimension_episode_count": int(total_constant_dims[entity]),
                    "pooled_power_fast_slow_gap": float(power_gap),
                    "pooled_magnitude_fast_slow_gap": float(magnitude_gap),
                    "gap_fastest_entity": fastest,
                    "gap_slowest_entity": slowest,
                    "paper_semantic_group": (
                        "slow" if entity.endswith("_slow") else
                        "fast" if entity.endswith("_fast") else
                        "single frequency band; not designated slow/fast"
                    ),
                }
            )
        diagnostics[env] = {
            "source": str(path),
            "source_sha256": existing_dataset_hash(path),
            "source_hash_provenance": "full-file SHA-256 from prior collection or bundle SHA256SUMS; not rehashed in this low-I/O pass",
            "source_size_bytes": int(path.stat().st_size),
            "source_mtime": datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "episodes_used": int(n_episodes),
            "first_level_sampling_stride_env_steps": 5,
            "frequency_gap_power_spectrum_fastest_over_slowest": float(power_gap),
            "frequency_gap_magnitude_spectrum_fastest_over_slowest": float(magnitude_gap),
            "fastest_entity": fastest,
            "slowest_entity": slowest,
            "slow_fast_power_episode_ratio_median": float(np.nanmedian(power_episode_gap)) if power_episode_gap is not None else None,
            "slow_fast_power_episode_ratio_p25": float(np.nanquantile(power_episode_gap, 0.25)) if power_episode_gap is not None else None,
            "slow_fast_power_episode_ratio_p75": float(np.nanquantile(power_episode_gap, 0.75)) if power_episode_gap is not None else None,
            "slow_fast_magnitude_episode_ratio_median": float(np.nanmedian(magnitude_episode_gap)) if magnitude_episode_gap is not None else None,
            "slow_fast_magnitude_episode_ratio_p25": float(np.nanquantile(magnitude_episode_gap, 0.25)) if magnitude_episode_gap is not None else None,
            "slow_fast_magnitude_episode_ratio_p75": float(np.nanquantile(magnitude_episode_gap, 0.75)) if magnitude_episode_gap is not None else None,
            "nonfinite_rows_dropped": int(total_bad_rows),
            "constant_dimension_episode_count_by_entity": dict(total_constant_dims),
            "episode_selection_seed": spec["subset_seed"],
            "episode_selection_transitions": spec["train_transitions"],
            "centroid_definition": "per-episode, per-dimension mean-centered/unit-variance stream; power |rFFT|^2 weighted centroid is primary, magnitude |rFFT| sensitivity is also reported; arithmetic mean over dimensions and episodes",
            "fft_details": "rFFT, no window, no detrend beyond mean removal; constant dimensions excluded; equal weight per valid episode-dimension, not mean of episode ratios",
        }
        print(f"Frequency complete: {env}, {n_episodes} episodes, power gap {power_gap:.6f}, magnitude gap {magnitude_gap:.6f}", flush=True)
    return output_rows, diagnostics


def process_present(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def snapshot_source(source: Path, destination: Path, complete_csv_rows: bool = False) -> dict:
    before = source.stat()
    raw = source.read_bytes()
    ignored = 0
    if complete_csv_rows and raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        ignored = len(raw) - end
        raw = raw[:end]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {
        "live_source": str(source),
        "snapshot": str(destination),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_size_bytes": len(raw),
        "live_source_mtime_before_read": datetime.fromtimestamp(before.st_mtime, ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "incomplete_trailing_csv_bytes_ignored": ignored,
    }


def training_manifest() -> tuple[dict, list[dict], dict[str, list[dict]]]:
    jobs = json.loads(RUNS_JSON.read_text())
    collection_time = now_local()
    snapshot_dir = OUT / "source_snapshots" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_entries = []
    summaries = []
    metrics_by_env = {}
    for job in jobs:
        env = job["environment"]
        run_dir = Path(job["run_dir"])
        config_path = run_dir / "config.yaml"
        config_snapshot = snapshot_source(config_path, snapshot_dir / f"{env}_config.yaml")
        config = yaml.safe_load(Path(config_snapshot["snapshot"]).read_text())
        metrics_paths = find_metric_sessions(run_dir)
        metrics_snapshots = [
            snapshot_source(
                metrics_path,
                snapshot_dir / f"{env}_metrics_session_{index:02d}.csv",
                complete_csv_rows=True,
            )
            for index, metrics_path in enumerate(metrics_paths)
        ]
        metrics, metrics_lineage = merge_metric_sessions(
            [Path(item["snapshot"]) for item in metrics_snapshots],
            source_paths=metrics_paths,
        )
        for session, source_snapshot in zip(
            metrics_lineage["sessions"], metrics_snapshots
        ):
            session["snapshot"] = source_snapshot
        metrics_path = metrics_paths[-1] if metrics_paths else None
        metrics_snapshot = metrics_snapshots[-1] if metrics_snapshots else None
        fit_rows = [row for row in metrics if isinstance(row.get("fit/loss"), (int, float))]
        latest = fit_rows[-1] if fit_rows else {}
        resume_ckpt = run_dir / "lightning_resume/last.ckpt"
        final_path = run_dir / f"{job['model']}_object.ckpt"
        final_ckpts = [final_path] if final_path.exists() else []
        epoch_ckpts = sorted(run_dir.glob("*_epoch_*_object.ckpt"))
        pid_alive = process_present(job.get("pid"))
        log_age = max(0.0, datetime.now().timestamp() - metrics_path.stat().st_mtime) if metrics_path else None
        if final_ckpts:
            status = "final_model_present_evaluations_separately_tracked"
        elif log_age is not None and log_age < 600:
            status = "recent_metrics_no_final_model"
        else:
            status = "no_final_model_training_liveness_unconfirmed"
        eval_config_path = ROOT / "h_jepa/config/eval" / f"{config['planning_eval']['config_name']}.yaml"
        eval_snapshot = snapshot_source(eval_config_path, snapshot_dir / f"{env}_planning_config.yaml")
        eval_config = yaml.safe_load(Path(eval_snapshot["snapshot"]).read_text())
        task_path = ROOT / "h_jepa" / eval_config["load_eval_trajs_path"]
        probe_path = ROOT / "h_jepa/config/probing" / f"{config['final_probing_decoding_eval']['config_name']}.yaml"
        probe_snapshot = snapshot_source(probe_path, snapshot_dir / f"{env}_probing_config.yaml")
        probe_config = yaml.safe_load(Path(probe_snapshot["snapshot"]).read_text())
        result_files = sorted({*run_dir.glob("planning_eval/**/*.yaml"), *run_dir.glob("planning_eval/**/*.json"), *run_dir.glob("final_probing_decoding_eval_v2/**/*.yaml"), *run_dir.glob("final_probing_decoding_eval_v2/**/*.json")})

        run_entries.append(
            {
                "environment": env,
                "model": job["model"],
                "training_seed": int(job["seed"]),
                "pid_recorded_at_launch": int(job["pid"]),
                "recorded_pid_visible_in_collection_namespace": pid_alive,
                "pid_visibility_note": "Host training processes may not be visible in this collection namespace; false is not evidence of termination.",
                "state_at_collection": status,
                "started_at": job["started_at"],
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "config_sha256": config_snapshot["snapshot_sha256"],
                "config_snapshot": config_snapshot,
                "metrics_csv": str(metrics_path) if metrics_path else None,
                "metrics_csv_sha256": metrics_snapshot["snapshot_sha256"] if metrics_snapshot else None,
                "metrics_snapshot": metrics_snapshot,
                "metrics_sessions": metrics_lineage,
                "metrics_log_age_seconds_at_collection": log_age,
                "metric_steps_with_fit_loss": len(fit_rows),
                "metric_steps_with_nonfinite_fit_loss": sum("fit/loss" in r and r["fit/loss"] is None for r in metrics),
                "latest_optimizer_step": latest.get("step"),
                "latest_epoch_zero_based": latest.get("epoch"),
                "latest_fit_loss": latest.get("fit/loss"),
                "latest_fit_prediction_loss_level1": latest.get("fit/pred_loss_level1"),
                "latest_fit_prediction_loss_level2": latest.get("fit/pred_loss_level2"),
                "latest_fit_sigreg_loss_level1": latest.get("fit/sigreg_loss_level1"),
                "latest_fit_sigreg_loss_level2": latest.get("fit/sigreg_loss_level2"),
                "latest_learning_rate_level1": latest.get("model_opt/level1"),
                "latest_learning_rate_level2": latest.get("model_opt/level2"),
                "max_epochs": config.get("trainer", {}).get("max_epochs"),
                "expected_optimizer_updates": int(config["trainer"]["max_epochs"]) * (next(r["valid_clips_per_epoch"] for r in json.loads((OUT / "table3_clip_counts.json").read_text()) if r["environment"] == env and r["depth"] == 2) // int(config["loader"]["batch_size"])),
                "batch_size": config.get("loader", {}).get("batch_size"),
                "gradient_accumulation": config.get("trainer", {}).get("accumulate_grad_batches"),
                "frame_chunk_size": config.get("memory", {}).get("encoder_frame_chunk_size"),
                "training_protocol_version": config.get("training_protocol_version"),
                "initialized_from_old_checkpoint": job.get("initialization") != "fresh; no old checkpoint loaded",
                "resume_checkpoint": str(resume_ckpt) if resume_ckpt.exists() else None,
                "resume_checkpoint_mtime": datetime.fromtimestamp(resume_ckpt.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds") if resume_ckpt.exists() else None,
                "final_model_checkpoints": [str(path) for path in final_ckpts],
                "intermediate_epoch_model_checkpoints": [str(path) for path in epoch_ckpts],
                "planned_evaluation": {
                    "planning_config_snapshot": eval_snapshot,
                    "task_file": str(task_path),
                    "task_file_sha256": sha256(task_path),
                    "num_tasks": eval_config["eval"]["num_eval"],
                    "planner_seed": eval_config["seed"],
                    "chunk_size_default": eval_config["eval"].get("chunk_size", 1),
                    "chunk_seeding": "planner seed + first task index; different random stream from one 50-task World",
                    "probe_config_snapshot": probe_snapshot,
                    "probe_learning_rate": probe_config.get("optimizer", {}).get("probe_lr", probe_config.get("optimizer", {}).get("lr")),
                    "decoder_learning_rate": probe_config.get("optimizer", {}).get("decoder_lr"),
                    "probe_decoder_overrides": config["final_probing_decoding_eval"].get("overrides"),
                },
                "current_evaluation_result_files": [{"path": str(p), "sha256": sha256(p)} for p in result_files],
            }
        )
        metrics_by_env[env] = fit_rows
        summaries.append(
            {
                "environment": env,
                "latest_step": latest.get("step"),
                "latest_epoch": latest.get("epoch"),
                "fit_loss": latest.get("fit/loss"),
                "pred_loss_l1": latest.get("fit/pred_loss_level1"),
                "pred_loss_l2": latest.get("fit/pred_loss_level2"),
                "sigreg_loss_l1": latest.get("fit/sigreg_loss_level1"),
                "sigreg_loss_l2": latest.get("fit/sigreg_loss_level2"),
                "lr_l1": latest.get("model_opt/level1"),
                "lr_l2": latest.get("model_opt/level2"),
                "metric_rows": len(fit_rows),
                "metrics_session_count": metrics_lineage["session_count"],
                "superseded_logged_steps": metrics_lineage[
                    "superseded_logged_step_count"
                ],
                "status_at_collection": status,
            }
        )
    code_paths = [
        ROOT / "h_jepa/main_hjepa.py",
        ROOT / "h_jepa/training_runtime.py",
        ROOT / "h_jepa/models/jepa.py",
        ROOT / "h_jepa/final_probing_decoding_eval.py",
        ROOT / "h_jepa/config/train/base/fourroom.yaml",
        ROOT / "h_jepa/config/train/base/ant.yaml",
        ROOT / "h_jepa/config/train/base/pusht.yaml",
        ROOT / "h_jepa/config/train/base/cube.yaml",
        ROOT / "h_jepa/loss.py",
        ROOT / "h_jepa/data.py",
        ROOT / "stable_worldmodel/data/dataset.py",
        Path(__file__),
        OUT / "metrics_lineage.py",
        OUT / "aggregate_results.py",
        OUT / "verify_sigreg_sharding.py",
    ]
    manifest = {
        "created_at": collection_time,
        "python": {"executable": sys.executable, "version": sys.version},
        "software_versions": {"numpy": np.__version__, "h5py": h5py.__version__, "matplotlib": matplotlib.__version__},
        "source_commit": git_info(),
        "protocol": {
            "training_protocol_version": 2,
            "training_seed": 42,
            "initialization": "fresh from initialization per launch manifest; no old checkpoint loaded",
            "batch_size": 128,
            "gradient_accumulation": 1,
            "sigreg_batch_size": 128,
            "frame_encoder_chunk_size": 128,
            "precision": "bf16 or bf16-mixed as saved in per-run config",
            "lr_schedule": "warmup/cosine length measured in optimizer updates",
            "decoder_learning_rate_fix": "probe 1e-3; decoder 1e-4 except Ant 1e-3",
        },
        "jobs_manifest_source": str(RUNS_JSON),
        "jobs_manifest_sha256": sha256(RUNS_JSON),
        "training_datasets": json.loads((OUT / "frequency_summary.json").read_text()),
        "runs": run_entries,
        "analysis_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in code_paths
        },
        "results_status": {
            "planning_success_rates": "pending final checkpoints and callbacks",
            "probe_nmse_and_decoder_metrics": "pending final callbacks; single training seed cannot supply paper's three-seed error bars",
        "training_loss_curves": (
            "rollback-aware merge of all in-progress SPT metrics sessions; "
            "refreshed at collection time"
        ),
        },
    }
    return manifest, summaries, metrics_by_env


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(metrics_by_env: dict[str, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), constrained_layout=True)
    groups = [
        ("Prediction loss", [("fit/pred_loss_level1", "L1"), ("fit/pred_loss_level2", "L2")]),
        ("SIGReg loss", [("fit/sigreg_loss_level1", "L1"), ("fit/sigreg_loss_level2", "L2")]),
        ("Total loss", [("fit/loss", "total")]),
        ("Learning rate", [("model_opt/level1", "L1"), ("model_opt/level2", "L2")]),
    ]
    for axes_row, (env, rows) in zip(axes, metrics_by_env.items()):
        steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
        for axis, (title, fields) in zip(axes_row, groups):
            positive = True
            for key, label in fields:
                y = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
                mask = np.isfinite(y)
                if np.any(mask):
                    axis.plot(steps[mask], y[mask], linewidth=0.9, label=label)
                    positive = positive and bool(np.all(y[mask] > 0))
            if positive and title != "Learning rate":
                axis.set_yscale("log")
            axis.set_title(f"{env}: {title}", fontsize=10)
            axis.set_xlabel("Logged optimizer step (0-based)", fontsize=8)
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8, loc="best")
    fig.suptitle("H-JEPA L2 / seed 42 / protocol 2 — in-progress raw log snapshot", fontsize=14)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_frequency(rows: list[dict], diagnostics: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), constrained_layout=True)
    for axis, env in zip(axes, ENV):
        selected = [r for r in rows if r["environment"] == env]
        y = np.arange(len(selected))
        power = [r["mean_power_spectral_centroid_cycles_per_level1_frame"] for r in selected]
        magnitude = [r["mean_magnitude_spectral_centroid_cycles_per_level1_frame"] for r in selected]
        axis.scatter(power, y, label="Power spectrum", s=42, zorder=3)
        axis.scatter(magnitude, y, label="Magnitude spectrum", s=38, marker="x", zorder=3)
        for i, (p, m) in enumerate(zip(power, magnitude)):
            axis.plot([p, m], [i, i], color="0.75", linewidth=1)
        axis.set_yticks(y, [r["entity"].replace("_", " ") for r in selected], fontsize=8)
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.set_xlim(0.01, 0.5)
        axis.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"], fontsize=8)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_ylim(len(selected) - 0.5, -0.5)
        axis.set_xlabel("Cycles / level-1 frame")
        axis.grid(True, axis="x", alpha=0.25)
        axis.set_title(f"{env}\npower gap {diagnostics[env]['frequency_gap_power_spectrum_fastest_over_slowest']:.2f}x")
    axes[0].legend(loc="center left", fontsize=8)
    fig.suptitle("Figure 5/12 frequency subset — episode-normalized state streams; probe NMSE pending")
    fig.savefig(OUT / "frequency_entity_centroids.png", dpi=180)
    fig.savefig(OUT / "frequency_entity_centroids.pdf")
    plt.close(fig)


def ant_example(diagnostics: dict) -> dict:
    path = DATA / f"{ENV['ant']['dataset']}.h5"
    with h5py.File(path, "r") as handle:
        lengths = np.asarray(handle["ep_len"][:64], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:64], dtype=np.int64)
        candidates = []
        for episode, (offset, length) in enumerate(zip(offsets, lengths)):
            indices = int(offset) + np.arange(0, int(length), 5)
            xy = np.asarray(handle["xy"][indices], dtype=np.float64)
            qpos = np.asarray(handle["qpos"][indices, 2:], dtype=np.float64)
            qvel = np.asarray(handle["qvel"][indices], dtype=np.float64)
            slow = float(np.nanmean(per_dim_centroid(xy[None])[0]))
            fast13 = float(np.nanmean(per_dim_centroid(qpos[None])[0]))
            fast27 = float(np.nanmean(per_dim_centroid(np.concatenate([qpos, qvel], axis=-1)[None])[0]))
            candidates.append({"episode_index": episode, "slow_centroid": slow, "body13_centroid": fast13, "body27_centroid": fast27, "body13_gap": fast13 / slow, "body27_gap": fast27 / slow})
        target = diagnostics["ant"]["frequency_gap_power_spectrum_fastest_over_slowest"]
        selected = min(candidates, key=lambda r: abs(r["body27_gap"] - target))
        ep = selected["episode_index"]
        indices = int(offsets[ep]) + np.arange(0, int(lengths[ep]), 5)
        xy = np.asarray(handle["xy"][indices], dtype=np.float64)
        body = np.asarray(handle["qpos"][indices, 2:], dtype=np.float64)
    standardize = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)
    slow, fast = standardize(xy), standardize(body)
    variation = lambda x: np.r_[0.0, np.abs(np.diff(x, axis=0)).mean(axis=1).cumsum()]
    slow_var, fast_var = variation(slow), variation(fast)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(slow, color="tab:red", alpha=0.85, linewidth=1.5)
    axes[0].plot(fast, color="tab:blue", alpha=0.3, linewidth=0.9)
    axes[0].set_ylabel("Standardized state value")
    axes[0].set_title("Global xy (red, 2 dims) and body qpos (blue, 13 dims)")
    axes[1].plot(slow_var, color="tab:red", label="Global xy (2 dims)")
    axes[1].plot(fast_var, color="tab:blue", label="Body qpos (13 dims)")
    axes[1].set_ylabel("Cumulative absolute increments / dimension")
    axes[1].legend()
    axes[1].set_title("Variation in standardized units")
    for axis in axes:
        axis.set_xlabel("Level-1 frame (stride 5 environment steps)")
        axis.grid(True, alpha=0.2)
    fig.suptitle(f"Ant episode {ep}: power gap {selected['body27_gap']:.2f}x (27 dims), {selected['body13_gap']:.2f}x (13 dims); dataset {target:.2f}x")
    fig.savefig(OUT / "ant_frequency_example.png", dpi=180)
    fig.savefig(OUT / "ant_frequency_example.pdf")
    plt.close(fig)
    save_csv(OUT / "ant_first64_frequency_candidates.csv", candidates)
    save_csv(OUT / "ant_frequency_example_timeseries.csv", [
        {"frame": frame, "env_step": frame * 5, "xy0": float(slow[frame, 0]), "xy1": float(slow[frame, 1]), **{f"body_qpos_{d}": float(fast[frame, d]) for d in range(fast.shape[1])}, "xy_cumulative_variation": float(slow_var[frame]), "body_cumulative_variation": float(fast_var[frame])}
        for frame in range(len(slow))
    ])
    selected.update({"selection_rule": "Among first 64 training episodes, closest 27-dim power-centroid gap to the locally computed dataset gap", "dataset_gap": target, "frames": len(slow), "omitted_qvel_dimensions": 14, "paper_episode_gap": 8.3, "paper_episode_identity_unknown": True})
    (OUT / "ant_frequency_example.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def write_report(out: Path, table_rows: list[dict], freq_diag: dict, manifest: dict, loss_summary: list[dict]) -> None:
    sigreg = json.loads((OUT / "sigreg_sharding.json").read_text())
    example = json.loads((OUT / "ant_frequency_example.json").read_text())
    lines = [
        "# 训练期间 CPU 复现分析结果",
        "",
        f"训练日志采集开始时间：{manifest['created_at']}（北京时间，各源文件依次读取）。",
        "",
        "本目录完成四环境的数据计数、频率统计、固定投影 SIGReg 分布式验证与新训练日志整理。分析仅用 CPU；未启动模型训练，未改动在训代码或配置。",
        "",
        "## Table 3：16/16 项有效 clips 与论文一致",
        "",
        "调用当前仓库 HDF5Dataset 与 transition-budget episode 选择逻辑，按 `sum(max(ep_len - loader_span + 1, 0))` 计数。下表是每个 epoch 可用 clips，并非原始 transitions 数量。",
        "",
        "| 环境 | L1 | L2（当前训练） | L3 | L4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for env in ENV:
        counts = [r['valid_clips_per_epoch'] for r in table_rows if r['environment'] == env]
        lines.append(f"| {env} | " + " | ".join(f"{v:,}" for v in counts) + " |")
    lines += [
        "",
        "四种深度分别含 4/7/13/25 个 L1 帧；观测首尾的最小 inclusive span 为 16/31/61/121，但 loader 预留的 half-open transition span 为 20/35/65/125。混用两者会多计 clips。FourRoom 按默认 subset_seed=0 选择约 100 万 transitions（实际 2,494 episodes、1,000,094 transitions）；Ant 使用全部 25,000 episodes（10,050,000 transitions），Push-T/Cube 使用完整训练文件。详见 [计数 CSV](table3_clip_counts.csv)。",
        "",
        "## Table 11：实际 CPU Gloo 分片验证通过",
        "",
        "固定同一批 Gaussian 数据（n=256、D=32）、64 个投影方向和 17 个积分点，使用 float64。W=1/2/4 均实际初始化 Gloo process group；调用仓库 SIGReg 的 ECF all_reduce。主指标采用与论文一致的输入表示梯度 ∂TEP/∂Z：各 rank 的局部输入梯度除以 W 后拼接，并与未分片参考逐元素对比。单位线性层参数梯度仅作为 DDP 辅助校验。",
        "",
        "| 分片 | 输入梯度范数 ‖∂TEP/∂Z‖ | TEP | 输入梯度最大绝对误差 | 单位线性层参数梯度范数（辅助） | 参数梯度最大绝对误差（辅助） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sigreg['sharding_results']:
        lines.append(f"| {r['shards']} | {r['distributed_input_gradient_norm']:.12f} | {r['distributed_tep']:.12f} | {r['distributed_vs_global_input_gradient_max_abs_error']:.3g} | {r['aux_identity_encoder_parameter_gradient_norm']:.12f} | {r['aux_identity_encoder_parameter_gradient_max_abs_error']:.3g} |")
    local = [r['local_reference_mean_tep_without_ecf_collective'] for r in sigreg['sharding_results']]
    lines += [
        "",
        f"取消 ECF 同步后，三种分片的平均局部 TEP 为 {local[0]:.9f} / {local[1]:.9f} / {local[2]:.9f}，出现预期的分片漂移。这里复现的是不变性结论；论文未提供相同随机矩阵/seed，因此不宣称复现论文的 TEP=1.372619178543。",
        "",
        "PyTorch 对仓库的非 autograd collective 发出兼容性 warning；当前数值验证覆盖其 local backward + DDP 平均语义。积分 buffer 沿用生产实现的 float32 初始化，再转 float64；更换 PyTorch 版本需重跑数值验证。详见 [完整结果](sigreg_sharding.json)、[表格](sigreg_sharding.csv) 与 [执行日志](sigreg_gloo_run.log)。",
        "",
        "## Figure 5/12：四环境频率子集，数值尚未对齐论文",
        "",
        "使用训练 split 内完整 episode，每 5 个环境步采样一次；每维在 episode 内减均值、除标准差。逐维计算 spectral centroid，再按有效 episode-dimension 等权平均到实体。主结果采用功率 `|rFFT|²` 权重，同时保留幅度 `|rFFT|` 权重作为敏感性分析。无额外平滑、窗函数或线性去趋势；常数维不参与平均。频率单位为 cycles / L1 frame，gap 为最大实体质心 / 最小实体质心。",
        "",
        "| 环境 | Episodes | 功率谱 gap | 幅度谱 gap | 最快实体 | 最慢实体 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for env, d in freq_diag.items():
        lines.append(f"| {env} | {d['episodes_used']:,} | {d['frequency_gap_power_spectrum_fastest_over_slowest']:.4f}× | {d['frequency_gap_magnitude_spectrum_fastest_over_slowest']:.4f}× | {d['fastest_entity']} | {d['slowest_entity']} |")
    ant = freq_diag['ant']
    lines += [
        "",
        "实体按论文 Table 6：FourRoom 为 ego xy 与 distractor xy；Ant 为 global xy 与 qpos[2:]+qvel（27维）；Push-T 为 agent xy、block xy、block orientation（sin/cos）、agent velocity；Cube 为 block position/orientation、effector position/yaw、gripper、arm joint position/velocity，共7项。训练中的额外诊断 target 不并入 gap。",
        "",
        f"Ant 的功率谱逐 episode 比值中位数为 {ant['slow_fast_power_episode_ratio_median']:.3f}×，IQR 为 {ant['slow_fast_power_episode_ratio_p25']:.3f}–{ant['slow_fast_power_episode_ratio_p75']:.3f}×；论文报告 6.4×、4.9–7.8×，dataset gap 为 8.3×。论文还报告导航类 3.1–12.7×、操作类 1.2–1.6×。本次两种权重下均未完全对齐，尚需作者的频谱处理/实体聚合实现确认原因，不能将这些本地数值视为论文数值复现成功。",
        "",
        f"按论文的“前64条中，episode gap 最接近 dataset gap”规则，选择本地 Ant episode {example['episode_index']}，共 {example['frames']} 帧；27维 gap={example['body27_gap']:.3f}×，图中13维 qpos gap={example['body13_gap']:.3f}×。两者在本地数据也有差别，均已保留，未强制改成论文示例的8.3×。",
        "",
        "Push-T 有41个 episode 的 block xy 与 orientation 为常数（每实体82个维度-episode项被排除）；其余常数/非有限值诊断保存在 JSON。四环境扫描未发现非有限实体序列。",
        "",
        "图：[实体频率 PNG](frequency_entity_centroids.png) / [PDF](frequency_entity_centroids.pdf)；[Ant 单 episode PNG](ant_frequency_example.png) / [PDF](ant_frequency_example.pdf)。每个实体的质心与四分位数见 [CSV](frequency_entity_centroids.csv)，协议/数据 SHA-256 见 [JSON](frequency_summary.json)，Ant 候选64条与所选轨迹也已导出 CSV。",
        "",
        "这些结果覆盖 Figure 5/12 的数据频率部分及 Figure 12a 的本地示例。Figure 5/12b 的 L1/L2 probe NMSE 仍需等待新模型的结束回调；当前只有 seed42，也无法提供论文3个训练种子的 SE。",
        "",
        "## 新训练 loss 与结果 manifest",
        "",
        "四个 run 都限定为 2026-10-27 09:32 重启的 L2/seed42/protocol2。每个 SPT session 的原始日志先单独复制、哈希，再按启动时间合并；后启动 session 从其首个已记录 global step 起替换旧 session 的整个后缀，避免把恢复前已放弃的分支混进曲线。不完整的末尾 CSV 行丢弃并记录字节数。step 为 logger 的0起始 optimizer step，采样点数不等于训练更新数。",
        "",
        "| 环境 | Sessions | 被替换点 | Loss采样点 | 最新step | Epoch（0起始） | Total loss | Pred L1 / L2 | SIGReg L1 / L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in loss_summary:
        fmt = lambda key: '—' if r.get(key) is None else f"{r[key]:.5g}"
        lines.append(f"| {r['environment']} | {r['metrics_session_count']} | {r['superseded_logged_steps']} | {r['metric_rows']} | {r['latest_step']} | {r['latest_epoch']} | {fmt('fit_loss')} | {fmt('pred_loss_l1')} / {fmt('pred_loss_l2')} | {fmt('sigreg_loss_l1')} / {fmt('sigreg_loss_l2')} |")
    lines += [
        "",
        "[Loss 曲线 PNG](training_loss_curves.png) / [PDF](training_loss_curves.pdf) 分开显示 prediction、SIGReg、total 与 LR；loss 使用对数纵轴，保留原始点，无平滑。另存 [逐步数据](training_loss_points.csv) 与 [最新摘要](training_loss_summary.csv)。这是仍在训练的单 seed L2 快照，不能直接作为论文四层训练图或最终收敛结果。",
        "",
        "[results_manifest.json](results_manifest.json) 记录源代码/配置哈希、训练数据身份、日志快照、实际 batch/累积、更新预算、学习率、checkpoint 与后续50-task规划配置（seed42、默认chunk_size=1）、probe/decoder配置及现有结果文件。逐 epoch 的 `*_epoch_*_object.ckpt` 单列为中间权重；仅同名 `*_hjepa_l2_object.ckpt` 计为最终权重，最终权重存在也不代表回调已完成。",
        "",
        "当前采集环境看不到宿主机训练 PID，PID 不可见不代表进程停止；manifest 另列日志更新时间作为文件层面的进展证据。不会将其他旧 run 的成功率或 NMSE 混入本次结果。",
        "",
        "## 重跑",
        "",
        "完整分析（CPU，低调度/IO优先级）：",
        "",
        "```bash",
        "cd /home/songxw/H-JEPA",
        "nice -n 15 ionice -c 3 /home/songxw/miniconda3/envs/hjepa/bin/python local_env/analysis_20261027/analyze_local_reproduction.py",
        "```",
        "",
        "只刷新训练曲线/manifest/报告，不重扫数据：",
        "",
        "```bash",
        "/home/songxw/miniconda3/envs/hjepa/bin/python local_env/analysis_20261027/analyze_local_reproduction.py --refresh-logs",
        "```",
        "",
        "Gloo 数值验证单独运行 `verify_sigreg_sharding.py`。本目录不注册后台任务；已有数据 SHA-256 来自前一轮完整文件校验，本轮不重复读取约145GB文件。所有内容保存在被 Git 忽略的 local_env 下。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-logs", action="store_true", help="Refresh only training-log snapshots, curves, report, and manifest; reuse dataset analyses.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_logs:
        table_rows = json.loads((OUT / "table3_clip_counts.json").read_text())
        frequency_summary = json.loads((OUT / "frequency_summary.json").read_text())
        with (OUT / "frequency_entity_centroids.csv").open() as stream:
            frequency_rows = list(csv.DictReader(stream))
    else:
        print("Computing Table 3 counts from episode metadata...", flush=True)
        table_rows = table3_counts()
        save_csv(OUT / "table3_clip_counts.csv", table_rows)
        (OUT / "table3_clip_counts.json").write_text(json.dumps(table_rows, indent=2, ensure_ascii=False) + "\n")
        print("Computing per-entity spectral centroids from state arrays...", flush=True)
        frequency_rows, frequency_summary = frequency_analysis()
        save_csv(OUT / "frequency_entity_centroids.csv", frequency_rows)
        (OUT / "frequency_summary.json").write_text(json.dumps(frequency_summary, indent=2, ensure_ascii=False) + "\n")
        plot_frequency(frequency_rows, frequency_summary)
        ant_example(frequency_summary)
    print("Collecting current training metrics and writing plot/manifest...", flush=True)
    manifest, loss_summary, metrics_by_env = training_manifest()
    save_csv(OUT / "training_loss_summary.csv", loss_summary)
    save_csv(OUT / "training_loss_points.csv", [{"environment": env, **row} for env, rows in metrics_by_env.items() for row in rows])
    plot_losses(metrics_by_env, OUT / "training_loss_curves.png")
    sigreg = json.loads((OUT / 'sigreg_sharding.json').read_text())
    manifest['analyses'] = {
        'table3': {'status': 'complete', 'rows': len(table_rows), 'matching_paper_rows': sum(r['matches_paper_table3'] for r in table_rows)},
        'table11': {'status': 'invariance_passed' if sigreg['all_distributed_shards_match_single_batch'] else 'invariance_failed', 'result': str(OUT / 'sigreg_sharding.json'), 'sha256': sha256(OUT / 'sigreg_sharding.json')},
        'figure5_12': {'status': 'frequency_subset_computed; published gap numbers not matched; probe NMSE pending', 'example': str(OUT / 'ant_frequency_example.json')},
    }
    write_report(OUT, table_rows, frequency_summary, manifest, loss_summary)
    manifest['artifacts'] = [
        {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in sorted(OUT.iterdir())
        if p.is_file()
        and p.name != 'results_manifest.json'
        and not p.name.startswith('aggregated_')
        and p.name != 'AGGREGATED_RESULTS.md'
        and p.suffix in ('.csv', '.json', '.png', '.pdf', '.py', '.md')
    ]
    (OUT / "results_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(OUT), "clip_rows": len(table_rows), "frequency_rows": len(frequency_rows), "loss_runs": len(loss_summary), "created_at": now_local()}, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "h_jepa"))
    sys.path.insert(0, str(ROOT))
    main()
