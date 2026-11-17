#!/usr/bin/env python3
"""Measure entity spectral centroids and relate them to probe NMSE (Figures 5/12)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from analysis_utils import add_common_output_args, default_data_root, pyplot, read_csv, save_figure, write_csv


DEFAULT_SPEC = {
    "fourroom": {
        "dataset": "fourroom_tp35_d1.h5",
        "frameskip": 5,
        "entities": {"xy": "xy", "distractor0_xy": "distractor0_xy"},
        "roles": {"xy": "retained", "distractor0_xy": "discarded"},
    },
    "ant": {
        "dataset": "visual_antmaze_medium_explore_stitch_train.h5",
        "frameskip": 5,
        "entities": {
            "xy": "xy",
            "proprio": [
                {"key": "qpos", "start": 2},
                {"key": "qvel"},
            ],
        },
        "roles": {"xy": "retained", "proprio": "discarded"},
    },
    "pusht": {
        "dataset": "pusht_expert_train.h5",
        "frameskip": 5,
        "entities": {
            "agent_xy": {"key": "state", "start": 0, "end": 2},
            "block_xy": {"key": "state", "start": 2, "end": 4},
            "block_ori": "block_ori",
            "agent_vel": {"key": "state", "start": 5, "end": 7},
        },
    },
    "cube": {
        "dataset": "cube_single_expert_train.h5",
        "frameskip": 5,
        "entities": {
            "block_pos": {"key": "observation", "start": 19, "end": 22},
            "block_ori": {"key": "observation", "start": 22, "end": 28},
            "effector_pos": {"key": "observation", "start": 12, "end": 15},
            "effector_yaw": {"key": "observation", "start": 15, "end": 17},
            "gripper": {"key": "observation", "start": 17, "end": 19},
            "arm_joint_pos": {"key": "observation", "start": 0, "end": 6},
            "arm_joint_vel": {"key": "observation", "start": 6, "end": 12},
        },
    },
}

METHOD = {
    "level1_sampling": "take every frameskip-th state independently within each episode",
    "normalization": "zero mean and unit variance per episode and scalar dimension",
    "spectrum_weight": "squared rFFT magnitude (periodogram), excluding DC",
    "entity_pooling": "sum weighted-frequency numerators and spectral-power denominators across dimensions and episodes",
    "paper_disclosure": (
        "Appendix F specifies amplitude normalization, level-1 sampling, and pooling over episodes, "
        "but does not publish the FFT weighting or exact pooling formula; periodogram weighting is "
        "selected because it matches the reported Ant per-episode ratio statistics much better than magnitude weighting."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="JSON mapping environments to dataset, frameskip, and entity column specs.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment to process; repeat as needed (default: all available defaults).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Deterministic prefix of episodes to use; zero uses every episode.",
    )
    parser.add_argument(
        "--centroids-csv",
        type=Path,
        help="Reuse a prior spectral_centroids.csv instead of rescanning HDF5 datasets.",
    )
    parser.add_argument(
        "--probe-csv",
        type=Path,
        help="probe_summary.csv from analyze_probes.py; enables Figures 5 and 12(b).",
    )
    add_common_output_args(parser, "frequency")
    return parser.parse_args()


def _source_array(handle: h5py.File, source, start: int, end: int) -> np.ndarray:
    if isinstance(source, str):
        data = handle[source][start:end]
    elif isinstance(source, dict):
        data = handle[source["key"]][start:end]
        data = data[..., slice(source.get("start"), source.get("end"))]
    elif isinstance(source, list):
        parts = [_source_array(handle, part, start, end) for part in source]
        data = np.concatenate([part.reshape(len(part), -1) for part in parts], axis=-1)
    else:
        raise TypeError(f"Unsupported entity source: {source!r}")
    return np.asarray(data, dtype=np.float64).reshape(len(data), -1)


def _normalized(sequence: np.ndarray) -> np.ndarray:
    mean = np.nanmean(sequence, axis=0, keepdims=True)
    std = np.nanstd(sequence, axis=0, keepdims=True)
    valid = np.isfinite(std[0]) & (std[0] > 1e-12)
    if not valid.any():
        return np.empty((len(sequence), 0), dtype=np.float64)
    return (sequence[:, valid] - mean[:, valid]) / std[:, valid]


def _spectral_terms(sequence: np.ndarray) -> tuple[float, float, int]:
    sequence = _normalized(sequence)
    if len(sequence) < 4 or sequence.shape[1] == 0:
        return 0.0, 0.0, 0
    # The paper's reported Ant per-episode median/IQR are obtained from the
    # periodogram (squared FFT magnitude), after amplitude normalization.
    spectrum = np.abs(np.fft.rfft(sequence, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(sequence), d=1.0)
    spectrum = spectrum[1:]
    frequencies = frequencies[1:]
    denominator = float(spectrum.sum())
    numerator = float((spectrum * frequencies[:, None]).sum())
    return numerator, denominator, int(sequence.shape[1])


def dataset_centroids(path: Path, spec: dict, max_episodes: int) -> tuple[list[dict], dict | None]:
    rows = []
    representative = None
    with h5py.File(path, "r") as handle:
        lengths = np.asarray(handle["ep_len"][:], dtype=int)
        offsets = np.asarray(handle["ep_offset"][:], dtype=int)
        count = len(lengths) if max_episodes <= 0 else min(max_episodes, len(lengths))
        accumulators = {
            entity: {"num": 0.0, "den": 0.0, "episodes": 0, "dims": set()}
            for entity in spec["entities"]
        }
        ant_candidates = []
        for episode in range(count):
            lo = int(offsets[episode])
            hi = lo + int(lengths[episode])
            episode_values = {}
            for entity, source in spec["entities"].items():
                sequence = _source_array(handle, source, lo, hi)[:: int(spec["frameskip"])]
                episode_values[entity] = sequence
                numerator, denominator, dimensions = _spectral_terms(sequence)
                if denominator <= 0:
                    continue
                acc = accumulators[entity]
                acc["num"] += numerator
                acc["den"] += denominator
                acc["episodes"] += 1
                acc["dims"].add(dimensions)
            if "xy" in episode_values and "proprio" in episode_values and episode < 64:
                slow = _spectral_terms(episode_values["xy"])
                fast = _spectral_terms(episode_values["proprio"])
                if slow[1] > 0 and fast[1] > 0:
                    ratio = (fast[0] / fast[1]) / (slow[0] / slow[1])
                    ant_candidates.append((ratio, episode, episode_values))

        centroids = {}
        for entity, acc in accumulators.items():
            centroid = acc["num"] / acc["den"] if acc["den"] else math.nan
            centroids[entity] = centroid
            rows.append(
                {
                    "entity": entity,
                    "centroid_cycles_per_frame": centroid,
                    "period_frames": 1.0 / centroid if centroid > 0 else math.nan,
                    "n_episodes": acc["episodes"],
                    "dimensions": ";".join(map(str, sorted(acc["dims"]))),
                }
            )
        finite = [value for value in centroids.values() if np.isfinite(value) and value > 0]
        gap = max(finite) / min(finite) if len(finite) >= 2 else math.nan
        for row in rows:
            row["frequency_gap"] = gap

        if ant_candidates and np.isfinite(gap):
            episode_gap, episode, values = min(ant_candidates, key=lambda item: abs(item[0] - gap))
            representative = {"episode": episode, "gap": episode_gap, "dataset_gap": gap, **values}
    return rows, representative


def plot_representative(payload: dict, output_dir: Path) -> None:
    slow = _normalized(payload["xy"])
    # The paper displays the 13 position-like body dimensions and omits velocities.
    fast = _normalized(payload["proprio"][:, :13])
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    axes[0].plot(slow, color="#b2182b", linewidth=1.2, alpha=0.8)
    axes[0].plot(fast, color="#2166ac", linewidth=0.55, alpha=0.35)
    axes[0].set(xlabel="Level-1 frame", ylabel="Normalized state", title=f"Ant episode {payload['episode']}")
    axes[1].plot(np.abs(np.diff(slow, axis=0)).cumsum(0).mean(1), color="#b2182b", label="position")
    axes[1].plot(np.abs(np.diff(fast, axis=0)).cumsum(0).mean(1), color="#2166ac", label="body state")
    axes[1].set(xlabel="Level-1 frame", ylabel="Cumulative absolute increment", title=f"gap={payload['gap']:.1f}x")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_12a")
    plt.close(fig)


def _probe_lookup(path: Path) -> dict[tuple[str, str, int], dict]:
    lookup = {}
    for row in read_csv(path):
        # Appendix F.2 pairs levels 1 and 2 from two-level H-JEPA models.
        if row["model"] != "hjepa_l2":
            continue
        key = (row["env"], row["entity"], int(row["level"]))
        if key in lookup:
            raise ValueError(f"Duplicate probe summary cell for {key} in {path}")
        lookup[key] = row
    return lookup


def _figure5_points(rows: list[dict], lookup: dict[tuple[str, str, int], dict]) -> list[dict]:
    env_rows = {}
    for row in rows:
        env_rows.setdefault(row["env"], []).append(row)

    points = []
    for env, values in env_rows.items():
        ordered = sorted(values, key=lambda row: row["centroid_cycles_per_frame"])
        explicit_roles = [row for row in ordered if row.get("role") in {"retained", "discarded"}]
        if explicit_roles:
            plotted = [(row["role"], row) for row in explicit_roles]
        elif float(ordered[0]["frequency_gap"]) < 2.0:
            # Figure 5 summarizes an environment with no separated frequency
            # band as one open "all retained" point. Figure 12(b) below remains
            # per-entity.
            deltas = []
            for row in ordered:
                level1 = lookup.get((env, row["entity"], 1))
                level2 = lookup.get((env, row["entity"], 2))
                if level1 and level2:
                    deltas.append(float(level2["nmse_mean"]) - float(level1["nmse_mean"]))
            if deltas:
                points.append(
                    {
                        **ordered[0],
                        "entity": "all",
                        "role": "retained",
                        "delta_nmse": float(np.mean(deltas)),
                        "n_entities": len(deltas),
                    }
                )
            plotted = []
        else:
            plotted = [("retained", ordered[0]), ("discarded", ordered[-1])]
        for role, row in plotted:
            level1 = lookup.get((env, row["entity"], 1))
            level2 = lookup.get((env, row["entity"], 2))
            if level1 and level2:
                points.append(
                    {
                        **row,
                        "role": role,
                        "delta_nmse": float(level2["nmse_mean"]) - float(level1["nmse_mean"]),
                        "n_entities": 1,
                    }
                )
    return points


def plot_frequency_and_probes(rows: list[dict], probe_csv: Path, output_dir: Path) -> None:
    lookup = _probe_lookup(probe_csv)
    points = _figure5_points(rows, lookup)

    plt = pyplot()
    if points:
        fig, axis = plt.subplots(figsize=(5.0, 3.4))
        env_labels = {
            "fourroom": "FourRoom",
            "ant": "AntMaze",
            "pusht": "Push-T",
            "cube": "Cube",
            "droid": "DROID",
            "humanoid": "Humanoid",
        }
        entity_labels = {"xy": "position", "proprio": "body", "distractor0_xy": "distractor"}
        for point in points:
            filled = point["role"] == "discarded"
            axis.scatter(
                point["frequency_gap"],
                point["delta_nmse"],
                facecolor="#b2182b" if filled else "white",
                edgecolor="#b2182b" if filled else "#2166ac",
                marker="o",
                s=48,
            )
            annotation = env_labels.get(point["env"], point["env"])
            if point["entity"] != "all":
                annotation += f"\n{entity_labels.get(point['entity'], point['entity'].replace('_', ' '))}"
            axis.annotate(
                annotation,
                (point["frequency_gap"], point["delta_nmse"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.axvspan(1.0, 2.0, color="#dddddd", alpha=0.5)
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_xscale("log")
        axis.set(xlabel="Entity-frequency gap (x)", ylabel="L2 NMSE - L1 NMSE", title="Frequency separation and selective abstraction")
        fig.tight_layout()
        save_figure(fig, output_dir / "figure_5")
        plt.close(fig)
        write_csv(output_dir / "figure_5.csv", points)

    paired = []
    for row in rows:
        for level in (1, 2):
            probe = lookup.get((row["env"], row["entity"], level))
            if probe:
                paired.append({**row, "level": level, "nmse": float(probe["nmse_mean"]), "nmse_se": float(probe["nmse_se"])})
    if not paired:
        return
    env_order = {env: index for index, env in enumerate(("fourroom", "ant", "humanoid", "pusht", "cube", "droid"))}
    labels = sorted(
        {(row["env"], row["entity"]) for row in paired},
        key=lambda label: (env_order.get(label[0], len(env_order)), label[0], label[1]),
    )
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.75 * len(labels)), 5.6), sharex=True)
    centroids = {(row["env"], row["entity"]): row["centroid_cycles_per_frame"] for row in rows}
    axes[0].scatter(x, [centroids[label] for label in labels], color="#333333", s=28)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Spectral centroid\n(cycles / L1 frame)")
    for level, offset, color in ((1, -0.1, "#2166ac"), (2, 0.1, "#b2182b")):
        values = {(row["env"], row["entity"]): row for row in paired if row["level"] == level}
        present = [idx for idx, label in enumerate(labels) if label in values]
        axes[1].errorbar(
            np.asarray(present) + offset,
            [values[labels[idx]]["nmse"] for idx in present],
            yerr=[values[labels[idx]]["nmse_se"] for idx in present],
            fmt="o",
            capsize=2,
            color=color,
            label=f"level {level}",
        )
    axes[1].set(ylabel="Probe NMSE", xticks=x, xticklabels=[f"{env}\n{entity}" for env, entity in labels])
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_12b")
    plt.close(fig)
    write_csv(output_dir / "figure_12b.csv", paired)


def main() -> None:
    args = parse_args()
    if args.centroids_csv:
        rows = read_csv(args.centroids_csv)
        if not rows:
            raise SystemExit(f"Centroid CSV is empty: {args.centroids_csv}")
        numeric = ("centroid_cycles_per_frame", "period_frames", "frequency_gap")
        all_rows = [
            {
                **row,
                **{name: float(row[name]) for name in numeric if row.get(name, "") != ""},
                "n_episodes": int(float(row["n_episodes"])),
                "frameskip": int(float(row["frameskip"])),
            }
            for row in rows
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "frequency_method.json").write_text(json.dumps(METHOD, indent=2) + "\n")
        if args.probe_csv:
            plot_frequency_and_probes(all_rows, args.probe_csv, args.output_dir)
        print(f"Reused {len(all_rows)} entity centroids from {args.centroids_csv}")
        return

    spec = json.loads(args.spec.read_text()) if args.spec else DEFAULT_SPEC
    selected = set(args.env) if args.env else set(spec)
    all_rows = []
    representative = None
    for env in sorted(selected):
        if env not in spec:
            raise SystemExit(f"Environment {env!r} is absent from the analysis spec.")
        env_spec = spec[env]
        dataset = Path(env_spec["dataset"])
        if not dataset.is_absolute():
            dataset = args.data_root / dataset
        if not dataset.exists():
            print(f"skip {env}: missing {dataset}")
            continue
        rows, payload = dataset_centroids(dataset, env_spec, args.max_episodes)
        for row in rows:
            row.update(
                {
                    "env": env,
                    "dataset": str(dataset),
                    "frameskip": int(env_spec["frameskip"]),
                    "role": env_spec.get("roles", {}).get(row["entity"], ""),
                }
            )
        all_rows.extend(rows)
        if env == "ant":
            representative = payload
    if not all_rows:
        raise SystemExit("No datasets from the requested specification were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frequency_method.json").write_text(json.dumps(METHOD, indent=2) + "\n")
    write_csv(args.output_dir / "spectral_centroids.csv", all_rows)
    if representative is not None:
        plot_representative(representative, args.output_dir)
    if args.probe_csv:
        plot_frequency_and_probes(all_rows, args.probe_csv, args.output_dir)
    print(f"Wrote {len(all_rows)} entity centroids to {args.output_dir}")


if __name__ == "__main__":
    main()
