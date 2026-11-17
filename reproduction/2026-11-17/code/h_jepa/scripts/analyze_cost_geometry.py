#!/usr/bin/env python3
"""Map distance to an anchor through a hierarchy of latent spaces (Figure 7)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from analysis_utils import accepted_run, add_common_output_args, default_data_root, pyplot, run_metadata, save_figure, write_csv

from data import IMAGENET_STATS, load_normalizer_artifact
from final_probing_decoding_eval import _dense_encode_no_stride
from utils import load_training_config_for_checkpoint, resolve_model_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="H-JEPA object checkpoint or containing run directory.")
    parser.add_argument("--dataset", type=Path, help="HDF5 source; defaults to the checkpoint validation dataset.")
    parser.add_argument(
        "--num-points",
        type=int,
        default=0,
        help="Number of deterministic random states; zero (default) encodes every state as in the paper.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-index", type=int)
    parser.add_argument("--anchor-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--coord-key", default="xy")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_output_args(parser, "cost_geometry")
    return parser.parse_args()


def _selected_column(handle: h5py.File, source, indices: np.ndarray) -> np.ndarray:
    # h5py requires increasing fancy indices.
    order = np.argsort(indices)
    inverse = np.argsort(order)
    sorted_indices = indices[order]
    if isinstance(source, str):
        return np.asarray(handle[source][sorted_indices])[inverse]
    if isinstance(source, (dict, DictConfig)):
        data = np.asarray(handle[source["key"]][sorted_indices])[inverse]
        return data[..., slice(source.get("start"), source.get("end"))]
    if isinstance(source, (list, tuple, ListConfig)):
        parts = [_selected_column(handle, item, indices) for item in source]
        return np.concatenate([part.reshape(len(indices), -1) for part in parts], axis=-1)
    raise TypeError(f"Unsupported column source: {source!r}")


def _normalizer(value: np.ndarray, artifact: dict, name: str) -> np.ndarray:
    stats = artifact["stats"].get(name)
    if stats is None:
        raise KeyError(f"normalizer.pt has no statistics for {name!r}")
    mean = np.asarray(stats["mean"]).reshape(-1)
    std = np.asarray(stats["std"]).reshape(-1)
    return (value.reshape(len(value), -1) - mean) / np.maximum(std, 1e-12)


def _pixel_tensor(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(pixels, device=device)
    if value.ndim != 4:
        raise ValueError(f"Expected four-dimensional images, got {tuple(value.shape)}")
    if value.shape[-1] in (1, 3):
        value = value.permute(0, 3, 1, 2)
    value = value.float().div(255.0) if value.dtype == torch.uint8 else value.float()
    mean = value.new_tensor(IMAGENET_STATS["mean"]).view(1, 3, 1, 1)
    std = value.new_tensor(IMAGENET_STATS["std"]).view(1, 3, 1, 1)
    return (value - mean) / std


@torch.no_grad()
def encode_points(model, pixels: np.ndarray, proprio: np.ndarray | None, batch_size: int, device: torch.device):
    per_level = {level: [] for level in range(1, int(model.num_levels) + 1)}
    for start in range(0, len(pixels), batch_size):
        stop = min(start + batch_size, len(pixels))
        batch = {"pixels_level1": _pixel_tensor(pixels[start:stop], device)[:, None]}
        if proprio is not None:
            batch["proprio_level1"] = torch.as_tensor(proprio[start:stop], dtype=torch.float32, device=device)[:, None]
        encoded = _dense_encode_no_stride(model, batch)
        for level in per_level:
            per_level[level].append(encoded[f"embed_{level}"][:, 0].float().cpu())
    return {level: torch.cat(chunks).numpy() for level, chunks in per_level.items()}


def main() -> None:
    args = parse_args()
    cache_dir = os.environ.get("HJEPA_HOME")
    checkpoint = resolve_model_checkpoint_path(args.checkpoint, cache_dir)
    metadata = run_metadata(checkpoint)
    if not accepted_run(metadata, 2):
        raise SystemExit(
            f"Refusing a checkpoint that is neither protocol-v2 nor in the verified release manifest: {checkpoint} "
            f"(training_protocol_version={metadata['protocol_version']!r})"
        )
    cfg = load_training_config_for_checkpoint(checkpoint)
    artifact = load_normalizer_artifact(checkpoint.parent / "normalizer.pt")
    dataset_cfg = cfg.data.dataset
    dataset = args.dataset
    if dataset is None:
        dataset = default_data_root() / f"{dataset_cfg.val_name}.h5"
    if not dataset.exists():
        raise SystemExit(f"Dataset does not exist: {dataset}")

    rng = np.random.default_rng(args.seed)
    with h5py.File(dataset, "r") as handle:
        total = int(handle["pixels"].shape[0])
        count = total if args.num_points <= 0 else min(args.num_points, total)
        indices = np.arange(total) if count == total else np.sort(rng.choice(total, size=count, replace=False))
        coords_all = handle[args.coord_key]
        if args.anchor_xy is not None:
            target = np.asarray(args.anchor_xy, dtype=float)
            best_distance = np.inf
            anchor_index = 0
            for start in range(0, total, 100_000):
                values = np.asarray(coords_all[start : start + 100_000], dtype=float)[..., :2]
                local = int(np.square(values - target).sum(-1).argmin())
                distance = float(np.square(values[local] - target).sum())
                if distance < best_distance:
                    best_distance = distance
                    anchor_index = start + local
        else:
            anchor_index = int(args.anchor_index if args.anchor_index is not None else indices[len(indices) // 2])
        if anchor_index < 0 or anchor_index >= total:
            raise SystemExit(f"anchor index {anchor_index} is outside [0, {total})")
        indices = np.unique(np.concatenate([[anchor_index], indices]))
        anchor_position = int(np.flatnonzero(indices == anchor_index)[0])
        pixels = _selected_column(handle, "pixels", indices)
        coords = _selected_column(handle, args.coord_key, indices).reshape(len(indices), -1)[:, :2]
        proprio = None
        if bool(cfg.level1.wm.get("use_proprio", False)):
            source = dataset_cfg.keys_to_merge.get("proprio", "proprio")
            proprio = _normalizer(_selected_column(handle, source, indices), artifact, "proprio")

    device = torch.device(args.device)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval().to(device)
    embeddings = encode_points(model, pixels, proprio, args.batch_size, device)
    result_rows = []
    distances = {}
    normalized_distances = {}
    for level, values in embeddings.items():
        anchor = values[anchor_position]
        distances[level] = np.linalg.norm(values - anchor[None], axis=1)
        scale = max(float(distances[level].max()), np.finfo(float).tiny)
        normalized_distances[level] = distances[level] / scale
    for row_index, source_index in enumerate(indices):
        row = {"row_index": int(source_index), "x": float(coords[row_index, 0]), "y": float(coords[row_index, 1])}
        row.update({f"distance_level{level}": float(values[row_index]) for level, values in distances.items()})
        row.update(
            {
                f"normalized_distance_level{level}": float(values[row_index])
                for level, values in normalized_distances.items()
            }
        )
        result_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cost_geometry.csv", result_rows)
    manifest = {
        "paper_reference": "Figure 7",
        "status": "qualified_reproduction",
        "qualification": (
            "The paper does not publish the anchor coordinates or color normalization. "
            "Raw Euclidean distances are retained; the 0-1 visualization divides each level by its maximum."
        ),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "states": len(indices),
        "sampling": "all validation states" if args.num_points <= 0 else f"deterministic {len(indices)}-state subset",
        "anchor_index": int(anchor_index),
        "anchor_xy": [float(coords[anchor_position, 0]), float(coords[anchor_position, 1])],
        "requested_anchor_xy": None if args.anchor_xy is None else list(args.anchor_xy),
        "distance": "Euclidean L2 in each level's latent space",
        "plot_normalization": "per-level distance / per-level maximum distance",
    }
    (args.output_dir / "cost_geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt = pyplot()
    fig, axes = plt.subplots(1, len(distances), figsize=(3.5 * len(distances), 3.2), squeeze=False)
    scatter = None
    for axis, (level, values) in zip(axes[0], normalized_distances.items()):
        scatter = axis.scatter(coords[:, 0], coords[:, 1], c=values, s=5, cmap="turbo", vmin=0, vmax=1, rasterized=True)
        axis.scatter(coords[anchor_position, 0], coords[anchor_position, 1], marker="*", s=130, color="black", edgecolor="white", linewidth=0.8)
        axis.set(title=f"Level {level}", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(
        scatter,
        ax=axes[0].tolist(),
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        label="Normalized latent distance to anchor",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.24, wspace=0.28)
    save_figure(fig, args.output_dir / "figure_7")
    plt.close(fig)
    print(f"Encoded {len(indices)} states from {dataset}; outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
