"""Shared, dependency-light helpers for the paper analysis scripts."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


HJEPA_DIR = Path(__file__).resolve().parents[1]
if str(HJEPA_DIR) not in sys.path:
    sys.path.insert(0, str(HJEPA_DIR))


def default_data_root() -> Path:
    """Resolve the repository's data root without requiring a shell export."""
    if "HJEPA_HOME" in os.environ:
        return Path(os.environ["HJEPA_HOME"]).expanduser()
    candidates = (Path.home() / "hjepa-data", Path.home() / ".stable_worldmodel")
    return next((path for path in candidates if path.exists()), candidates[-1])


def existing_paths(values: Sequence[str | Path]) -> list[Path]:
    """Expand files, directories, and glob expressions without duplicates."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        raw = Path(value).expanduser()
        matches = [raw] if raw.exists() else [Path(match) for match in glob.glob(str(raw), recursive=True)]
        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def find_files(inputs: Sequence[str | Path], names: Sequence[str]) -> list[Path]:
    """Find files named in ``names`` below each input path."""
    found: list[Path] = []
    seen: set[Path] = set()
    for path in existing_paths(inputs):
        candidates = [path] if path.is_file() else [p for name in names for p in path.rglob(name)]
        for candidate in candidates:
            if candidate.name not in names:
                continue
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
    return sorted(found)


def load_yaml(path: str | Path) -> dict:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def run_metadata(path: str | Path) -> dict[str, object]:
    """Infer environment, model, and seed from the closest training config/path."""
    path = Path(path).resolve()
    cursor = path if path.is_dir() else path.parent
    closest_config = None
    closest_dir = None
    training_config = None
    training_dir = None
    for parent in (cursor, *cursor.parents):
        candidate = parent / "config.yaml"
        if candidate.exists():
            value = load_yaml(candidate)
            if closest_config is None:
                closest_config = value
                closest_dir = parent
            if "env" in value and "output_model_name" in value:
                training_config = value
                training_dir = parent
                break

    config = training_config or closest_config
    run_dir = training_dir or closest_dir

    text = str(path)
    seed_match = re.search(r"(?:^|/)seed(-?\d+)(?:[^/]*)?(?:/|$)", text)
    seed = int(seed_match.group(1)) if seed_match else None
    env = None
    model = None
    if config:
        env = config.get("env")
        model = config.get("output_model_name")
        try:
            seed = int(config.get("seed", seed))
        except (TypeError, ValueError):
            pass
    if model is None:
        model_match = re.search(
            r"(?:^|/)(ant|fourroom|cube|pusht|droid)_"
            r"(lewm|hwm_l\d+|hjepa_l\d+)(?:/|$)",
            text,
        )
        if model_match:
            env = env or model_match.group(1)
            model = f"{model_match.group(1)}_{model_match.group(2)}"
    short_model = str(model or "unknown")
    if env and short_model.startswith(f"{env}_"):
        short_model = short_model[len(str(env)) + 1 :]
    protocol_version = config.get("training_protocol_version") if config else None
    try:
        protocol_version = int(protocol_version)
    except (TypeError, ValueError):
        protocol_version = None
    source = "local"
    release_revision = None
    for parent in (cursor, *cursor.parents):
        manifest_path = parent / "OFFICIAL_WEIGHTS_MANIFEST.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = load_json(manifest_path)
            relative = path.relative_to(parent)
            parts = relative.parts
            if len(parts) >= 3:
                run_prefix = "/".join(parts[:3])
                files = manifest.get("files", {})
                if any(name == run_prefix or name.startswith(f"{run_prefix}/") for name in files):
                    source = "official_release"
                    release_revision = manifest.get("revision")
        except (OSError, ValueError, TypeError):
            pass
        break
    return {
        "env": str(env or "unknown"),
        "model": short_model,
        "seed": seed,
        "protocol_version": protocol_version,
        "source": source,
        "release_revision": release_revision,
        "run_dir": str(run_dir or cursor),
    }


def accepted_run(metadata: Mapping, protocol_version: int = 2) -> bool:
    """Accept repaired local runs or files rooted in the verified release manifest."""
    return (
        metadata.get("protocol_version") == protocol_version
        or metadata.get("source") == "official_release"
    )


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if array.size == 0:
        return math.nan, math.nan, 0
    se = 0.0 if array.size == 1 else float(array.std(ddof=1) / np.sqrt(array.size))
    return float(array.mean()), se, int(array.size)


def group_rows(rows: Iterable[Mapping], keys: Sequence[str]) -> dict[tuple, list[Mapping]]:
    groups: dict[tuple, list[Mapping]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def write_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def pareto_indices(cost: Sequence[float], score: Sequence[float]) -> np.ndarray:
    """Indices on the lower-cost/higher-score Pareto frontier."""
    cost_array = np.asarray(cost, dtype=float)
    score_array = np.asarray(score, dtype=float)
    order = np.argsort(cost_array, kind="stable")
    keep: list[int] = []
    best = -np.inf
    for idx in order:
        if not (np.isfinite(cost_array[idx]) and np.isfinite(score_array[idx])):
            continue
        if score_array[idx] > best:
            keep.append(int(idx))
            best = score_array[idx]
    return np.asarray(keep, dtype=int)


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 200,
        }
    )
    return plt


def save_figure(fig, output_stem: str | Path) -> None:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def add_common_output_args(parser, default_stem: str) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / default_stem,
        help="Directory for CSV and figure outputs.",
    )
