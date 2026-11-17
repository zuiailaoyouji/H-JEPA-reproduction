#!/usr/bin/env python3
"""Wait for the official queue, then run strict aggregation and verification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from analysis_utils import default_data_root, load_json
from run_official_reproduction import complete, droid_jobs, probe_jobs


HJEPA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = HJEPA_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--geometry-gpu", default="1")
    parser.add_argument("--scope", choices=("full", "probe-droid"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_status(path: Path, **values) -> None:
    payload = {"updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _commands(data_root: Path, revision: str, scope: str) -> list[tuple[str, list[str], dict[str, str]]]:
    python = sys.executable
    ckpt_root = data_root / "ckpts"
    output_dir = data_root / "analyse" / f"official_release_{revision[:8]}"
    if scope == "probe-droid":
        output_dir = output_dir / "probe_droid_final"
        return [
            (
                "aggregate_droid",
                [
                    python,
                    "scripts/analyze_official_release.py",
                    "--experiments",
                    "droid",
                    "--output-dir",
                    str(output_dir),
                ],
                {},
            ),
            (
                "aggregate_probes",
                [
                    python,
                    "scripts/analyze_probes.py",
                    str(ckpt_root),
                    "--model",
                    "hjepa_l2",
                    "--model",
                    "hjepa_l4",
                    "--require-seeds",
                    "3",
                    "--output-dir",
                    str(output_dir),
                ],
                {},
            ),
            (
                "plot_frequency_probes",
                [
                    python,
                    "scripts/analyze_frequency.py",
                    "--centroids-csv",
                    str(data_root / "analyse" / "spectral_centroids.csv"),
                    "--probe-csv",
                    str(output_dir / "probe_summary.csv"),
                    "--output-dir",
                    str(output_dir),
                ],
                {},
            ),
        ]

    tree = REPO_DIR / "local_env" / "official_reproduction_20261027" / f"model_tree_{revision}.json"
    commands = [
        ("aggregate_release", [python, "scripts/analyze_official_release.py"], {}),
        (
            "aggregate_probes",
            [
                python,
                "scripts/analyze_probes.py",
                str(ckpt_root),
                "--model",
                "hjepa_l2",
                "--model",
                "hjepa_l4",
                "--require-seeds",
                "3",
                "--output-dir",
                str(output_dir),
            ],
            {},
        ),
        (
            "plot_frequency_probes",
            [
                python,
                "scripts/analyze_frequency.py",
                "--centroids-csv",
                str(data_root / "analyse" / "spectral_centroids.csv"),
                "--probe-csv",
                str(output_dir / "probe_summary.csv"),
                "--output-dir",
                str(output_dir),
            ],
            {},
        ),
        (
            "figure7_cost_geometry",
            [
                python,
                "scripts/analyze_cost_geometry.py",
                str(ckpt_root / "ant" / "ant_hjepa_l3" / "seed42" / "ant_hjepa_l3_object.ckpt"),
                "--anchor-xy",
                "8",
                "8",
                "--output-dir",
                str(data_root / "analyse" / "qualified_figure7"),
            ],
            {"CUDA_VISIBLE_DEVICES": "{geometry_gpu}"},
        ),
        (
            "verify_weights",
            [
                python,
                "scripts/verify_official_weights.py",
                "--root",
                str(ckpt_root),
                "--tree-json",
                str(tree),
                "--revision",
                revision,
            ],
            {},
        ),
        ("tests", [python, "-m", "pytest", "tests", "-q"], {}),
    ]
    return commands


def main() -> None:
    args = parse_args()
    data_root = default_data_root()
    manifest = load_json(data_root / "ckpts" / "OFFICIAL_WEIGHTS_MANIFEST.json")
    revision = str(manifest["revision"])
    output_dir = data_root / "analyse" / f"official_release_{revision[:8]}"
    runner_status = output_dir / "runner_status_droid_probe_depth_cost.json"
    finalizer_status = output_dir / "finalizer_status.json"
    commands = _commands(data_root, revision, args.scope)
    commands = [
        (name, command, {key: value.format(geometry_gpu=args.geometry_gpu) for key, value in extra_env.items()})
        for name, command, extra_env in commands
    ]

    if args.dry_run:
        for name, command, extra_env in commands:
            prefix = " ".join(f"{key}={value}" for key, value in extra_env.items())
            print(name, "::", prefix, " ".join(command))
        return

    _write_status(finalizer_status, state="waiting", scope=args.scope)
    if args.scope == "probe-droid":
        watched = droid_jobs(data_root / "ckpts") + probe_jobs(data_root / "ckpts")
        while True:
            droid_done = sum(complete(job) for job in watched if job.kind == "droid")
            probe_done = sum(complete(job) for job in watched if job.kind.startswith("probe"))
            if droid_done == 27 and probe_done == 18:
                break
            _write_status(
                finalizer_status,
                state="waiting",
                scope=args.scope,
                droid_complete=droid_done,
                droid_total=27,
                probe_complete=probe_done,
                probe_total=18,
            )
            time.sleep(args.poll_seconds)
    else:
        while True:
            try:
                runner = json.loads(runner_status.read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(args.poll_seconds)
                continue
            failures = runner.get("failures", [])
            if failures:
                _write_status(finalizer_status, state="blocked", runner_failures=failures)
                raise SystemExit("runner reported failures; finalization was not started")
            if runner.get("complete") == runner.get("total") and not runner.get("active"):
                break
            _write_status(
                finalizer_status,
                state="waiting",
                scope=args.scope,
                runner_complete=runner.get("complete"),
                runner_total=runner.get("total"),
                runner_active=runner.get("active", []),
            )
            time.sleep(args.poll_seconds)

    completed = []
    base_env = os.environ.copy()
    base_env["HJEPA_HOME"] = str(data_root)
    for name, command, extra_env in commands:
        _write_status(finalizer_status, state="running", scope=args.scope, step=name, completed_steps=completed)
        env = {**base_env, **extra_env}
        result = subprocess.run(command, cwd=HJEPA_DIR, env=env)
        if result.returncode:
            _write_status(
                finalizer_status,
                state="failed",
                scope=args.scope,
                step=name,
                returncode=result.returncode,
                command=command,
                completed_steps=completed,
            )
            raise SystemExit(f"{name} failed with return code {result.returncode}")
        completed.append(name)
    _write_status(finalizer_status, state="complete", scope=args.scope, completed_steps=completed)


if __name__ == "__main__":
    main()
