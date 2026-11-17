#!/usr/bin/env python3
"""Aggregate H-JEPA training, planning, probing, and decoder evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

import yaml

from metrics_lineage import find_metric_sessions, merge_metric_sessions


ROOT = Path("/home/songxw/H-JEPA")
DEFAULT_CKPT_ROOT = Path("/home/songxw/hjepa-data/ckpts")
DEFAULT_OUTPUT_DIR = ROOT / "local_env/analysis_20261027"
FRESH_JOBS = ROOT / "local_env/fixes_20261027/restarted_l2_jobs.json"
TASK_CHECKSUMS = ROOT / "h_jepa/assets/eval_trajs/SHA256SUMS"


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


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text())
    return value if isinstance(value, dict) else {}


def nested(mapping: dict, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_rows"}]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def model_metadata(model: str, config: dict) -> tuple[str, int | None]:
    if "hjepa" in model:
        family = "hjepa"
    elif "hwm" in model:
        family = "hwm"
    elif "lewm" in model:
        family = "lewm"
    else:
        family = "unknown"
    depth = nested(config, "num_levels")
    if depth is None:
        match = re.search(r"_l(\d+)", model)
        depth = int(match.group(1)) if match else (1 if family == "lewm" else None)
    return family, int(depth) if depth is not None else None


def training_protocol(config: dict) -> tuple[str, int | None, int | None, int | None]:
    version = config.get("training_protocol_version")
    batch = nested(config, "loader", "batch_size")
    accumulation = nested(config, "trainer", "accumulate_grad_batches", default=1)
    if version is not None:
        name = f"v{version}_b{batch}_accum{accumulation}"
    elif batch is not None:
        name = f"legacy_or_unspecified_b{batch}_accum{accumulation}"
    else:
        name = "unknown"
    return name, version, batch, accumulation


def known_fresh_runs() -> set[Path]:
    if not FRESH_JOBS.is_file():
        return set()
    return {Path(item["run_dir"]).resolve() for item in json.loads(FRESH_JOBS.read_text())}


def load_provenance_overrides(path: Path | None) -> dict[Path, dict]:
    """Load explicit source labels; never infer that an external file is official."""
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    entries = raw.get("runs", raw) if isinstance(raw, dict) else raw
    if isinstance(entries, dict):
        entries = [dict(value, run_path=key) for key, value in entries.items()]
    if not isinstance(entries, list):
        raise ValueError("provenance manifest must contain a run list or run-path mapping")
    output = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("run_path"):
            raise ValueError("every provenance entry requires run_path")
        source = entry.get("model_weight_source")
        if source not in {
            "local_fresh_training",
            "local_training",
            "official_pretrained",
            "external_pretrained",
        }:
            raise ValueError(f"unsupported explicit model_weight_source: {source!r}")
        output[Path(entry["run_path"]).resolve()] = entry
    return output


def known_task_hashes() -> dict[Path, str]:
    output = {}
    if TASK_CHECKSUMS.is_file():
        for line in TASK_CHECKSUMS.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                output[(TASK_CHECKSUMS.parent / fields[1]).resolve()] = fields[0]
    manifest = DEFAULT_OUTPUT_DIR / "results_manifest.json"
    if manifest.is_file():
        try:
            for run in json.loads(manifest.read_text()).get("runs", []):
                planned = run.get("planned_evaluation", {})
                path, digest = planned.get("task_file"), planned.get("task_file_sha256")
                if path and digest:
                    output[Path(path).resolve()] = digest
        except (OSError, json.JSONDecodeError):
            pass
    return output


def resolve_input_path(raw, *, base: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [base / path, ROOT / "h_jepa" / path, ROOT / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def file_identity(path: Path | None, hashes: dict[Path, str]) -> dict:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    resolved = path.resolve()
    exists = resolved.is_file()
    digest = hashes.get(resolved)
    if digest is None and exists and resolved.stat().st_size <= 16 * 1024 * 1024:
        digest = sha256(resolved)
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "sha256": digest,
        "hash_note": None if digest else "large file not hashed and no trusted manifest hash found",
    }


def discover_run_dirs(ckpt_root: Path) -> list[Path]:
    runs = []
    if not ckpt_root.is_dir():
        return runs
    for environment_dir in sorted(path for path in ckpt_root.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in environment_dir.iterdir() if path.is_dir()):
            for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                evidence = [run_dir / "config.yaml", run_dir / "normalizer.pt", run_dir / "spt"]
                if any(path.exists() for path in evidence) or list(run_dir.glob("*_object.ckpt")):
                    runs.append(run_dir)
    return runs


def result_kind(relative: Path, config_name: str | None, *, droid: bool = False) -> str:
    first = relative.parts[0]
    if first == "planning_eval":
        return f"automatic_{config_name or 'hierarchical'}"
    label = first.removeprefix("eval_")
    prefix = "droid_" if droid else "standalone_"
    return prefix + label


def collect_sim_planning(
    run: dict, run_dir: Path, task_hashes: dict[Path, str]
) -> list[dict]:
    rows = []
    for metrics_path in sorted(run_dir.rglob("metrics.yaml")):
        relative = metrics_path.relative_to(run_dir)
        if not (
            "planning_eval" in relative.parts
            or any(part.startswith("eval_") for part in relative.parts)
        ):
            continue
        if any(part.startswith("final_probing_decoding_eval") for part in relative.parts):
            continue
        result_dir = metrics_path.parent
        eval_config_path = result_dir / "eval_config.yaml"
        config = load_yaml(eval_config_path)
        metrics = load_yaml(metrics_path)
        successes = metrics.get("episode_successes")
        successes = successes if isinstance(successes, list) else []
        expected = nested(config, "eval", "num_eval")
        expected = int(expected) if expected is not None else None
        completed = len(successes)
        reported = finite_float(metrics.get("success_rate"))
        recomputed = 100.0 * sum(value is True for value in successes) / completed if completed else None
        consistent = (
            reported is not None
            and recomputed is not None
            and abs(reported - recomputed) < 1e-8
        )
        complete = expected is not None and completed == expected and consistent
        task_path = resolve_input_path(
            config.get("load_eval_trajs_path"), base=ROOT / "h_jepa"
        )
        policy_path = resolve_input_path(config.get("policy"), base=run_dir)
        config_name = nested(run["training_config"], "planning_eval", "config_name")
        rows.append(
            {
                "run_path": str(run_dir),
                "environment": run["environment"],
                "model": run["model"],
                "model_family": run["model_family"],
                "model_depth": run["model_depth"],
                "training_seed": run["training_seed"],
                "planner_seed": config.get("seed"),
                "training_protocol": run["training_protocol"],
                "model_weight_source": run["model_weight_source"],
                "evaluation_kind": result_kind(relative, config_name),
                "metric_name": "success_rate_pct",
                "metric_value": reported,
                "success_rate_pct": reported,
                "frechet_fidelity_pct": None,
                "successful_tasks": sum(value is True for value in successes),
                "completed_tasks": completed,
                "expected_tasks": expected,
                "status": "complete" if complete else "partial_or_invalid",
                "metric_consistent_with_episode_list": consistent,
                "result_path": str(metrics_path),
                "result_sha256": sha256(metrics_path),
                "eval_config_path": str(eval_config_path) if eval_config_path.is_file() else None,
                "eval_config_sha256": sha256(eval_config_path),
                "policy_checkpoint": str(policy_path) if policy_path else None,
                "task": file_identity(task_path, task_hashes),
                "result_mtime": metrics_path.stat().st_mtime,
            }
        )
    return rows


def collect_droid_planning(
    run: dict, run_dir: Path, task_hashes: dict[Path, str]
) -> list[dict]:
    rows = []
    for result_path in sorted(run_dir.rglob("eval.csv")):
        relative = result_path.relative_to(run_dir)
        if not any(part.startswith("eval_") for part in relative.parts):
            continue
        with result_path.open(newline="") as stream:
            record = next(csv.DictReader(stream), {})
        config_path = result_path.parent / "plan_config.yaml"
        config = load_yaml(config_path)
        episodes = int(float(record["n_episodes"])) if record.get("n_episodes") else 0
        expected = config.get("num_eval")
        skill = finite_float(record.get("frechet/skill_mean"))
        policy_path = resolve_input_path(config.get("policy"), base=run_dir)
        clips_path = resolve_input_path(config.get("clips"), base=ROOT / "h_jepa")
        if expected is None and clips_path is not None and clips_path.is_file():
            try:
                clips = json.loads(clips_path.read_text())
                expected = len(clips) if isinstance(clips, list) else None
            except (OSError, json.JSONDecodeError):
                expected = None
        expected = int(expected) if expected is not None else None
        complete = expected is not None and episodes == expected and skill is not None
        rows.append(
            {
                "run_path": str(run_dir),
                "environment": run["environment"],
                "model": run["model"],
                "model_family": run["model_family"],
                "model_depth": run["model_depth"],
                "training_seed": run["training_seed"],
                "planner_seed": config.get("seed"),
                "training_protocol": run["training_protocol"],
                "model_weight_source": run["model_weight_source"],
                "evaluation_kind": result_kind(relative, None, droid=True),
                "metric_name": "frechet_fidelity_pct",
                "metric_value": 100.0 * skill if skill is not None else None,
                "success_rate_pct": None,
                "frechet_fidelity_pct": 100.0 * skill if skill is not None else None,
                "successful_tasks": None,
                "completed_tasks": episodes,
                "expected_tasks": expected,
                "status": "complete" if complete else "partial_or_invalid",
                "metric_consistent_with_episode_list": None,
                "result_path": str(result_path),
                "result_sha256": sha256(result_path),
                "eval_config_path": str(config_path) if config_path.is_file() else None,
                "eval_config_sha256": sha256(config_path),
                "policy_checkpoint": str(policy_path) if policy_path else None,
                "task": file_identity(clips_path, task_hashes),
                "result_mtime": result_path.stat().st_mtime,
            }
        )
    return rows


def collect_posthoc(run: dict, run_dir: Path) -> tuple[list[dict], list[dict]]:
    results, metric_rows = [], []
    for metrics_path in sorted(run_dir.glob("final_probing_decoding_eval*/metrics.yaml")):
        output_name = metrics_path.parent.name
        config_path = metrics_path.parent / "config.yaml"
        config = load_yaml(config_path)
        metrics = load_yaml(metrics_path)
        numeric = {key: finite_float(value) for key, value in metrics.items()}
        numeric = {key: value for key, value in numeric.items() if value is not None}
        eval_metrics = {key: value for key, value in numeric.items() if "/eval/" in key}
        complete = bool(eval_metrics)
        result_id = f"{run_dir}:{output_name}"
        results.append(
            {
                "result_id": result_id,
                "run_path": str(run_dir),
                "environment": run["environment"],
                "model": run["model"],
                "training_seed": run["training_seed"],
                "training_protocol": run["training_protocol"],
                "model_weight_source": run["model_weight_source"],
                "output_name": output_name,
                "status": "complete" if complete else "partial_or_invalid",
                "metric_count": len(numeric),
                "eval_metric_count": len(eval_metrics),
                "probe_seed": config.get("seed"),
                "probe_lr": nested(config, "optimizer", "probe_lr", default=nested(config, "optimizer", "lr")),
                "decoder_lr": nested(config, "optimizer", "decoder_lr", default=nested(config, "optimizer", "lr")),
                "result_path": str(metrics_path),
                "result_sha256": sha256(metrics_path),
                "config_path": str(config_path) if config_path.is_file() else None,
                "config_sha256": sha256(config_path),
                "result_mtime": metrics_path.stat().st_mtime,
            }
        )
        for key, value in numeric.items():
            split = "eval" if "/eval/" in key else "train" if "/train/" in key else "other"
            level_match = re.search(r"level(\d+)", key)
            metric_rows.append(
                {
                    "result_id": result_id,
                    "run_path": str(run_dir),
                    "environment": run["environment"],
                    "model": run["model"],
                    "training_seed": run["training_seed"],
                    "training_protocol": run["training_protocol"],
                    "model_weight_source": run["model_weight_source"],
                    "output_name": output_name,
                    "split": split,
                    "level": int(level_match.group(1)) if level_match else None,
                    "metric_name": key,
                    "metric_value": value,
                }
            )
    return results, metric_rows


def mark_latest_planning(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["run_path"],
            row["evaluation_kind"],
            row["planner_seed"],
            row["policy_checkpoint"],
        )
        groups[key].append(row)
    for values in groups.values():
        latest = max(values, key=lambda row: row["result_mtime"])
        for row in values:
            row["is_latest_for_evaluation_key"] = row is latest


def planning_summaries(rows: list[dict]) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["status"] == "complete" and row["is_latest_for_evaluation_key"]
    ]
    groups = defaultdict(list)
    for row in selected:
        key = (
            row["environment"],
            row["model"],
            row["evaluation_kind"],
            row["metric_name"],
            row["training_protocol"],
            row["model_weight_source"],
        )
        groups[key].append(row)

    summaries = []
    for key, values in sorted(groups.items()):
        per_training_seed = defaultdict(list)
        for row in values:
            per_training_seed[row["training_seed"]].append(row["metric_value"])
        seed_means = {
            seed: mean(metric_values)
            for seed, metric_values in per_training_seed.items()
        }
        across_seed = list(seed_means.values())
        summaries.append(
            {
                "environment": key[0],
                "model": key[1],
                "evaluation_kind": key[2],
                "metric_name": key[3],
                "training_protocol": key[4],
                "model_weight_source": key[5],
                "complete_evaluation_count": len(values),
                "training_seed_count": len(seed_means),
                "training_seeds": sorted(seed_means, key=lambda value: (value is None, value)),
                "planner_seeds": sorted(
                    {row["planner_seed"] for row in values},
                    key=lambda value: (value is None, value),
                ),
                "mean": mean(across_seed),
                "standard_error": (
                    stdev(across_seed) / math.sqrt(len(across_seed))
                    if len(across_seed) > 1
                    else None
                ),
                "statistics_unit": "training seed after averaging planner seeds",
            }
        )
    return summaries


def collect_run(
    run_dir: Path,
    *,
    fresh_runs: set[Path],
    task_hashes: dict[Path, str],
    provenance_overrides: dict[Path, dict],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    environment, model, run_name = run_dir.parts[-3:]
    config_path = run_dir / "config.yaml"
    config = load_yaml(config_path)
    family, depth = model_metadata(model, config)
    seed = config.get("seed")
    if seed is None:
        match = re.search(r"seed(\d+)", run_name)
        seed = int(match.group(1)) if match else None
    protocol, version, batch, accumulation = training_protocol(config)
    sessions = find_metric_sessions(run_dir)
    metrics, lineage = merge_metric_sessions(sessions)
    latest = metrics[-1] if metrics else {}
    final_name = config.get("output_model_name", model)
    final_checkpoint = run_dir / f"{final_name}_object.ckpt"
    epoch_checkpoints = sorted(run_dir.glob("*_epoch_*_object.ckpt"))
    recent_metric_age = (
        datetime.now().timestamp() - sessions[-1].stat().st_mtime if sessions else None
    )
    if final_checkpoint.is_file():
        training_status = "complete"
    elif recent_metric_age is not None and recent_metric_age < 600:
        training_status = "in_progress"
    elif sessions or epoch_checkpoints:
        training_status = "incomplete_stale"
    else:
        training_status = "no_local_training_evidence"

    provenance = provenance_overrides.get(run_dir.resolve())
    if provenance is not None:
        source = provenance["model_weight_source"]
    elif run_dir.resolve() in fresh_runs:
        source = "local_fresh_training"
    elif sessions or epoch_checkpoints:
        source = "local_training"
    elif final_checkpoint.is_file():
        source = "external_or_unknown"
    else:
        source = "unknown"

    resume_logs = sorted(run_dir.glob("resume*.log"))
    nonresumable_warning = any(
        "dataloader is not resumable" in path.read_text(errors="replace")
        for path in resume_logs
    )
    run = {
        "environment": environment,
        "model": model,
        "model_family": family,
        "model_depth": depth,
        "run_name": run_name,
        "run_path": str(run_dir),
        "training_seed": int(seed) if seed is not None else None,
        "model_weight_source": source,
        "model_weight_source_note": provenance.get("note") if provenance else None,
        "model_source_revision": provenance.get("source_revision") if provenance else None,
        "training_protocol": protocol,
        "training_protocol_version": version,
        "batch_size_per_device": batch,
        "accumulate_grad_batches": accumulation,
        "training_status": training_status,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint.is_file() else None,
        "final_checkpoint_size_bytes": final_checkpoint.stat().st_size if final_checkpoint.is_file() else None,
        "intermediate_checkpoint_count": len(epoch_checkpoints),
        "latest_logged_step": latest.get("step"),
        "latest_logged_epoch": latest.get("epoch"),
        "metrics_session_count": lineage["session_count"],
        "resumed_or_restarted": lineage["has_resume_or_restart"],
        "superseded_logged_step_count": lineage["superseded_logged_step_count"],
        "metrics_lineage": lineage,
        "resume_logs": [str(path) for path in resume_logs],
        "nonresumable_dataloader_warning": nonresumable_warning,
        "config_path": str(config_path) if config_path.is_file() else None,
        "config_sha256": sha256(config_path),
        "training_config": config,
        "planning_callback_enabled": bool(nested(config, "planning_eval", "enabled", default=False)),
        "posthoc_callback_enabled": bool(nested(config, "final_probing_decoding_eval", "enabled", default=False)),
    }
    planning = collect_sim_planning(run, run_dir, task_hashes)
    planning.extend(collect_droid_planning(run, run_dir, task_hashes))
    posthoc, probe_metrics = collect_posthoc(run, run_dir)
    return run, planning, posthoc, probe_metrics


def coverage_rows(
    runs: list[dict], planning: list[dict], posthoc: list[dict]
) -> list[dict]:
    planning_by_run = defaultdict(list)
    posthoc_by_run = defaultdict(list)
    for row in planning:
        planning_by_run[row["run_path"]].append(row)
    for row in posthoc:
        posthoc_by_run[row["run_path"]].append(row)

    output = []
    for run in runs:
        path = run["run_path"]
        plan_complete = sum(row["status"] == "complete" for row in planning_by_run[path])
        post_complete = sum(row["status"] == "complete" for row in posthoc_by_run[path])

        def callback_status(enabled: bool, complete_count: int) -> str:
            if not enabled:
                return "not_expected"
            if complete_count:
                return "complete"
            if run["training_status"] == "complete":
                return "missing"
            if run["training_status"] == "in_progress":
                return "pending_training"
            return "blocked_on_incomplete_training"

        output.append(
            {
                key: run[key]
                for key in (
                    "environment",
                    "model",
                    "run_name",
                    "run_path",
                    "training_seed",
                    "model_weight_source",
                    "training_protocol",
                    "training_status",
                    "final_checkpoint",
                    "metrics_session_count",
                    "resumed_or_restarted",
                    "superseded_logged_step_count",
                    "nonresumable_dataloader_warning",
                )
            }
            | {
                "complete_planning_result_count": plan_complete,
                "planning_status": callback_status(
                    run["planning_callback_enabled"], plan_complete
                ),
                "complete_posthoc_result_count": post_complete,
                "posthoc_status": callback_status(
                    run["posthoc_callback_enabled"], post_complete
                ),
            }
        )
    return output


def write_report(
    path: Path,
    runs: list[dict],
    planning: list[dict],
    summaries: list[dict],
    coverage: list[dict],
) -> None:
    complete_planning = [row for row in planning if row["status"] == "complete"]
    lines = [
        "# H-JEPA unified result aggregation",
        "",
        f"Collected at {now_local()} from `{runs[0]['run_path'].split('/ckpts/')[0] + '/ckpts' if runs else 'no checkpoint root'}`.",
        "",
        "Missing results remain missing; they are never converted to zero. Local training, fresh protocol-2 training, and external/unknown weights are separate provenance classes. Planning summaries first average planner seeds within each training seed, then compute mean and SE across training seeds.",
        "",
        "## Coverage",
        "",
        "| Env | Model | Run | Source | Protocol | Training | Planning | Posthoc | Sessions | Resume warning |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['environment']} | {row['model']} | {row['run_name']} | "
            f"{row['model_weight_source']} | {row['training_protocol']} | "
            f"{row['training_status']} | {row['planning_status']} | "
            f"{row['posthoc_status']} | {row['metrics_session_count']} | "
            f"{row['nonresumable_dataloader_warning']} |"
        )
    lines += [
        "",
        "## Complete Planning Results",
        "",
        "| Env | Model | Train seed | Planner seed | Evaluation | Metric | Value | Tasks | Protocol | Source |",
        "|---|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in complete_planning:
        lines.append(
            f"| {row['environment']} | {row['model']} | {row['training_seed']} | "
            f"{row['planner_seed']} | {row['evaluation_kind']} | {row['metric_name']} | "
            f"{row['metric_value']:.6g} | {row['completed_tasks']} | "
            f"{row['training_protocol']} | {row['model_weight_source']} |"
        )
    lines += [
        "",
        "## Summary Groups",
        "",
        "| Env | Model | Evaluation | Metric | Seeds | Mean | SE | Protocol | Source |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        se = "NA" if row["standard_error"] is None else f"{row['standard_error']:.6g}"
        lines.append(
            f"| {row['environment']} | {row['model']} | {row['evaluation_kind']} | "
            f"{row['metric_name']} | {row['training_seed_count']} | {row['mean']:.6g} | "
            f"{se} | {row['training_protocol']} | {row['model_weight_source']} |"
        )
    lines += [
        "",
        "Machine-readable files: `aggregated_results.json`, `aggregated_runs.csv`, `aggregated_coverage.csv`, `aggregated_planning_results.csv`, `aggregated_probe_results.csv`, and `aggregated_planning_summary.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def aggregate(
    ckpt_root: Path, output_dir: Path, provenance_manifest: Path | None = None
) -> dict:
    task_hashes = known_task_hashes()
    fresh_runs = known_fresh_runs()
    provenance_overrides = load_provenance_overrides(provenance_manifest)
    runs, planning, posthoc, probe_metrics = [], [], [], []
    for run_dir in discover_run_dirs(ckpt_root):
        run, run_planning, run_posthoc, run_probe_metrics = collect_run(
            run_dir,
            fresh_runs=fresh_runs,
            task_hashes=task_hashes,
            provenance_overrides=provenance_overrides,
        )
        runs.append(run)
        planning.extend(run_planning)
        posthoc.extend(run_posthoc)
        probe_metrics.extend(run_probe_metrics)
    mark_latest_planning(planning)
    summaries = planning_summaries(planning)
    coverage = coverage_rows(runs, planning, posthoc)

    public_runs = [{key: value for key, value in run.items() if key != "training_config"} for run in runs]
    result = {
        "schema_version": 1,
        "created_at": now_local(),
        "checkpoint_root": str(ckpt_root),
        "aggregation_rules": {
            "missing_values": "preserved as null; never converted to zero",
            "planning_completion": "completed task count must equal configured task count; simulation success_rate must match episode_successes",
            "planning_statistics": "average planner seeds within training seed, then mean and sample-SE across training seeds",
            "metrics_sessions": "later SPT session replaces prior suffix from its first logged global step",
            "weight_source": "local evidence and explicit fresh-run manifest only; unverified external files are not labeled official",
            "explicit_provenance_manifest": str(provenance_manifest) if provenance_manifest else None,
        },
        "runs": public_runs,
        "coverage": coverage,
        "planning_results": planning,
        "posthoc_results": posthoc,
        "probe_decoder_metrics": probe_metrics,
        "planning_summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregated_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(output_dir / "aggregated_runs.csv", public_runs)
    write_csv(output_dir / "aggregated_coverage.csv", coverage)
    write_csv(output_dir / "aggregated_planning_results.csv", planning)
    write_csv(output_dir / "aggregated_probe_results.csv", probe_metrics)
    write_csv(output_dir / "aggregated_planning_summary.csv", summaries)
    write_report(
        output_dir / "AGGREGATED_RESULTS.md", runs, planning, summaries, coverage
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        help="Optional JSON assigning official/external source labels to exact run paths.",
    )
    args = parser.parse_args()
    result = aggregate(args.ckpt_root, args.output_dir, args.provenance_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "runs": len(result["runs"]),
                "complete_planning_results": sum(
                    row["status"] == "complete" for row in result["planning_results"]
                ),
                "complete_posthoc_results": sum(
                    row["status"] == "complete" for row in result["posthoc_results"]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
