#!/usr/bin/env python3
"""Read and merge SPT metric sessions without hiding resumed-run rollbacks."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path


def _session_sort_key(path: Path) -> tuple[str, str, str]:
    """Sort the standard spt/runs/YYYYMMDD/HHMMSS/RUN_ID layout."""
    parts = path.parts
    try:
        runs_index = parts.index("runs")
        date, time, run_id = parts[runs_index + 1 : runs_index + 4]
        return date, time, run_id
    except (ValueError, IndexError):
        return "", "", str(path)


def find_metric_sessions(run_dir: Path) -> list[Path]:
    candidates = run_dir.glob("spt/runs/*/*/*/metrics.csv")
    return sorted(candidates, key=_session_sort_key)


def _read_metrics_with_diagnostics(path: Path) -> tuple[list[dict], int]:
    """Merge sparse CSV rows belonging to the same logged optimizer step."""
    by_step: dict[int, dict] = {}
    raw = path.read_bytes()
    ignored = 0
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        ignored = len(raw) - end
        raw = raw[:end]
    with io.StringIO(raw.decode(errors="replace"), newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                step = int(float(row["step"]))
            except (KeyError, TypeError, ValueError):
                continue
            item = by_step.setdefault(step, {"step": step})
            for key, raw in row.items():
                if key == "step" or raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    item[key] = raw
                    continue
                item[key] = value if math.isfinite(value) else None
    return [by_step[key] for key in sorted(by_step)], ignored


def read_metrics(path: Path) -> list[dict]:
    return _read_metrics_with_diagnostics(path)[0]


def merge_metric_sessions(
    paths: list[Path], *, source_paths: list[Path] | None = None
) -> tuple[list[dict], dict]:
    """Merge chronologically ordered sessions with rollback-aware precedence.

    A resumed SPT logger starts a new metrics.csv. Its first logged global step is
    the earliest directly observable replacement boundary. Every older point at
    or after that step belongs to the abandoned branch and is removed, even when
    logging cadences differ. The later session then supplies the effective suffix.
    """
    if source_paths is not None and len(source_paths) != len(paths):
        raise ValueError("source_paths must have the same length as paths")

    ordered_indices = sorted(range(len(paths)), key=lambda index: _session_sort_key(paths[index]))
    effective: dict[int, dict] = {}
    sessions = []
    superseded_steps = 0

    for session_index, input_index in enumerate(ordered_indices):
        path = paths[input_index]
        source = source_paths[input_index] if source_paths is not None else path
        rows, ignored = _read_metrics_with_diagnostics(path)
        first_step = rows[0]["step"] if rows else None
        last_step = rows[-1]["step"] if rows else None
        invalidated = (
            sorted(step for step in effective if step >= first_step)
            if first_step is not None
            else []
        )
        for step in invalidated:
            del effective[step]
        superseded_steps += len(invalidated)

        session_id = source.parent.name
        for row in rows:
            effective[row["step"]] = {
                **row,
                "_metrics_session_index": session_index,
                "_metrics_session_id": session_id,
                "_metrics_source": str(source),
            }

        sessions.append(
            {
                "session_index": session_index,
                "session_id": session_id,
                "source": str(source),
                "parsed_from": str(path),
                "first_logged_step": first_step,
                "last_logged_step": last_step,
                "logged_step_count": len(rows),
                "fit_loss_step_count": sum(
                    isinstance(row.get("fit/loss"), (int, float)) for row in rows
                ),
                "incomplete_trailing_bytes_ignored": ignored,
                "superseded_prior_step_count": len(invalidated),
                "superseded_prior_first_step": invalidated[0] if invalidated else None,
                "superseded_prior_last_step": invalidated[-1] if invalidated else None,
                "replacement_boundary": (
                    "first logged step; the actual checkpoint boundary may be earlier"
                    if session_index > 0 and first_step is not None
                    else None
                ),
            }
        )

    rows = [effective[step] for step in sorted(effective)]
    lineage = {
        "merge_policy": (
            "sessions ordered by SPT start time; each later session replaces the entire "
            "prior suffix beginning at its first logged global step"
        ),
        "session_count": len(sessions),
        "has_resume_or_restart": len(sessions) > 1,
        "sessions": sessions,
        "superseded_logged_step_count": superseded_steps,
        "effective_logged_step_count": len(rows),
        "effective_first_step": rows[0]["step"] if rows else None,
        "effective_last_step": rows[-1]["step"] if rows else None,
    }
    return rows, lineage
