#!/usr/bin/env python3
"""Stop queue dispatch after every Probe job has been launched and completed.

The runner process is stopped without signaling its active children. This lets the
last Probe jobs finish while preventing the combined queue from launching Depth or
Cost jobs. Once stopped, this watcher owns the runner status file until all probes
are complete.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from analysis_utils import default_data_root
from run_official_reproduction import complete, cost_jobs, depth_jobs, droid_jobs, probe_jobs


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    result_root = data_root / "analyse" / "official_release_08fc8a21"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, default=data_root / "ckpts")
    parser.add_argument(
        "--runner-status",
        type=Path,
        default=result_root / "runner_status_droid_probe_depth_cost.json",
    )
    parser.add_argument(
        "--pause-status",
        type=Path,
        default=result_root / "probe_pause_status.json",
    )
    parser.add_argument("--runner-target", default="hjepa_repro:0.0")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def runner_pid(target: str) -> int:
    output = subprocess.run(
        ["tmux", "display-message", "-p", "-t", target, "#{pane_pid}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(output)


def process_state(pid: int) -> str:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("State:"):
            return line.split()[1]
    raise RuntimeError(f"process state is unavailable for PID {pid}")


def completed_names(jobs: list) -> set[str]:
    return {job.name for job in jobs if complete(job)}


def write_pause_status(path: Path, *, state: str, pid: int, done: int, total: int, detail: str) -> None:
    write_json(
        path,
        {
            "updated_at": timestamp(),
            "state": state,
            "detail": detail,
            "runner_pid": pid,
            "probe_complete": done,
            "probe_total": total,
        },
    )


def main() -> None:
    args = parse_args()
    probes = probe_jobs(args.ckpt_root)
    expected = {job.name for job in probes}
    all_jobs = (
        droid_jobs(args.ckpt_root)
        + probes
        + depth_jobs(args.ckpt_root)
        + cost_jobs(args.ckpt_root)
    )
    total_jobs = len(all_jobs)
    pid = runner_pid(args.runner_target)
    done = completed_names(probes)
    runner = read_json(args.runner_status)
    active = set(runner.get("active", []))
    print(
        f"runner_pid={pid} probe_complete={len(done)}/{len(probes)} "
        f"active_probe={len(active & expected)} total_jobs={total_jobs}",
        flush=True,
    )
    if args.dry_run:
        return

    write_pause_status(
        args.pause_status,
        state="waiting_for_all_probes_to_start",
        pid=pid,
        done=len(done),
        total=len(probes),
        detail="The combined runner is active; no Depth or Cost job has been allowed to start.",
    )

    while True:
        runner = read_json(args.runner_status)
        active = set(runner.get("active", []))
        done = completed_names(probes)
        active_probe = active & expected
        active_non_probe = active - expected
        if active_non_probe:
            os.kill(pid, signal.SIGSTOP)
            write_pause_status(
                args.pause_status,
                state="runner_paused_non_probe_detected",
                pid=pid,
                done=len(done),
                total=len(probes),
                detail=f"Runner was stopped after unexpected non-Probe dispatch: {sorted(active_non_probe)}",
            )
            raise RuntimeError(f"non-Probe jobs were dispatched: {sorted(active_non_probe)}")
        if expected <= done | active_probe:
            os.kill(pid, signal.SIGSTOP)
            for _ in range(50):
                if process_state(pid) in {"T", "t"}:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"runner PID {pid} did not enter a stopped state")
            print(f"runner PID {pid} stopped; all {len(probes)} probes are complete or active", flush=True)
            break
        write_pause_status(
            args.pause_status,
            state="waiting_for_all_probes_to_start",
            pid=pid,
            done=len(done),
            total=len(probes),
            detail=f"{len(active_probe)} Probe jobs active; waiting for the final Probe dispatch.",
        )
        time.sleep(args.poll_seconds)

    # Recompute non-Probe completions from artifacts instead of relying on a possibly
    # stale runner counter at the exact instant SIGSTOP was delivered.
    non_probe_jobs = droid_jobs(args.ckpt_root) + depth_jobs(args.ckpt_root) + cost_jobs(args.ckpt_root)
    non_probe_complete = sum(complete(job) for job in non_probe_jobs)
    failures = runner.get("failures", [])
    while True:
        done = completed_names(probes)
        incomplete = sorted(expected - done)
        aggregate_complete = non_probe_complete + len(done)
        aggregate = {
            "release_revision": runner.get("release_revision"),
            "total": total_jobs,
            "complete": aggregate_complete,
            "pending": total_jobs - aggregate_complete - len(incomplete),
            "active": incomplete,
            "failures": failures,
            "state": "paused_after_probe_dispatch",
        }
        write_json(args.runner_status, aggregate)
        if not incomplete:
            aggregate["active"] = []
            aggregate["pending"] = total_jobs - aggregate_complete
            aggregate["state"] = "paused_after_probes"
            write_json(args.runner_status, aggregate)
            write_pause_status(
                args.pause_status,
                state="paused_after_probes",
                pid=pid,
                done=len(done),
                total=len(probes),
                detail="All Probe jobs are complete. The runner remains SIGSTOP'd; no Depth or Cost job was launched.",
            )
            print(f"all {len(probes)} Probe jobs complete; runner PID {pid} remains stopped", flush=True)
            return
        write_pause_status(
            args.pause_status,
            state="finishing_active_probes",
            pid=pid,
            done=len(done),
            total=len(probes),
            detail=f"Runner is stopped; {len(incomplete)} already-launched Probe jobs are still running.",
        )
        time.sleep(max(args.poll_seconds, 2.0))


if __name__ == "__main__":
    main()
