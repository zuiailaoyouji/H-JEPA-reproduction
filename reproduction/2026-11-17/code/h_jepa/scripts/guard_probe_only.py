#!/usr/bin/env python3
"""Allow Probe children to run while freezing every later queue group.

The combined runner keeps stopped non-Probe children in its active slots, so it
cannot advance past the Probe boundary. This avoids tmux job-control automatically
resuming a stopped foreground runner.
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
from run_official_reproduction import complete, probe_jobs


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    result_root = data_root / "analyse" / "official_release_08fc8a21"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, default=data_root / "ckpts")
    parser.add_argument("--status", type=Path, default=result_root / "probe_pause_status.json")
    parser.add_argument("--runner-target", default="hjepa_repro:0.0")
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def runner_pid(target: str) -> int:
    output = subprocess.run(
        ["tmux", "display-message", "-p", "-t", target, "#{pane_pid}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(output)


def process_table(pgid: int) -> dict[int, dict]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = {}
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=4)
        if len(parts) != 5 or int(parts[2]) != pgid:
            continue
        pid, ppid, row_pgid, state, command = parts
        rows[int(pid)] = {
            "ppid": int(ppid),
            "pgid": int(row_pgid),
            "state": state,
            "command": command,
        }
    return rows


def descendants(root: int, rows: dict[int, dict]) -> set[int]:
    result = {root}
    while True:
        added = {pid for pid, row in rows.items() if row["ppid"] in result}
        updated = result | added
        if updated == result:
            return result
        result = updated


def freeze_non_probe_children(pid: int, pgid: int) -> list[dict]:
    rows = process_table(pgid)
    frozen = []
    for child_pid, row in rows.items():
        if row["ppid"] != pid:
            continue
        command = row["command"]
        if "main_probing_decoding_eval.py" in command:
            continue
        if " eval.py " not in f" {command} ":
            continue
        tree = descendants(child_pid, rows)
        # Stop descendants first so no worker can continue after its parent stops.
        for process_pid in sorted(tree, reverse=True):
            try:
                os.kill(process_pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        frozen.append({"pid": child_pid, "command": command})
    return frozen


def main() -> None:
    args = parse_args()
    probes = probe_jobs(args.ckpt_root)
    pid = runner_pid(args.runner_target)
    pgid = os.getpgid(pid)
    done = sum(complete(job) for job in probes)
    frozen = freeze_non_probe_children(pid, pgid)
    print(
        f"runner_pid={pid} runner_pgid={pgid} probe_complete={done}/{len(probes)} "
        f"frozen_non_probe={len(frozen)}",
        flush=True,
    )
    if args.dry_run:
        return

    last_status_write = 0.0
    while True:
        frozen = freeze_non_probe_children(pid, pgid)
        now = time.monotonic()
        if now - last_status_write >= 2.0:
            done = sum(complete(job) for job in probes)
            state = "paused_after_probes" if done == len(probes) else "protecting_probe_only"
            detail = (
                "All Probe jobs are complete; non-Probe runner children remain frozen."
                if done == len(probes)
                else "Probe jobs may continue; every Depth or Cost child is immediately frozen."
            )
            write_json(
                args.status,
                {
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "state": state,
                    "detail": detail,
                    "runner_pid": pid,
                    "runner_pgid": pgid,
                    "probe_complete": done,
                    "probe_total": len(probes),
                    "frozen_non_probe_count": len(frozen),
                    "frozen_non_probe": frozen,
                },
            )
            last_status_write = now
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
