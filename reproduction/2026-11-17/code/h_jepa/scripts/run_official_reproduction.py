#!/usr/bin/env python3
"""Run the complete checkpoint-supported paper protocol across available GPUs.

Jobs are independent and resumable. Simulation cells are complete only after 50 episode
outcomes are present; DROID cells are complete only after all 16 clips are aggregated;
probe cells require every paper entity/level metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from analysis_utils import default_data_root, load_json, load_yaml
from analyze_official_release import DEPTH_MODELS, DROID_MODELS, DROID_SEEDS, PLAN_SEEDS, SIM_ENVS, SIM_SEEDS


HJEPA_DIR = Path(__file__).resolve().parents[1]
PROBE_TARGETS = {
    "ant": ("xy", "proprio"),
    "fourroom": ("xy", "distractor0_xy"),
    "cube": ("block_pos", "block_ori", "effector_pos", "effector_yaw", "gripper", "arm_joint_pos", "arm_joint_vel"),
    "pusht": ("agent_xy", "block_xy", "block_ori", "agent_vel"),
}


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    result: Path
    kind: str
    priority: float = 1.0


def _sim_job(ckpt_root: Path, env: str, model: str, planner: str, seed: int, tag: str) -> Job:
    run = ckpt_root / env / f"{env}_{model}" / f"seed{seed}"
    output = run / f"eval_{planner}"
    return Job(
        name=f"{tag}.{env}.{model}.seed{seed}.{planner}",
        command=(
            sys.executable,
            "eval.py",
            "--config-name",
            f"{env}_{planner}",
            f"seed={seed}",
            f"policy={run / f'{env}_{model}_object.ckpt'}",
            f"output.dir={output}",
            f"hydra.run.dir={output / 'hydra'}",
        ),
        result=output / "metrics.yaml",
        kind="simulation",
        priority=float(int(model.rsplit("l", 1)[-1]) if model != "lewm" else 1),
    )


def depth_jobs(ckpt_root: Path) -> list[Job]:
    return [
        _sim_job(ckpt_root, env, model, planner, seed, "depth")
        for env in SIM_ENVS
        for model, planner, _depth in DEPTH_MODELS
        for seed in SIM_SEEDS
    ]


def cost_jobs(ckpt_root: Path) -> list[Job]:
    jobs = []
    for env in SIM_ENVS:
        for depth in (2, 3, 4):
            for cost_level in range(1, depth + 1):
                planner = "flat" if cost_level == 1 else f"l{cost_level}_project"
                for seed in SIM_SEEDS:
                    jobs.append(_sim_job(ckpt_root, env, f"hjepa_l{depth}", planner, seed, "cost"))
    return jobs


def droid_jobs(ckpt_root: Path) -> list[Job]:
    jobs = []
    for model, planner in DROID_MODELS:
        for seed in DROID_SEEDS:
            run = ckpt_root / "droid" / f"droid_{model}" / f"seed{seed}"
            for plan_seed in PLAN_SEEDS:
                output = run / f"eval_{planner}" / f"plan_seed{plan_seed}"
                jobs.append(
                    Job(
                        name=f"droid.{model}.seed{seed}.plan_seed{plan_seed}",
                        command=(
                            sys.executable,
                            "eval.py",
                            "--config-name",
                            f"droid_{planner}",
                            f"policy={run / f'droid_{model}_object.ckpt'}",
                            f"seed={plan_seed}",
                            f"output.dir={output}",
                            f"hydra.run.dir={output / 'hydra'}",
                        ),
                        result=output / "eval.csv",
                        kind="droid",
                        priority=2.0 if planner == "l2" else 1.0,
                    )
                )
    return jobs


def probe_jobs(ckpt_root: Path) -> list[Job]:
    jobs = []
    # Figure 5/12(b): paired L1/L2 probes from two-level H-JEPA in every released
    # simulation environment. Figure 4: four-level Ant/FourRoom probes plus the
    # post-hoc L1-L3 decoders shown in its right panel.
    cells = [(env, 2, False) for env in SIM_ENVS]
    cells += [(env, 4, True) for env in ("ant", "fourroom")]
    for env, depth, decoders in cells:
        model = f"hjepa_l{depth}"
        for seed in SIM_SEEDS:
            train_decoders = decoders and seed == 42
            run = ckpt_root / env / f"{env}_{model}" / f"seed{seed}"
            output = run / "final_probing_decoding_eval_v2"
            command = [
                sys.executable,
                "main_probing_decoding_eval.py",
                "--config-name",
                env,
                f"policy={run / f'{env}_{model}_object.ckpt'}",
                f"seed={seed}",
                f"hydra.run.dir={output / 'hydra'}",
            ]
            if train_decoders:
                command.extend(f"+level{level}.train_decoder=true" for level in (1, 2, 3))
            jobs.append(
                Job(
                    name=f"probe.{env}.{model}.seed{seed}",
                    command=tuple(command),
                    result=output / "metrics.yaml",
                    kind="probe_decoder" if train_decoders else "probe",
                    priority=float((60 if env == "fourroom" else 25) * (1.25 if train_decoders else 1.0)),
                )
            )
    return jobs


def complete(job: Job) -> bool:
    if not job.result.exists():
        return False
    try:
        if job.kind == "simulation":
            return len(load_yaml(job.result).get("episode_successes", [])) == 50
        if job.kind.startswith("probe"):
            metrics = load_yaml(job.result)
            parts = job.name.split(".")
            env, model = parts[1], parts[2]
            depth = int(model.removeprefix("hjepa_l"))
            required = {
                f"final_probing_decoding_eval/eval/level{level}_probe_{target}_nmse"
                for level in range(1, depth + 1)
                for target in PROBE_TARGETS[env]
            }
            if not required <= set(metrics):
                return False
            if job.kind == "probe_decoder":
                return all((job.result.parent / f"decoder_level{level}.ckpt").exists() for level in (1, 2, 3))
            return True
        with job.result.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == 1 and int(rows[0]["n_episodes"]) == 16
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_release(ckpt_root: Path, jobs: list[Job], include_droid: bool) -> str:
    manifest = load_json(ckpt_root / "OFFICIAL_WEIGHTS_MANIFEST.json")
    files = manifest["files"]
    missing = []
    for job in jobs:
        policy_arg = next(arg for arg in job.command if arg.startswith("policy="))[7:]
        policy = Path(policy_arg)
        normalizer = policy.parent / "normalizer.pt"
        for path in (policy, normalizer):
            relative = str(path.relative_to(ckpt_root))
            if relative not in files or not path.exists():
                missing.append(relative)
    if missing:
        preview = "\n".join(sorted(set(missing))[:20])
        raise FileNotFoundError(f"release is incomplete ({len(set(missing))} missing files):\n{preview}")
    if include_droid:
        clips = default_data_root() / "droid/droid_raw/1.0.1"
        if not clips.exists() or not any(clips.iterdir()):
            raise FileNotFoundError(f"DROID evaluation clips are missing from {clips}")
    return str(manifest["revision"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, default=default_data_root() / "ckpts")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("depth", "cost", "droid", "probe"),
        default=("depth", "droid", "probe", "cost"),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,6", help="Comma-separated physical GPU IDs; GPU 5 is excluded by default.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factories = {"depth": depth_jobs, "cost": cost_jobs, "droid": droid_jobs, "probe": probe_jobs}
    jobs = [
        job
        for experiment in args.experiments
        for job in sorted(factories[experiment](args.ckpt_root), key=lambda item: item.priority, reverse=True)
    ]
    # Native H-JEPA hierarchical results and cost-ladder cells are distinct, but reject
    # accidental duplicate command/output definitions before launching a multi-day run.
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate job names generated")

    if args.dry_run:
        for job in jobs:
            print(job.name, "::", " ".join(job.command))
        print(f"{len(jobs)} jobs")
        return

    revision = validate_release(args.ckpt_root, jobs, "droid" in args.experiments)
    pending = [job for job in jobs if not complete(job)]
    skipped = len(jobs) - len(pending)
    result_root = default_data_root() / "analyse" / f"official_release_{revision[:8]}"
    log_dir = result_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = result_root / f"runner_status_{'_'.join(args.experiments)}.json"
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    print(f"release={revision} jobs={len(jobs)} skipped={skipped} pending={len(pending)} gpus={gpus}", flush=True)

    active: dict[subprocess.Popen, tuple[Job, str, object, float]] = {}
    failures: list[dict] = []
    completed = skipped
    while pending or active:
        used = {value[1] for value in active.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            log_path = log_dir / f"{job.name}.log"
            log = log_path.open("a")
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "MUJOCO_GL": "egl",
                    "MUJOCO_EGL_DEVICE_ID": "0",
                    "HJEPA_HOME": str(default_data_root()),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} ===\n")
            log.flush()
            process = subprocess.Popen(job.command, cwd=HJEPA_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[process] = (job, gpu, log, time.time())
            print(f"START gpu={gpu} {job.name}", flush=True)

        time.sleep(2)
        for process, (job, gpu, log, started) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            elapsed = time.time() - started
            del active[process]
            if returncode == 0 and complete(job):
                completed += 1
                print(f"DONE  gpu={gpu} {job.name} seconds={elapsed:.1f} progress={completed}/{len(jobs)}", flush=True)
            else:
                failures.append({"job": job.name, "returncode": returncode, "log": str(log_dir / f'{job.name}.log')})
                print(f"FAIL  gpu={gpu} {job.name} returncode={returncode}", flush=True)

        status = {
            "release_revision": revision,
            "total": len(jobs),
            "complete": completed,
            "pending": len(pending),
            "active": [value[0].name for value in active.values()],
            "failures": failures,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if failures:
            for process, (_job, _gpu, _log, _started) in active.items():
                process.terminate()
            for process, (_job, _gpu, log, _started) in active.items():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            raise SystemExit(f"stopped after {len(failures)} failed job(s); see {log_dir}")

    print(f"all {len(jobs)} jobs complete", flush=True)


if __name__ == "__main__":
    main()
