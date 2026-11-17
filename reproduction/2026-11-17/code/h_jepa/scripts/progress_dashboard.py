#!/usr/bin/env python3
"""Serve a live dashboard for the official-checkpoint reproduction queue."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from analysis_utils import default_data_root
from run_official_reproduction import cost_jobs, depth_jobs, droid_jobs, probe_jobs, complete


GROUPS = {
    "droid": ("DROID", droid_jobs),
    "probe": ("Probes", probe_jobs),
    "depth": ("Depth", depth_jobs),
    "cost": ("Cost ladder", cost_jobs),
}

# Queue-time priors are replaced by measured medians as soon as a group finishes
# its first cell. They keep the dashboard useful while the first depth/cost cells
# are still waiting behind the probe queue.
DEFAULT_JOB_HOURS = {"droid": 0.25, "probe": 2.5, "depth": 0.7, "cost": 0.6}
LAUNCH_RE = re.compile(r"=== launch ([^=]+?) gpu=(\d+) ===")


def _tail(path: Path, size: int = 1 << 20) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(handle.tell() - size, 0))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def _latest_attempt(text: str) -> tuple[datetime | None, str | None, str]:
    """Return launch metadata and only the log text from the latest attempt."""
    matches = list(LAUNCH_RE.finditer(text))
    if not matches:
        return None, None, text
    match = matches[-1]
    try:
        launched = datetime.strptime(match.group(1).strip(), "%Y-%m-%d %H:%M:%S").astimezone()
    except ValueError:
        launched = None
    return launched, match.group(2), text[match.end() :]


def _gpu_stats() -> dict[str, dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    stats = {}
    for line in output.splitlines():
        index, memory, utilization = (part.strip() for part in line.split(","))
        stats[index] = {"memory_mib": int(memory), "utilization": int(utilization)}
    return stats


def _active_detail(name: str, job, log_dir: Path) -> dict:
    log_path = log_dir / f"{name}.log"
    text = _tail(log_path)
    launched, gpu, attempt = _latest_attempt(text)
    detail = "starting"
    current = 0
    total = None
    if job.kind == "droid":
        current = len(list(job.result.parent.glob("ep_*/actions.pt")))
        total = 16
        detail = f"clip {current}/{total}"
    elif job.kind.startswith("probe"):
        epochs = re.findall(r"epoch (\d+)/(\d+)", attempt)
        if epochs:
            current, total = map(int, epochs[-1])
            detail = f"epoch {current}/{total}"
        else:
            detail = "loading data/model"
    else:
        episodes = re.findall(r"(?:episode|eval)[ _-]?(\d+)", attempt, flags=re.IGNORECASE)
        if episodes:
            current = max(map(int, episodes))
            total = 50
            detail = f"episode {current}/{total}"
        else:
            detail = "planning/evaluation"
    return {
        "name": name,
        "group": name.split(".", 1)[0],
        "gpu": gpu,
        "detail": detail,
        "current": current,
        "total": total,
        "elapsed_seconds": max(0, int((datetime.now().astimezone() - launched).total_seconds())) if launched else None,
        "log": str(log_path),
    }


def _completed_duration(job, log_dir: Path) -> float | None:
    launched, _gpu, _attempt = _latest_attempt(_tail(log_dir / f"{job.name}.log"))
    if launched is None:
        return None
    try:
        seconds = job.result.stat().st_mtime - launched.timestamp()
    except OSError:
        return None
    return seconds if seconds > 0 else None


class DashboardState:
    def __init__(self, ckpt_root: Path, status_path: Path, gpus: list[str]):
        self.ckpt_root = ckpt_root
        self.status_path = status_path
        self.log_dir = status_path.parent / "logs"
        self.gpus = gpus
        self.jobs_by_group = {
            key: factory(ckpt_root) for key, (_label, factory) in GROUPS.items()
        }
        self.jobs = {job.name: job for jobs in self.jobs_by_group.values() for job in jobs}

    def snapshot(self) -> dict:
        try:
            runner = json.loads(self.status_path.read_text())
        except (OSError, json.JSONDecodeError):
            runner = {"active": [], "failures": []}
        try:
            finalizer = json.loads((self.status_path.parent / "finalizer_status.json").read_text())
        except (OSError, json.JSONDecodeError):
            finalizer = {"state": "not started"}
        active_names = list(runner.get("active", []))
        completed_names = {name for name, job in self.jobs.items() if complete(job)}
        groups = []
        for key, (label, _factory) in GROUPS.items():
            jobs = self.jobs_by_group[key]
            done = sum(job.name in completed_names for job in jobs)
            active = sum(job.name in active_names for job in jobs)
            groups.append({"key": key, "label": label, "done": done, "active": active, "total": len(jobs)})

        active = [
            _active_detail(name, self.jobs[name], self.log_dir)
            for name in active_names
            if name in self.jobs
        ]
        active_by_name = {item["name"]: item for item in active}
        active_by_gpu = {item["gpu"]: item["name"] for item in active if item["gpu"] is not None}
        gpu_stats = _gpu_stats()
        gpus = [
            {
                "index": gpu,
                "task": active_by_gpu.get(gpu),
                **gpu_stats.get(gpu, {"memory_mib": None, "utilization": None}),
            }
            for gpu in self.gpus
        ]
        completed = sorted(
            (job for job in self.jobs.values() if job.name in completed_names),
            key=lambda job: job.result.stat().st_mtime,
            reverse=True,
        )[:10]
        durations: dict[str, list[float]] = {key: [] for key in GROUPS}
        for name in completed_names:
            job = self.jobs[name]
            seconds = _completed_duration(job, self.log_dir)
            if seconds is not None:
                durations[name.split(".", 1)[0]].append(seconds)
        expected = {
            key: statistics.median(values) if values else DEFAULT_JOB_HOURS[key] * 3600
            for key, values in durations.items()
        }
        remaining_work = 0.0
        for name, job in self.jobs.items():
            if name in completed_names:
                continue
            seconds = expected[name.split(".", 1)[0]]
            item = active_by_name.get(name)
            if item and item["total"]:
                seconds *= max(0.0, 1.0 - item["current"] / item["total"])
            remaining_work += seconds
        eta_hours = remaining_work / max(len(self.gpus), 1) / 3600
        total = len(self.jobs)
        return {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "revision": runner.get("release_revision"),
            "done": len(completed_names),
            "active_count": len(active),
            "pending": total - len(completed_names) - len(active),
            "total": total,
            "eta_low_hours": 0.85 * eta_hours,
            "eta_high_hours": 1.3 * eta_hours,
            "eta_basis": {
                key: "measured" if durations[key] else "prior"
                for key in GROUPS
            },
            "groups": groups,
            "active": active,
            "gpus": gpus,
            "recent": [job.name for job in completed],
            "failures": runner.get("failures", []),
            "finalizer": finalizer,
            "status_path": str(self.status_path),
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>H-JEPA reproduction progress</title>
  <style>
    :root { color-scheme: light; --ink:#18201c; --muted:#66716a; --line:#d8ded9; --paper:#f7f8f5; --white:#fff; --green:#17784b; --amber:#b56b09; --red:#b42318; }
    * { box-sizing:border-box; letter-spacing:0; }
    body { margin:0; font:14px/1.45 ui-sans-serif,system-ui,sans-serif; color:var(--ink); background:var(--paper); }
    header { background:#202824; color:#fff; padding:20px 28px; border-bottom:4px solid #48a977; }
    header h1 { margin:0; font-size:22px; font-weight:650; }
    header p { margin:4px 0 0; color:#cad2cd; }
    main { max-width:1280px; margin:0 auto; padding:24px 28px 40px; }
    .summary { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }
    .metric { background:var(--white); padding:16px; min-height:86px; }
    .metric .value { display:block; font-size:28px; font-weight:700; font-variant-numeric:tabular-nums; }
    .metric .value.state { font-size:20px; overflow-wrap:anywhere; }
    .metric .label { color:var(--muted); }
    section { margin-top:24px; }
    h2 { margin:0 0 10px; font-size:16px; }
    .bar-row { display:grid; grid-template-columns:110px minmax(120px,1fr) 92px; align-items:center; gap:12px; margin:10px 0; }
    .track { height:12px; background:#e5e9e6; overflow:hidden; }
    .fill { height:100%; background:var(--green); transition:width .35s ease; }
    .count { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
    .gpu-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
    .gpu { border:1px solid var(--line); background:var(--white); padding:12px; min-height:98px; }
    .gpu-head { display:flex; justify-content:space-between; margin-bottom:7px; }
    .gpu strong { font-size:16px; }
    .util { color:var(--green); font-variant-numeric:tabular-nums; }
    .task { color:var(--muted); overflow-wrap:anywhere; }
    table { width:100%; border-collapse:collapse; background:var(--white); border:1px solid var(--line); }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:12px; font-weight:600; text-transform:uppercase; }
    td:first-child { width:64px; font-variant-numeric:tabular-nums; }
    td:nth-child(2) { overflow-wrap:anywhere; }
    .task-progress { width:100%; max-width:180px; height:5px; margin-top:5px; background:#e5e9e6; overflow:hidden; }
    .task-progress span { display:block; height:100%; background:var(--amber); }
    .empty { padding:18px; border:1px solid var(--line); background:var(--white); color:var(--muted); }
    .failure { color:var(--red); }
    footer { margin-top:24px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    @media (max-width:760px) { main { padding:18px 14px 30px; } header { padding:17px 14px; } .summary { grid-template-columns:repeat(2,1fr); } .gpu-grid { grid-template-columns:1fr; } .bar-row { grid-template-columns:82px 1fr 76px; } th:nth-child(3),td:nth-child(3) { display:none; } }
  </style>
</head>
<body>
  <header><h1>H-JEPA reproduction</h1><p id="stamp">Connecting...</p></header>
  <main>
    <div class="summary">
      <div class="metric"><span class="value" id="overall">0%</span><span class="label">Overall complete</span></div>
      <div class="metric"><span class="value" id="done">0 / 0</span><span class="label">Finished tasks</span></div>
      <div class="metric"><span class="value" id="active-count">0</span><span class="label">Active GPUs</span></div>
      <div class="metric"><span class="value" id="pending">0</span><span class="label">Queued tasks</span></div>
      <div class="metric"><span class="value" id="eta">--</span><span class="label">Queue ETA</span></div>
      <div class="metric"><span class="value state" id="finalizer">--</span><span class="label" id="finalizer-label">Postprocess</span></div>
    </div>
    <section><h2>Experiment groups</h2><div id="groups"></div></section>
    <section><h2>GPU pool</h2><div class="gpu-grid" id="gpus"></div></section>
    <section><h2>Active tasks</h2><div id="active"></div></section>
    <section><h2>Recent completions</h2><div id="recent"></div></section>
    <section><h2>Failures</h2><div id="failures"></div></section>
    <footer id="footer"></footer>
  </main>
  <script>
    const node = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
    const empty = text => { const n=node('div','empty',text); return n; };
    async function refresh() {
      try {
        const data = await fetch('/api/status', {cache:'no-store'}).then(r => r.json());
        const pct = data.total ? 100*data.done/data.total : 0;
        document.getElementById('overall').textContent = pct.toFixed(1)+'%';
        document.getElementById('done').textContent = data.done+' / '+data.total;
        document.getElementById('active-count').textContent = data.active_count;
        document.getElementById('pending').textContent = data.pending;
        document.getElementById('eta').textContent = Math.round(data.eta_low_hours)+'–'+Math.round(data.eta_high_hours)+' h';
        document.getElementById('finalizer').textContent = data.finalizer.state || 'not started';
        document.getElementById('finalizer-label').textContent = data.finalizer.step ? 'Postprocess · '+data.finalizer.step : 'Postprocess';
        document.getElementById('stamp').textContent = 'Updated '+data.updated_at+' · revision '+(data.revision || 'loading');
        const groups=document.getElementById('groups'); groups.replaceChildren();
        data.groups.forEach(g => { const row=node('div','bar-row'); row.append(node('div','',g.label)); const track=node('div','track'); const fill=node('div','fill'); fill.style.width=(100*g.done/g.total)+'%'; track.append(fill); row.append(track,node('div','count',g.done+' / '+g.total+(g.active?' · '+g.active+' active':''))); groups.append(row); });
        const gpus=document.getElementById('gpus'); gpus.replaceChildren();
        data.gpus.forEach(g => { const box=node('div','gpu'); const head=node('div','gpu-head'); head.append(node('strong','', 'GPU '+g.index),node('span','util',g.utilization===null?'n/a':g.utilization+'%')); box.append(head,node('div','task',g.task || 'Idle')); if(g.memory_mib!==null) box.append(node('div','task',g.memory_mib+' MiB')); gpus.append(box); });
        const active=document.getElementById('active'); active.replaceChildren();
        if (!data.active.length) active.append(empty('No active tasks.'));
        else { const table=node('table'); const tr=node('tr'); ['GPU','Task','Group','Within task'].forEach(x=>tr.append(node('th','',x))); const thead=node('thead'); thead.append(tr); table.append(thead); const body=node('tbody'); data.active.forEach(a=>{ const r=node('tr'); [a.gpu ?? '?',a.name,a.group].forEach(x=>r.append(node('td','',x))); const elapsed=a.elapsed_seconds===null?'':(' · '+Math.floor(a.elapsed_seconds/3600)+'h '+Math.floor((a.elapsed_seconds%3600)/60)+'m'); const progress=node('td','',a.detail+elapsed); if(a.total){ const track=node('div','task-progress'); const fill=node('span'); fill.style.width=(100*a.current/a.total)+'%'; track.append(fill); progress.append(track); } r.append(progress); body.append(r); }); table.append(body); active.append(table); }
        const recent=document.getElementById('recent'); recent.replaceChildren();
        if (!data.recent.length) recent.append(empty('No completed tasks yet.'));
        else { const table=node('table'); const body=node('tbody'); data.recent.forEach(name=>{ const r=node('tr'); r.append(node('td','',name)); body.append(r); }); table.append(body); recent.append(table); }
        const failures=document.getElementById('failures'); failures.replaceChildren();
        if (!data.failures.length) failures.append(empty('No failures.'));
        else data.failures.forEach(f=>failures.append(node('div','empty failure',f.job+' · '+f.log)));
        const measured=Object.entries(data.eta_basis).filter(([,v])=>v==='measured').map(([k])=>k).join(', ') || 'none';
        document.getElementById('footer').textContent='Status source: '+data.status_path+' · ETA measured groups: '+measured;
      } catch (error) { document.getElementById('stamp').textContent='Dashboard update failed: '+error; }
    }
    refresh(); setInterval(refresh,3000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpus", default="0,1,2,3,4,6")
    parser.add_argument("--ckpt-root", type=Path, default=default_data_root() / "ckpts")
    parser.add_argument("--status", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status is None:
        result_root = default_data_root() / "analyse/official_release_08fc8a21"
        args.status = result_root / "runner_status_droid_probe_depth_cost.json"
    state = DashboardState(args.ckpt_root, args.status, [part.strip() for part in args.gpus.split(",") if part.strip()])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/api/status":
                payload = json.dumps(state.snapshot()).encode()
                content_type = "application/json"
            elif self.path.split("?", 1)[0] in ("/", "/index.html"):
                payload = HTML.encode()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard=http://{args.host}:{args.port} status={args.status}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
