"""Read-only H-JEPA audit; writes only small reports beside this script.

No jobs are launched and no checkpoints, configs, or datasets are modified.
Run with the existing hjepa environment (PyYAML is required).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import re
import statistics

import yaml

ROOT = pathlib.Path('/home/songxw/H-JEPA')
DATA = pathlib.Path('/home/songxw/hjepa-data')
OUT = ROOT / 'local_env'
TZ = dt.timezone(dt.timedelta(hours=8))
CLIPS = {
    'ant_hjepa_l2': 9_200_000,
    'fourroom_hjepa_l2': 915_298,
    'fourroom_hjepa_l3': 840_478,
    'pusht_hjepa_l2': 1_701_446,
    'pusht_hjepa_l3': 1_143_623,
    'cube_hjepa_l2': 1_636_600,
    'cube_hjepa_l3': 1_342_600,
}


def stamp(epoch):
    return dt.datetime.fromtimestamp(epoch, TZ).isoformat(timespec='seconds')


def collect():
    now = dt.datetime.now(TZ)
    runs = []
    for run in sorted((DATA / 'ckpts').glob('*/*/seed*')):
        config = run / 'config.yaml'
        if not config.exists():
            continue
        cfg = yaml.safe_load(config.read_text())
        model = run.parent.name
        batch = cfg.get('loader', {}).get('batch_size')
        epochs = cfg.get('trainer', {}).get('max_epochs')
        row = {
            'environment': run.parent.parent.name,
            'model': model,
            'seed': cfg.get('seed'),
            'path': str(run),
            'batch_size': batch,
            'accumulate_grad_batches': cfg.get('trainer', {}).get('accumulate_grad_batches', 1),
            'epochs_requested': epochs,
            'precision': cfg.get('trainer', {}).get('precision'),
        }
        final = run / f'{model}_object.ckpt'
        row['final_checkpoint'] = str(final) if final.exists() else None
        row['training_finished_at'] = stamp(final.stat().st_mtime) if final.exists() else None
        summaries = sorted(run.glob('spt/runs/*/*/*/summary.json'), key=lambda f: f.stat().st_mtime)
        if summaries:
            s = json.loads(summaries[-1].read_text())
            row.update(summary_path=str(summaries[-1]), summary_step=s.get('step'),
                       epoch_zero_based=s.get('epoch'), summary_updated_at=stamp(s['updated_at']),
                       training_lr_last=s.get('metrics', {}).get('model_opt/level1', {}).get('last'))
        evals = []
        for p in sorted(run.glob('planning_eval/*/metrics.yaml')):
            m = yaml.safe_load(p.read_text())
            es = m.get('episode_successes', [])
            evals.append({'path': str(p), 'success_rate_pct': m.get('success_rate'),
                          'successful_tasks': sum(bool(x) for x in es), 'tasks': len(es),
                          'evaluation_seconds': m.get('evaluation_time'),
                          'chunk_size': m.get('chunk_size'), 'finished_at': stamp(p.stat().st_mtime)})
        row['planning'] = evals
        probe = run / 'final_probing_decoding_eval' / 'metrics.yaml'
        row['posthoc_finished'] = probe.exists()
        if probe.exists():
            row['posthoc_metrics_path'] = str(probe)
            row['posthoc_finished_at'] = stamp(probe.stat().st_mtime)
            row['posthoc_eval'] = {k.split('/eval/', 1)[-1]: v for k, v in
                                  yaml.safe_load(probe.read_text()).items() if '/eval/' in k}
        log = run / 'hydra' / 'main_hjepa.log'
        if not final.exists() and log.exists() and model in CLIPS and batch and epochs:
            marks = []
            for line in log.read_text(errors='replace').splitlines():
                m = re.search(r'^\[([^\]]+)\].*Resume checkpoint after train batch (\d+)', line)
                if m:
                    t = dt.datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=TZ)
                    marks.append((t, int(m[2])))
            recent = marks[-7:]
            rates = [(b[0]-a[0]).total_seconds()/(b[1]-a[1])
                     for a, b in zip(recent, recent[1:])
                     if b[1] > a[1]]
            if rates:
                rate = statistics.median(rates)
                total = (CLIPS[model] // batch) * epochs
                last_t, last_step = marks[-1]
                seconds_left = max(0, total-last_step) * rate
                row['training_eta'] = {
                    'basis': 'median seconds per logged microbatch over last six checkpoint intervals',
                    'valid_clips_per_epoch': CLIPS[model],
                    'total_microbatches_estimate': total,
                    'seconds_per_microbatch': rate,
                    'training_hours_remaining_at_last_mark': seconds_left / 3600,
                    'last_mark_time': last_t.isoformat(timespec='seconds'),
                    'estimated_finish': (last_t+dt.timedelta(seconds=seconds_left)).isoformat(timespec='seconds'),
                    'excludes': 'validation, future contention, final planning evaluation, posthoc probes/decoders',
                }
        runs.append(row)
    result = {'captured_at': now.isoformat(timespec='seconds'), 'runs': runs,
              'note': 'Single seed local runs; does not certify paper-protocol equivalence.'}
    (OUT / 'reproduction_status_20261027.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (OUT / 'reproduction_results_20261027.csv').open('w', newline='') as f:
        keys = ['environment','model','seed','batch_size','accumulate_grad_batches','training_finished_at',
                'success_rate_pct','successful_tasks','tasks','evaluation_seconds','chunk_size','posthoc_finished','metrics_path']
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in runs:
            for m in r['planning']:
                d = {k: r.get(k) for k in keys if k in r}
                d.update({k: m.get(k) for k in keys if k in m})
                d['metrics_path'] = m['path']
                w.writerow(d)
    for r in runs:
        print(json.dumps({k: r.get(k) for k in ['model','final_checkpoint','summary_step','epoch_zero_based',
                          'posthoc_finished','training_eta']}, ensure_ascii=False))


if __name__ == '__main__':
    collect()
