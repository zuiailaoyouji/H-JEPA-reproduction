"""Download and verify only the four official simulation evaluation task files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'h_jepa/assets/eval_trajs'
REVISION = 'cc05cb8d2aaf9aec0d5364d073c6369ff760b60b'
BASE = f'https://huggingface.co/datasets/jepa-world-models/h-jepa/resolve/{REVISION}/eval_trajs/'
FILES = {
    'fourroom': 'fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt',
    'ant': 'ant/expert_grid_d3_n50.pt',
    'cube': 'ogbench/goal_offset_20_pickup_val.pt',
    'pusht': 'pusht/goal_offset_75_val.pt',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envs', nargs='+', choices=FILES)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    checksums = {line.split()[1]: line.split()[0]
                 for line in (DEST / 'SHA256SUMS').read_text().splitlines() if line.strip()}
    for env in args.envs:
        relative = FILES[env]
        path = DEST / relative
        if path.exists():
            if digest(path) != checksums[relative]:
                raise RuntimeError(f'Existing file has wrong checksum; preserved for inspection: {path}')
            print(f'{env}: already downloaded, SHA256 OK', flush=True)
            continue
        part = path.with_name(path.name + '.part')
        if part.exists() and digest(part) == checksums[relative]:
            part.rename(path)
            print(f'{env}: completed partial file verified and finalized', flush=True)
            continue
        if args.verify_only:
            raise FileNotFoundError(f'Missing complete task file: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            'curl', '--proxy', 'http://127.0.0.1:17890', '--fail', '--location', '--http1.1',
            '--retry', '0',
            '--connect-timeout', '15', '--max-time', '1200',
            '--speed-time', '60', '--speed-limit', '1024', '--continue-at', '-',
            '--silent', '--show-error', '--output', str(part), BASE + relative,
        ]
        print(f'{env}: downloading {relative}', flush=True)
        process = subprocess.Popen(command)
        last_size = part.stat().st_size if part.exists() else 0
        last_time = time.monotonic()
        while process.poll() is None:
            time.sleep(15)
            size = part.stat().st_size if part.exists() else 0
            now = time.monotonic()
            rate = (size - last_size) / (now - last_time) / 1024**2
            print(f'{env}: {size / 1024**2:.1f} MiB downloaded; {rate:.2f} MiB/s', flush=True)
            last_size, last_time = size, now
        if process.returncode != 0:
            raise RuntimeError(f'{env}: curl exited {process.returncode}; partial file kept for resume')
        if digest(part) != checksums[relative]:
            raise RuntimeError(f'{env}: SHA256 mismatch; partial file kept for inspection')
        part.rename(path)
        print(f'{env}: download complete, SHA256 OK', flush=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    reports = []
    for env in args.envs:
        relative = FILES[env]
        path = DEST / relative
        # Only deserialize official files after their published SHA256 matches.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert isinstance(payload, dict) and 'data' in payload and 'columns' in payload
        assert len(payload['data']) == 50, (env, len(payload['data']))
        assert 'pixels' in payload['columns'], (env, payload['columns'])
        result = {'env': env, 'path': str(path), 'bytes': path.stat().st_size,
                  'tasks': len(payload['data']), 'columns': list(payload['columns']),
                  'sha256': checksums[relative], 'revision': REVISION,
                  'hash_check': 'passed', 'payload_load': 'passed'}
        reports.append(result)
        print(f'{env}: official task payload loaded on CPU, 50 tasks verified', flush=True)
        del payload
    (ROOT / ('local_env/eval-task-validation-' + '-'.join(args.envs) + '.json')).write_text(
        json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
