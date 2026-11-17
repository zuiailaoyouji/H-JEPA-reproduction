"""Resumable, verified range downloads for the official simulation task files."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'h_jepa/assets/eval_trajs'
REV = 'cc05cb8d2aaf9aec0d5364d073c6369ff760b60b'
FILES = {
    'fourroom': ('fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt', 573644209),
    'pusht': ('pusht/goal_offset_75_val.pt', 564814221),
    'ant': ('ant/expert_grid_d3_n50.pt', 155119789),
    'cube': ('ogbench/goal_offset_20_pickup_val.pt', 152425457),
}
CHUNK = 16 * 1024**2
PROXY = 'http://127.0.0.1:17890'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def fetch_chunk(url, path, start, end, total):
    need = end - start + 1
    for attempt in range(8):
        have = path.stat().st_size if path.exists() else 0
        if have == need:
            return
        assert have < need, (path, have, need)
        requested_start = start + have
        tail = path.with_suffix('.response')
        headers = path.with_suffix('.headers')
        command = [
            'curl', '--proxy', PROXY, '--http1.1', '--fail', '--location',
            '--retry', '0', '--connect-timeout', '15', '--max-time', '180',
            '--speed-time', '45', '--speed-limit', '1024',
            '--range', f'{requested_start}-{end}', '--max-filesize', str(need - have),
            '--silent', '--show-error', '--dump-header', str(headers),
            '--output', str(tail), url,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        matches = re.findall(r'^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                             headers.read_text(errors='replace') if headers.exists() else '',
                             flags=re.IGNORECASE | re.MULTILINE)
        valid_range = matches and tuple(map(int, matches[-1])) == (requested_start, end, total)
        got = tail.stat().st_size if tail.exists() else 0
        if valid_range and 0 < got <= need - have:
            with path.open('ab') as dst, tail.open('rb') as src:
                shutil.copyfileobj(src, dst, 1024**2)
            tail.unlink()
            if path.stat().st_size == need:
                headers.unlink(missing_ok=True)
                return
        else:
            tail.unlink(missing_ok=True)
        print(f'  range {start}-{end}: retry {attempt + 1}, retained '
              f'{path.stat().st_size if path.exists() else 0} bytes; curl={result.returncode}', flush=True)
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(f'Range failed after retries; retained partial chunk {path}')


def download(env, expected):
    relative, size = FILES[env]
    target = DEST / relative
    if target.exists():
        assert target.stat().st_size == size and sha256(target) == expected
        print(f'{env}: already complete, SHA256 OK', flush=True)
        return
    directory = target.parent / (target.name + '.ranges')
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, size, CHUNK)):
        end = min(size, start + CHUNK) - 1
        chunks.append((directory / f'{index:04d}.chunk', start, end))
    # Salvage the valid prefix from the earlier single-stream transfer.
    legacy = target.with_name(target.name + '.part')
    if legacy.exists():
        prefix_size = min(legacy.stat().st_size, size)
        with legacy.open('rb') as src:
            for path, start, end in chunks:
                if start >= prefix_size:
                    break
                length = min(end + 1, prefix_size) - start
                if not path.exists():
                    src.seek(start)
                    path.write_bytes(src.read(length))
    def completed_bytes():
        return sum(p.stat().st_size for p, _, _ in chunks if p.exists())
    print(f'{env}: {size / 1024**2:.1f} MiB, 4 range connections; '
          f'reusing {completed_bytes() / 1024**2:.1f} MiB', flush=True)
    url = f'https://huggingface.co/datasets/jepa-world-models/h-jepa/resolve/{REV}/eval_trajs/{relative}'
    started = time.monotonic()
    last_time, last_bytes = started, completed_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch_chunk, url, p, start, end, size) for p, start, end in chunks}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=15,
                                                   return_when=concurrent.futures.FIRST_EXCEPTION)
            for future in done:
                future.result()
            now, n = time.monotonic(), completed_bytes()
            print(f'{env}: {n / 1024**2:.1f}/{size / 1024**2:.1f} MiB '
                  f'({100*n/size:.1f}%), {(n-last_bytes)/(now-last_time)/1024**2:.2f} MiB/s', flush=True)
            last_time, last_bytes = now, n
    assembled = target.with_name(target.name + '.assembling')
    digest = hashlib.sha256()
    with assembled.open('wb') as dst:
        for path, start, end in chunks:
            assert path.stat().st_size == end-start+1
            with path.open('rb') as src:
                for block in iter(lambda: src.read(8 * 1024**2), b''):
                    dst.write(block)
                    digest.update(block)
    assert assembled.stat().st_size == size
    if digest.hexdigest() != expected:
        raise RuntimeError(f'{env}: SHA256 mismatch; all downloaded pieces retained')
    assembled.rename(target)
    for path, _, _ in chunks:
        path.unlink()
    directory.rmdir()
    legacy.unlink(missing_ok=True)
    print(f'{env}: finalized, SHA256 OK, {time.monotonic()-started:.1f} seconds', flush=True)


if __name__ == '__main__':
    checksums = {line.split()[1]: line.split()[0]
                 for line in (DEST / 'SHA256SUMS').read_text().splitlines() if line.strip()}
    for env in sys.argv[1:]:
        download(env, checksums[FILES[env][0]])
