#!/usr/bin/env python3
"""Verify and restore the release attachments without assembling checkpoint parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        with path.open('rb') as handle:
            for buf in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                h.update(buf)
    return h.hexdigest()


def destination(name, archive, data_dir):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe archive member: ' + name)
    if not rel.parts:
        return None
    head, *tail = rel.parts
    if archive == 'fixed_eval_tasks.tar.zst' and head == 'assets':
        return ROOT.joinpath('code/h_jepa/assets', *tail)
    if head == 'hjepa-data':
        return data_dir.joinpath(*tail)
    if head == 'source':
        return data_dir.joinpath('archive_source', *tail)
    raise ValueError('Unexpected archive member: ' + name)


def unpack(paths, archive, data_dir):
    feed = subprocess.Popen(['cat', *map(str, paths)], stdout=subprocess.PIPE)
    decode = subprocess.Popen(['zstd', '-dc'], stdin=feed.stdout, stdout=subprocess.PIPE)
    feed.stdout.close()
    try:
        with tarfile.open(fileobj=decode.stdout, mode='r|') as handle:
            for member in handle:
                target = destination(member.name, archive, data_dir)
                if target is None:
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    # Neither archived links nor pre-existing symlink parents may escape the destination.
                    boundary = ROOT / 'code/h_jepa/assets' if archive == 'fixed_eval_tasks.tar.zst' else data_dir
                    target.resolve().relative_to(boundary.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(member) as src, target.open('wb') as out:
                        shutil.copyfileobj(src, out)
                else:
                    raise ValueError('Unsupported archive member type: ' + member.name)
        # Consume the stream to let both subprocesses exit normally.
        while decode.stdout.read(1024 * 1024):
            pass
    finally:
        decode.stdout.close()
        decode_result = decode.wait()
        feed_result = feed.wait()
    if decode_result or feed_result:
        raise RuntimeError(f'Archive decompression failed: {archive}')


def verify_members(manifest_name, archive, data_dir):
    records = json.loads((ROOT / 'provenance' / manifest_name).read_text())['files']
    for rec in records:
        path = destination(rec['path'], archive, data_dir)
        if not path.is_file() or path.stat().st_size != rec['bytes'] or digest([path]) != rec['sha256']:
            raise RuntimeError('Restored file check failed: ' + rec['path'])
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--skip-checkpoints', action='store_true')
    parser.add_argument('--verify-only', action='store_true', help='Verify attachments without extraction.')
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads((ROOT / 'provenance/large_assets.json').read_text())
    checked = []
    restored = {}
    for archive in manifest['archives']:
        name = archive['name']
        if args.skip_checkpoints and name == 'local_checkpoints.tar.zst':
            continue
        parts = archive['parts'] or [archive]
        paths = [args.assets_dir / item['name'] for item in parts]
        for item, path in zip(parts, paths):
            if not path.is_file() or path.stat().st_size != item['bytes'] or digest([path]) != item['sha256']:
                raise SystemExit('Attachment verification failed: ' + str(path))
        if archive['parts'] and digest(paths) != archive['sha256']:
            raise SystemExit('Combined checkpoint stream checksum failed')
        checked.append(name)
        if args.verify_only or name.endswith('.zip'):
            continue
        if name == 'cost_geometry.csv.gz':
            target = ROOT / 'results/official/qualified_figure7/cost_geometry.csv.gz'
            shutil.copy2(paths[0], target)
            restored[name] = 1
            continue
        unpack(paths, name, data_dir)
        if name == 'raw_results.tar.zst':
            restored[name] = verify_members('raw_results_manifest.json', name, data_dir)
        elif name == 'local_checkpoints.tar.zst':
            restored[name] = verify_members('local_checkpoints_manifest.json', name, data_dir)
        elif name == 'fixed_eval_tasks.tar.zst':
            restored[name] = verify_members('fixed_eval_tasks_manifest.json', name, data_dir)
    if not args.verify_only:
        target = data_dir / 'ckpts/OFFICIAL_WEIGHTS_MANIFEST.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'provenance/assets/OFFICIAL_WEIGHTS_MANIFEST.json', target)
    print(json.dumps({'verified_archives': checked, 'restored_files': restored, 'data_dir': str(data_dir)}, indent=2))


if __name__ == '__main__':
    main()
