#!/usr/bin/env python3
"""Check the published files and the archived experiment coverage using stdlib."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    count = 0
    for line in (ROOT / 'provenance/MANIFEST.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit('File checksum failed: ' + name)
        count += 1
    coverage = json.loads((ROOT / 'results/official/combined_coverage.json').read_text())
    expected = {'depth': 84, 'cost_ladder': 108, 'droid': 27, 'probe': 18}
    if not coverage['complete']:
        raise SystemExit('Coverage is marked incomplete')
    for group, n in expected.items():
        item = coverage['experiments'][group]
        if (item['found'], item['expected']) != (n, n):
            raise SystemExit('Coverage mismatch: ' + group)
    rows = {}
    for name, n in [('depth_raw.csv', 84), ('cost_ladder_raw.csv', 108), ('droid_raw.csv', 27)]:
        with (ROOT / 'results/official' / name).open(newline='') as handle:
            actual = len(list(csv.DictReader(handle)))
        if actual != n:
            raise SystemExit(f'{name}: expected {n} rows, found {actual}')
        rows[name] = actual
    print(json.dumps({'verified_files': count, 'experiment_cells': sum(expected.values()), 'coverage': expected, 'raw_rows': rows}, indent=2))


if __name__ == '__main__':
    main()
