#!/usr/bin/env python3
"""Verify a Hugging Face model-tree snapshot and write a pinned release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path: Path, lfs: bool) -> str:
    hasher = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tree-json", type=Path, required=True)
    parser.add_argument("--repository", default="jepa-world-models/h-jepa")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [entry for entry in json.loads(args.tree_json.read_text()) if entry["type"] == "file"]

    def verify(entry: dict) -> tuple[str, dict, str | None]:
        path = args.root / entry["path"]
        lfs = entry.get("lfs") is not None
        expected_digest = entry["lfs"]["oid"] if lfs else entry["oid"]
        expected_size = int(entry["size"])
        if not path.exists():
            return entry["path"], {}, "missing"
        if path.stat().st_size != expected_size:
            return entry["path"], {}, f"size {path.stat().st_size} != {expected_size}"
        actual_digest = digest(path, lfs)
        if actual_digest != expected_digest:
            return entry["path"], {}, f"digest {actual_digest} != {expected_digest}"
        metadata = {"size": expected_size}
        metadata["sha256" if lfs else "git_oid"] = expected_digest
        return entry["path"], metadata, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(verify, entries))
    failures = {path: error for path, _metadata, error in results if error}
    if failures:
        for path, error in failures.items():
            print(f"FAIL {path}: {error}")
        raise SystemExit(f"official release verification failed for {len(failures)}/{len(entries)} files")

    files = {path: metadata for path, metadata, _error in results}
    checkpoints = sum(path.endswith("_object.ckpt") for path in files)
    normalizers = sum(path.endswith("/normalizer.pt") for path in files)
    if (len(files), checkpoints, normalizers) != (189, 93, 93):
        raise SystemExit(
            f"unexpected release shape: files={len(files)}, checkpoints={checkpoints}, normalizers={normalizers}"
        )
    manifest = {
        "repository": args.repository,
        "revision": args.revision,
        "file_count": len(files),
        "checkpoint_count": checkpoints,
        "normalizer_count": normalizers,
        "total_bytes": sum(metadata["size"] for metadata in files.values()),
        "files": files,
    }
    if args.write_manifest:
        output = args.root / "OFFICIAL_WEIGHTS_MANIFEST.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(output)
    print(
        f"verified {len(files)} files, {checkpoints} checkpoints, {normalizers} normalizers, "
        f"{manifest['total_bytes']} bytes at {args.revision}"
    )


if __name__ == "__main__":
    main()
