#!/usr/bin/env python3
"""Snapshot store for the csrc build-tree cache (no retention logic).

The store is a plain directory of tar archives named
``snapshot-<target>-<snapshot-id>.tar.<zst|gz>``. Retention is fully
delegated to the store owner: in CI the store directory is persisted via
the GitHub cache, whose own LRU eviction governs cross-entry retention
(each saved entry is pruned to a single archive by the caller), so this
tool intentionally ships no budget/eviction ledger. The file name keeps
the historical "lru" suffix only for call-site compatibility.

Commands:
  --upload --path P --target T --snapshot-id S
                        Pack P (tar+zstd, gzip fallback) into the store as
                        snapshot-<T>-<S>.tar.zst. P is archived relative to
                        its parent, so --upload csrc/build restores as
                        <dest>/build.
  --restore --target T --dest D
                        Extract the newest snapshot for T (by file mtime)
                        into D.

Exit codes: 0 = ok, 1 = no snapshot for the target (cold), 2 = bad usage,
3 = unexpected failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_STORE_DIR = "~/.cache/csrc_cache"
SNAPSHOT_PREFIX = "snapshot"


def log(msg: str) -> None:
    print(f"[snapshot-store] {msg}", flush=True)


def zstd_available() -> bool:
    try:
        subprocess.run(["zstd", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def tar_flags(suffix: str) -> list[str]:
    """Compression flags for a full archive suffix ('.tar.zst' or '.tar.gz')."""
    return ["-I", "zstd"] if suffix == ".tar.zst" else ["-z"]


def cmd_upload(args: argparse.Namespace) -> int:
    src = Path(args.path).resolve()
    if not src.is_dir():
        print(f"::error::--upload path is not a directory: {src}", flush=True)
        return 2
    store = Path(args.store_dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    suffix = ".tar.zst" if zstd_available() else ".tar.gz"
    name = f"{SNAPSHOT_PREFIX}-{args.target}-{args.snapshot_id}{suffix}"
    # Pack inside the store under a temp name and rename atomically so a
    # concurrent --restore never observes a half-written archive.
    archive = store / f".tmp-{name}"
    t0 = time.monotonic()
    subprocess.run(
        ["tar", "-cf", str(archive), *tar_flags(suffix), "-C", str(src.parent), src.name],
        check=True,
    )
    final = store / name
    os.replace(archive, final)
    log(f"uploaded {name} ({final.stat().st_size / 1024**2:.1f}MiB) in {time.monotonic() - t0:.1f}s")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    store = Path(args.store_dir).expanduser()
    if store.is_dir():
        candidates = [
            p
            for p in store.glob(f"{SNAPSHOT_PREFIX}-{args.target}-*.tar.*")
            if p.suffix in (".zst", ".gz") and p.is_file()
        ]
    else:
        candidates = []
    if not candidates:
        log(f"no snapshot for target {args.target} (cold); caller should fall back to a full build")
        return 1
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    suffix = ".tar.zst" if newest.suffix == ".zst" else ".tar.gz"
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    subprocess.run(
        ["tar", "-xf", str(newest), *tar_flags(suffix), "-C", str(dest)],
        check=True,
    )
    log(f"restored {newest.name} -> {dest} in {time.monotonic() - t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", default=DEFAULT_STORE_DIR, help=f"cache directory, default {DEFAULT_STORE_DIR}")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--upload", action="store_true")
    sub.add_argument("--restore", action="store_true")
    ap.add_argument("--path", help="with --upload: directory to pack")
    ap.add_argument("--target", help="cache target id (e.g. a2-arm64-ubuntu)")
    ap.add_argument("--snapshot-id", help="with --upload: snapshot version identifier (e.g. the csrc hash)")
    ap.add_argument("--dest", help="with --restore: extraction parent dir")
    args = ap.parse_args()
    log(f"store-dir={args.store_dir}")
    try:
        if args.upload:
            if not (args.path and args.target and args.snapshot_id):
                print("::error::--upload requires --path --target --snapshot-id", flush=True)
                return 2
            return cmd_upload(args)
        if args.restore:
            if not (args.target and args.dest):
                print("::error::--restore requires --target and --dest", flush=True)
                return 2
            return cmd_restore(args)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"::error::backend command failed: {' '.join(exc.cmd[:4])}... (exit {exc.returncode})", flush=True)
        return 3
    except Exception as exc:  # noqa: BLE001 - top-level CI error surface
        print(f"::error::snapshot store failed unexpectedly: {exc!r}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
