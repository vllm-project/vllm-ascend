#!/usr/bin/env python3
"""Classic LRU governance for a build snapshot cache (default 5GB budget).

Nothing here is csrc-specific: any "directory to snapshot per target"
cache works.

Snapshots and the index ledger live in a local cache directory (default
~/.cache/csrc_cache, configurable via --store-dir). The ledger tracks size,
creation time and last access per entry; this script implements the classic
LRU algorithm over it:

  --touch KEY             bump last_access (call after every restore hit)
  --add KEY --size N      record a newly saved snapshot (size accepts 5G,
                          512MiB, 5368709120)
  --evict                 evict least-recently-used entries until total size
                          fits the budget (per-target latest entry protected
                          by default)
  --list                  audit view: totals, per-target breakdown, LRU order
  --upload PATH ...       tar+zstd PATH -> store -> --add -> --evict
  --restore --target T    extract newest snapshot for T -> touch
  --init                  create an empty ledger

All knobs are parameters: budget size (--budget 5G), cache directory
(--store-dir, default ~/.cache/csrc_cache), ledger name (--index-name),
snapshot naming (--name-prefix), snapshot identity (--snapshot-id).

Exit codes: 0 = ok, 1 = no snapshot available (cold target), 2 = fail-closed
(ledger unusable), 3 = unexpected.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_STORE_DIR = "~/.cache/csrc_cache"
DEFAULT_INDEX_NAME = "index.json"
DEFAULT_BUDGET = "5G"  # human-readable; parsed by parse_size()
DEFAULT_NAME_PREFIX = "snapshot"
INDEX_SCHEMA = 1


def log(msg: str) -> None:
    print(f"[lru] {msg}", flush=True)


def parse_size(text: str) -> int:
    """Accept '5G', '5GB', '512MiB', '1.5GB', '5368709120' -> bytes."""
    s = text.strip().upper().replace("IB", "")  # KIB/MIB/GB -> K/M/G
    units = {
        "TB": 1024**4,
        "GB": 1024**3,
        "MB": 1024**2,
        "KB": 1024,
        "B": 1,
        "T": 1024**4,
        "G": 1024**3,
        "M": 1024**2,
        "K": 1024,
    }
    for unit in ("TB", "GB", "MB", "KB", "T", "G", "M", "K", "B"):
        if s.endswith(unit):
            num = s[: -len(unit)].strip()
            try:
                value = float(num)
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid size {text!r}: {num!r} is not a number")
            if value < 0:
                raise argparse.ArgumentTypeError(f"invalid size {text!r}: negative")
            return int(value * units[unit])
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid size {text!r}: no unit and not a plain byte count")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Local cache store: snapshot archives + index ledger in one directory
# ---------------------------------------------------------------------------


class LocalStore:
    """Directory-backed store (default ~/.cache/csrc_cache)."""

    def __init__(self, root: Path, index_name: str) -> None:
        self.root = root
        self.index_name = index_name
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, path: Path) -> None:
        (self.root / name).write_bytes(path.read_bytes())

    def get(self, name: str, dest: Path) -> None:
        import shutil

        shutil.copyfile(self.root / name, dest)

    def delete(self, name: str) -> None:
        p = self.root / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def load_index(store) -> dict:
    p = store.root / store.index_name
    if not p.is_file():
        return new_index()
    return json.loads(p.read_text())


def new_index() -> dict:
    return {"schema": INDEX_SCHEMA, "entries": {}}


def save_index(store, index: dict) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(index, f, indent=2, sort_keys=True)
        tmp = Path(f.name)
    store.put(store.index_name, tmp)
    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Classic LRU
# ---------------------------------------------------------------------------


def lru_evict(index: dict, budget: int, protect_latest_per_target: bool, dry_run: bool) -> list[dict]:
    """Evict LRU-first until total size fits the budget.

    Returns the eviction list (already applied to the index unless dry-run).
    """
    entries = index["entries"]
    total = sum(e["size"] for e in entries.values())
    if total <= budget:
        log(f"within budget: {total} / {budget} bytes ({len(entries)} entries), nothing to evict")
        return []

    # Dry-run must not mutate the ledger: work on a copy of the entries.
    snapshot = dict(entries) if dry_run else None

    protected: set[str] = set()
    if protect_latest_per_target:
        newest_by_target: dict[str, tuple[float, str]] = {}
        for key, e in entries.items():
            cur = newest_by_target.get(e["target"])
            if cur is None or e["created"] > cur[0]:
                newest_by_target[e["target"]] = (e["created"], key)
        protected = {key for _, key in newest_by_target.values()}

    # Classic LRU order: least recently accessed first.
    order = sorted(entries.items(), key=lambda kv: kv[1]["last_access"])
    evicted, running = [], total
    for key, e in order:
        if running <= budget:
            break
        if key in protected:
            continue
        evicted.append({"key": key, "size": e["size"], "target": e["target"], "last_access": e["last_access"]})
        running -= e["size"]
        del index["entries"][key]

    if running > budget:
        log(
            f"::warning::budget still exceeded ({running} > {budget}) after "
            f"evicting {len(evicted)} non-protected entries; protected "
            f"per-target latest snapshots pin the remainder"
        )
    if dry_run:
        # Restore the pre-eviction view (dry-run must not mutate).
        index["entries"] = snapshot
    return evicted


def fmt_size(n: int) -> str:
    return f"{n / 1024**3:.2f}GiB" if n >= 1024**3 else f"{n / 1024**2:.0f}MiB"


def compression() -> tuple[list[str], str, str]:
    """Prefer zstd (CI installs it); fall back to gzip for local runs."""
    try:
        run(["zstd", "--version"])
        return (["-I", "zstd"], ".tar.zst", "zstd")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return (["-z"], ".tar.gz", "gzip")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def make_store(args) -> object:
    return LocalStore(Path(args.store_dir).expanduser(), args.index_name)


def cmd_touch(args) -> int:
    store = make_store(args)
    index = load_index(store)
    key = args.key
    if key not in index["entries"]:
        log(f"::warning::touch on unknown key {key} (evicted or never added); LRU precision only, not an error")
        return 0
    index["entries"][key]["last_access"] = time.time()
    save_index(store, index)
    log(f"touched {key}")
    return 0


def cmd_add(args) -> int:
    store = make_store(args)
    index = load_index(store)
    key = args.key
    now = time.time()
    if key in index["entries"]:
        # Idempotent re-add of identical content == an access.
        index["entries"][key]["last_access"] = now
        save_index(store, index)
        log(f"key already present, refreshed last_access: {key}")
        return 0
    index["entries"][key] = {
        "size": args.size,
        "target": args.target,
        "created": now,
        "last_access": now,
    }
    save_index(store, index)
    log(f"added {key}: {args.size} bytes ({fmt_size(args.size)}) target={args.target}")
    return 0


def cmd_evict(args) -> int:
    store = make_store(args)
    index = load_index(store)
    budget = args.budget
    evicted = lru_evict(index, budget, args.protect_latest, args.dry_run)
    if evicted and not args.dry_run:
        for e in evicted:
            store.delete(e["key"])
        save_index(store, index)
    for e in evicted:
        log(
            f"  evicted {e['key']} ({fmt_size(e['size'])}, "
            f"last_access={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(e['last_access']))}, "
            f"target={e['target']})"
        )
    total = sum(e["size"] for e in index["entries"].values())
    log(
        f"evict done: {len(evicted)} removed, ledger now {total} / {budget} "
        f"bytes ({len(index['entries'])} entries)" + (" [dry-run: assets and ledger untouched]" if args.dry_run else "")
    )
    return 0


def cmd_list(args) -> int:
    store = make_store(args)
    index = load_index(store)
    entries = index["entries"]
    budget = args.budget
    total = sum(e["size"] for e in entries.values())
    log(f"ledger: {len(entries)} entries, {total} / {budget} bytes ({fmt_size(total)} / {fmt_size(budget)})")
    by_target: dict[str, list[tuple[str, dict]]] = {}
    for key, e in entries.items():
        by_target.setdefault(e["target"], []).append((key, e))
    for target in sorted(by_target):
        items = by_target[target]
        tsize = sum(e["size"] for _, e in items)
        oldest = min(e["last_access"] for _, e in items)
        log(
            f"  {target}: {len(items)} snapshots, {fmt_size(tsize)}, "
            f"oldest access "
            f"{time.strftime('%Y-%m-%d', time.gmtime(oldest))}"
        )
    # LRU order preview (what --evict would consider first).
    order = sorted(entries.items(), key=lambda kv: kv[1]["last_access"])
    log("LRU eviction candidates (oldest first):")
    for key, e in order[:10]:
        log(
            f"    {key} {fmt_size(e['size'])} "
            f"last_access={time.strftime('%Y-%m-%dT%H:%M', time.gmtime(e['last_access']))}"
        )
    if len(order) > 10:
        log(f"    ... {len(order) - 10} more")
    return 0


def cmd_init(args) -> int:
    store = make_store(args)
    save_index(store, new_index())
    log(f"initialized empty ledger at {store.index_name}")
    return 0


def cmd_upload(args) -> int:
    store = make_store(args)
    src = Path(args.path).resolve()
    if not src.is_dir():
        print(f"::error::--upload path is not a directory: {src}", flush=True)
        return 2
    flags, suffix, algo = compression()
    name = f"{args.name_prefix}-{args.target}-{args.snapshot_id}{suffix}"
    archive = src.parent / name
    t0 = time.monotonic()
    # The archive stores the build dir relative to its parent, so
    # --upload <dir> restores as <dest>/<dir-name> with --dest <parent>.
    run(["tar", "-cf", str(archive), *flags, "-C", str(src.parent), src.name])
    size = archive.stat().st_size
    log(f"packed {archive} with {algo} ({fmt_size(size)}) in {time.monotonic() - t0:.1f}s")
    store.put(name, archive)
    archive.unlink(missing_ok=True)
    args.key = name
    args.size = size
    cmd_add(args)
    args.dry_run = False
    cmd_evict(args)
    cmd_list(args)
    return 0


def cmd_restore(args) -> int:
    store = make_store(args)
    index = load_index(store)
    items = [(k, e) for k, e in index["entries"].items() if e["target"] == args.target]
    if not items:
        log(f"no snapshot for target {args.target} (cold or evicted); caller should fall back to a full build")
        return 1
    key, _ = max(items, key=lambda kv: kv[1]["created"])
    flags, _suffix, algo = compression()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    store.get(key, dest / key)
    run(["tar", "-xf", str(dest / key), *flags, "-C", str(dest)])
    (dest / key).unlink(missing_ok=True)
    log(f"restored {key} -> {dest} with {algo} in {time.monotonic() - t0:.1f}s")
    args.key = key
    cmd_touch(args)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", default=DEFAULT_STORE_DIR, help=f"cache directory, default {DEFAULT_STORE_DIR}")
    ap.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help=f"ledger file name, default {DEFAULT_INDEX_NAME}")
    ap.add_argument(
        "--budget",
        type=parse_size,
        default=DEFAULT_BUDGET,
        help="cache budget as compressed bytes; accepts 5G, 512MiB, 5368709120. Default 5G",
    )
    ap.add_argument(
        "--name-prefix", default=DEFAULT_NAME_PREFIX, help=f"snapshot asset name prefix, default {DEFAULT_NAME_PREFIX}"
    )
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--init", action="store_true")
    sub.add_argument("--touch", action="store_true")
    sub.add_argument("--add", action="store_true")
    sub.add_argument("--evict", action="store_true")
    sub.add_argument("--list", action="store_true")
    sub.add_argument("--upload", action="store_true")
    sub.add_argument("--restore", action="store_true")
    ap.add_argument("--key", help="asset/ledger key for --touch/--add")
    ap.add_argument("--size", type=parse_size, help="compressed size for --add (5G / 512MiB / bytes)")
    ap.add_argument("--target", help="cache target id (e.g. a2-arm64-ubuntu)")
    ap.add_argument("--snapshot-id", help="with --upload: snapshot version identifier (e.g. the csrc hash)")
    ap.add_argument(
        "--path",
        help="with --upload: directory to pack; it is "
        "archived relative to its parent, so --upload csrc/build "
        "unpacks to <dest>/build",
    )
    ap.add_argument("--dest", help="with --restore: extraction parent dir (e.g. --dest csrc unpacks to csrc/build)")
    ap.add_argument(
        "--no-protect-latest",
        dest="protect_latest",
        action="store_false",
        default=True,
        help="disable per-target latest-snapshot protection",
    )
    ap.add_argument("--dry-run", action="store_true", help="with --evict: report without deleting")
    args = ap.parse_args()
    log(f"budget={args.budget} bytes ({fmt_size(args.budget)}) store-dir={args.store_dir}")
    import traceback

    try:
        if args.init:
            return cmd_init(args)
        if args.touch:
            return cmd_touch(args)
        if args.add:
            if not (args.key and args.size and args.target):
                print("::error::--add requires --key --size --target", flush=True)
                return 2
            return cmd_add(args)
        if args.evict:
            return cmd_evict(args)
        if args.list:
            return cmd_list(args)
        if args.upload:
            if not (args.path and args.target and args.snapshot_id):
                print("::error::--upload requires --path --target --snapshot-id", flush=True)
                return 2
            return cmd_upload(args)
        if args.dest and args.target:
            return cmd_restore(args)
        print("::error::--restore requires --target and --dest", flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        print(
            f"::error::backend command failed: {' '.join(exc.cmd[:4])}... (exit {exc.returncode})\n{exc.stderr}",
            flush=True,
        )
        return 2
    except BrokenPipeError:
        # Stdout was truncated by a downstream pipe (e.g. `| head`); the
        # useful output is already delivered, exit quietly.
        with contextlib.suppress(BrokenPipeError):
            sys.stdout.close()
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CI error surface
        print(f"::error::csrc_cache_lru failed unexpectedly: {exc!r}", flush=True)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
