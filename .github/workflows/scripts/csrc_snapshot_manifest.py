#!/usr/bin/env python3
"""Per-operator manifest and invalidation for the csrc build snapshot cache.

Modes:
  --generate            Scan csrc sources + build artifacts, write
                        csrc/build/.cache-manifest.json (run after a build).
  --invalidate          Compare the snapshot manifest against current sources,
                        remove stale per-op artifacts (.done files are the
                        ninja skip switches) so the next build recompiles
                        only the ops whose inputs changed.
  --check-freshness OPS Verify that the artifacts of the listed ops were
                        rebuilt after a reference timestamp (post-build
                        self-check, fails closed).

All hashes are working-tree content digests: the build consumes working-tree
files (src_copy does `cp -rf`), so uncommitted edits must be visible to the
invalidation logic. Exit codes: 0 = all ops cached / checks passed,
1 = some ops invalidated (incremental rebuild needed), 2 = full rebuild
(global change or unusable manifest), 3 = unexpected failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import regex as re

MANIFEST_NAME = ".cache-manifest.json"
MANIFEST_SCHEMA = 1

# Same pathspec set as the existing csrc_hash computation, before the per-op
# directories are carved out.
GLOBAL_PATHSPEC = ["csrc", "setup.py", "CMakeLists.txt", "cmake"]

# Directory search order copied from build_aclnn.sh resolve_op_dir().
OP_CATEGORIES = ["moe", "gmm", "attention", "mc2", "ffn", "posembedding"]


# SOC_VERSION (e.g. ascend910b1) -> build subdirectory (SOC_ARG).
def soc_dir_for(soc_version: str) -> str:
    if soc_version.startswith("ascend910b"):
        return "ascend910b"
    if soc_version.startswith("ascend910_93"):
        return "ascend910_93"
    if soc_version.startswith("ascend310"):
        return "ascend310p"
    if soc_version.startswith("ascend950"):
        return "ascend950"
    raise SystemExit(f"::error::unknown soc_version {soc_version!r}")


def run_git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def soc_branch_for(soc_version: str) -> str:
    if soc_version.startswith("ascend910b"):
        return "ascend910b"
    if soc_version.startswith("ascend910_93"):
        return "ascend910_93"
    if soc_version.startswith("ascend310"):
        return "ascend310"
    if soc_version.startswith("ascend950"):
        return "ascend950"
    raise SystemExit(f"::error::no SOC branch for {soc_version!r}")


def parse_custom_ops_branch(repo: Path, branch: str) -> list[str]:
    """Extract the CUSTOM_OPS_ARRAY for one SOC branch from build_aclnn.sh."""
    script = (repo / "csrc" / "build_aclnn.sh").read_text(encoding="utf-8")
    marker = re.search(rf'if \[\[ "\$SOC_VERSION" =~ \^{re.escape(branch)}', script)
    if not marker:
        raise SystemExit(f"::error::SOC branch {branch} not found in build_aclnn.sh")
    section = script[marker.start() :]
    m = re.search(r"CUSTOM_OPS_ARRAY=\((.*?)\)", section, re.DOTALL)
    if not m:
        raise SystemExit("::error::CUSTOM_OPS_ARRAY not parsed")
    return re.findall(r'"([^"]+)"', m.group(1))


def parse_custom_ops(repo: Path, soc_version: str) -> list[str]:
    """Extract the CUSTOM_OPS_ARRAY for the SOC branch from build_aclnn.sh."""
    return parse_custom_ops_branch(repo, soc_branch_for(soc_version))


def global_exclude_prefixes(repo: Path) -> list[str]:
    """Op-directory prefixes excluded from the global hash: the UNION of the
    op directories of every SOC branch, not just the current one.

    An op that this SOC does not compile cannot affect this SOC's artifacts,
    so its directory must stay out of the global inputs - otherwise a change
    to it is misread as a global change and discards the snapshot for SOCs
    that never compile it (observed live: an upstream msa_index_score change,
    an a2/a3/a5-only op, forced a full 310p rebuild). Changes to ops outside
    the current SOC's list are then simply ignored, which is the correct
    behavior. Branches are discovered from the script itself so new SOC
    branches are picked up automatically.
    """
    script = (repo / "csrc" / "build_aclnn.sh").read_text(encoding="utf-8")
    branches = re.findall(r'SOC_VERSION" =~ \^([A-Za-z0-9_]+)', script)
    if not branches:
        raise SystemExit("::error::no SOC branches found in build_aclnn.sh")
    prefixes: dict[str, None] = {}
    for branch in branches:
        for op in parse_custom_ops_branch(repo, branch):
            d = resolve_op_dir(repo, op)
            if d is not None:
                prefixes.setdefault(d.relative_to(repo).as_posix() + "/")
    return sorted(prefixes)


def resolve_op_dir(repo: Path, op: str) -> Path | None:
    csrc = repo / "csrc"
    for cat in OP_CATEGORIES:
        cand = csrc / cat / op
        if cand.is_dir():
            return cand
    # Fallback shape of build_aclnn.sh (find csrc -maxdepth 3), guarded so
    # missing category dirs cannot raise. Never walk third_party (huge).
    for cat in ["kernels", "common", "ascendc"]:
        cat_dir = csrc / cat
        if not cat_dir.is_dir():
            continue
        for d in sorted(cat_dir.iterdir()):
            if d.is_dir() and d.name == op:
                return d
    for cat_dir in sorted(c for c in csrc.iterdir() if c.is_dir()):
        if cat_dir.name == "third_party":
            continue
        if cat_dir.name == op:
            return cat_dir
        for sub in sorted(cat_dir.glob(f"*/{op}")):
            if sub.is_dir():
                return sub
    return None


def workspace_digest(repo: Path, pathspecs: list[str], exclude_prefixes: list[str] | None = None) -> str:
    """Content hash of the *working tree* files under pathspecs.

    Working tree, not the git index: `git ls-files -s` only sees the index, so
    uncommitted edits would be invisible. The build consumes working tree
    files (src_copy does `cp -rf`), so hash what will actually be compiled:
    tracked files (missing ones counted as DELETED) plus untracked ones.
    """
    if exclude_prefixes is None:
        exclude_prefixes = []
    h = hashlib.sha256()
    files: set[str] = set()
    for spec in pathspecs:
        files.update(
            ln
            for ln in run_git(repo, "ls-files", "--", spec).splitlines()
            if not any(ln.startswith(p) for p in exclude_prefixes)
        )
        files.update(
            ln
            for ln in run_git(repo, "ls-files", "--others", "--exclude-standard", "--", spec).splitlines()
            if not any(ln.startswith(p) for p in exclude_prefixes)
        )
    for path in sorted(files):
        f = repo / path
        h.update(path.encode())
        if f.is_file():
            h.update(f.read_bytes())
        else:
            h.update(b"<DELETED>")
    return h.hexdigest()


def build_dir(repo: Path) -> Path:
    return repo / "csrc" / "build"


def binary_root(repo: Path, soc_version: str) -> Path:
    return build_dir(repo) / "binary" / soc_dir_for(soc_version)


def collect_variants(repo: Path, op: str, soc_version: str) -> list[str]:
    """Variant artifacts of one op: file names under binary/<soc>/bin/<op>/."""
    bin_dir = binary_root(repo, soc_version) / "bin" / op
    if not bin_dir.is_dir():
        return []
    return sorted(
        p.name for p in bin_dir.iterdir() if p.suffix in (".o", ".json") and not p.name.endswith("_relocatable.json")
    )


def camel_prefix_from_sh(repo: Path, op: str, soc_version: str) -> str | None:
    """Derive the CamelCase op prefix (HcPost) from gen/<Op>-<op>-N.sh."""
    gen = binary_root(repo, soc_version) / "gen"
    if not gen.is_dir():
        return None
    for p in sorted(gen.glob(f"*-{op}-*.sh")):
        m = re.match(r"([A-Za-z0-9]+)-" + re.escape(op) + r"-\d+\.sh$", p.name)
        if m:
            return m.group(1)
    return None


def remove_quiet(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def invalidate_op(repo: Path, op: str, soc_version: str) -> dict:
    """Remove every stale artifact of one op; .done is the ninja switch.

    Returns a removal count per artifact class so the log shows exactly
    what was deleted (an audit trail for wrong-invalidation debugging).
    """
    root = binary_root(repo, soc_version)
    stats = {"done": 0, "sh": 0, "param": 0, "bin_files": 0, "src": 0}
    # Resolve the CamelCase prefix BEFORE deleting the .sh files: the prefix
    # is derived from gen/<Op>-<op>-N.sh names.
    prefix = camel_prefix_from_sh(repo, op, soc_version)
    # Kernel binaries + metadata.
    bin_dir = root / "bin" / op
    if bin_dir.is_dir():
        stats["bin_files"] = sum(1 for _ in bin_dir.rglob("*") if _.is_file())
        remove_quiet(bin_dir)
    # .done switches: <op>_<soc>_<N>.done
    for p in (root / "gen").glob(f"{op}_{soc_dir_for(soc_version)}_*.done"):
        p.unlink()
        stats["done"] += 1
    # Variant compile scripts: <Op>-<op>-N.sh
    for p in (root / "gen").glob(f"*-{op}-*.sh"):
        p.unlink()
        stats["sh"] += 1
    # Variant param definitions: <Op>_<hash>_param.json (+ derived params)
    if prefix:
        for p in (root / "gen").glob(f"{prefix}_*_param.json"):
            p.unlink()
            stats["param"] += 1
    # Stale src_copy target side (cp -rf does not delete removed files).
    src_dir = root / "src" / op
    if src_dir.is_dir():
        stats["src"] = sum(1 for _ in src_dir.rglob("*") if _.is_file())
        remove_quiet(src_dir)
    return stats


def load_manifest(repo: Path) -> dict | None:
    path = build_dir(repo) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != MANIFEST_SCHEMA:
            return None
        return manifest
    except (json.JSONDecodeError, OSError):
        return None


def cmd_generate(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    soc = args.soc_version
    t0 = time.monotonic()
    print(f"[generate] repo={repo} soc={soc} image_tag={args.image_tag or '<none>'}", flush=True)
    ops = parse_custom_ops(repo, soc)
    op_dirs, entries, skipped = {}, {}, []
    for op in ops:
        d = resolve_op_dir(repo, op)
        if d is None:
            skipped.append(op)
            continue
        op_dirs[op] = d
        entries[op] = {
            "hash": workspace_digest(repo, [d.relative_to(repo).as_posix()]),
            "variants": collect_variants(repo, op, soc),
        }
    if skipped:
        print(
            f"::warning::{len(skipped)} ops have no source directory and are not cached: {', '.join(sorted(skipped))}",
            flush=True,
        )
    # Global inputs exclude the UNION of all SOC branches' op directories:
    # an op this SOC does not compile cannot affect its artifacts (see
    # global_exclude_prefixes).
    global_hash = (
        workspace_digest(repo, GLOBAL_PATHSPEC, exclude_prefixes=global_exclude_prefixes(repo))
        + hashlib.sha256(f"image={args.image_tag}|soc={soc}".encode()).hexdigest()
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "global_hash": global_hash,
        "soc_version": soc,
        "ops": entries,
    }
    out = build_dir(repo) / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_variants = sum(len(e["variants"]) for e in entries.values())
    print(f"[generate] manifest written: {out}", flush=True)
    print(
        f"[generate] ops: {len(entries)}/{len(ops)} resolved, "
        f"{total_variants} variant artifacts, global_hash={global_hash[:12]}, "
        f"took {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    for op, e in sorted(entries.items()):
        print(f"  {op}: {e['hash'][:12]} ({len(e['variants'])} variant files)", flush=True)
    return 0


def cmd_invalidate(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    soc = args.soc_version
    t0 = time.monotonic()
    print(
        f"[invalidate] repo={repo} soc={soc} image_tag={args.image_tag or '<none>'} dry_run={args.dry_run}", flush=True
    )
    manifest = load_manifest(repo)
    if manifest is None:
        print(
            f"::error::manifest missing or unreadable at "
            f"{build_dir(repo) / MANIFEST_NAME} -> full rebuild (fail closed)",
            flush=True,
        )
        return 2
    print(
        f"[invalidate] manifest loaded: schema={manifest.get('schema')} "
        f"ops={len(manifest.get('ops', {}))} "
        f"soc_version={manifest.get('soc_version')}",
        flush=True,
    )

    current_global = (
        workspace_digest(repo, GLOBAL_PATHSPEC, exclude_prefixes=global_exclude_prefixes(repo))
        + hashlib.sha256(f"image={args.image_tag}|soc={soc}".encode()).hexdigest()
    )
    snap_global = str(manifest.get("global_hash", ""))
    if snap_global != current_global:
        print(
            f"[invalidate] global inputs changed -> full rebuild\n"
            f"  snapshot global_hash={snap_global[:12]}\n"
            f"  current  global_hash={current_global[:12]}",
            flush=True,
        )
        return 2

    stale, fresh, added, orphan = [], [], [], []
    for op in parse_custom_ops(repo, soc):
        entry = manifest["ops"].get(op)
        d = resolve_op_dir(repo, op)
        if d is None:
            orphan.append(op)
            continue
        if entry is None:
            added.append(op)
            continue
        current = workspace_digest(repo, [d.relative_to(repo).as_posix()])
        if entry["hash"] != current:
            stale.append(op)
            print(f"  [miss] {op}: source hash {entry['hash'][:12]} -> {current[:12]}", flush=True)
        elif sorted(entry["variants"]) != collect_variants(repo, op, soc):
            # Variant set drifted without an op hash change: treat as stale.
            stale.append(op)
            print(
                f"  [miss] {op}: variant set drift "
                f"({len(entry['variants'])} -> "
                f"{len(collect_variants(repo, op, soc))} files)",
                flush=True,
            )
        else:
            fresh.append(op)

    miss = stale + added + orphan
    for op in sorted(miss):
        if op in added:
            print(f"  [miss] {op}: new op, not in manifest", flush=True)
        elif op in orphan:
            print(f"  [miss] {op}: source directory removed (orphan cleanup)", flush=True)
        if not args.dry_run:
            stats = invalidate_op(repo, op, soc)
            print(
                f"  invalidated {op}: -{stats['done']} .done, "
                f"-{stats['sh']} .sh, -{stats['param']} param.json, "
                f"-{stats['bin_files']} bin files, -{stats['src']} src files",
                flush=True,
            )

    print(
        f"[invalidate] ops: {len(fresh)} hit / {len(stale)} changed / "
        f"{len(added)} new / {len(orphan)} orphan "
        f"(took {time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    print(f"hit_list={json.dumps(sorted(fresh))}", flush=True)
    print(f"miss_list={json.dumps(sorted(miss))}", flush=True)
    if args.dry_run and miss:
        print("dry-run: no files were removed", flush=True)
    return 0 if not miss else 1


def cmd_check_freshness(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    soc = args.soc_version
    ref = args.since_epoch
    if not args.ops:
        print(
            "::error::--check-freshness requires a non-empty --ops list; an empty check would silently pass", flush=True
        )
        return 3
    bad = []
    for op in args.ops:
        bin_dir = binary_root(repo, soc) / "bin" / op
        # Only compiled objects count: the trailing ops_config pass rewrites
        # every *.json metadata file on every build, so json mtimes say
        # nothing about whether the kernels were recompiled.
        objs = [p for p in bin_dir.glob("*.o")] if bin_dir.is_dir() else []
        if not objs:
            bad.append(f"{op}: no object artifacts under {bin_dir}")
            continue
        newest = max(p.stat().st_mtime for p in objs)
        if newest <= ref:
            bad.append(f"{op}: artifacts not rebuilt (newest .o mtime {newest:.0f} <= ref {ref:.0f})")
        else:
            print(f"  {op}: fresh (newest .o at {newest:.0f}, ref {ref:.0f})", flush=True)
    if bad:
        for b in bad:
            print(f"::error::{b}", flush=True)
        return 1
    print(f"freshness OK for {len(args.ops)} ops", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="vllm-ascend repo root")
    ap.add_argument("--soc-version", default="ascend910b1")
    ap.add_argument("--image-tag", default="", help="CANN image tag, mixed into the global hash")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--generate", action="store_true")
    sub.add_argument("--invalidate", action="store_true")
    sub.add_argument("--check-freshness", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="with --invalidate: report without deleting")
    ap.add_argument("--ops", nargs="*", default=[], help="with --check-freshness: ops expected rebuilt")
    ap.add_argument(
        "--since-epoch", type=float, default=time.time() - 3600, help="with --check-freshness: reference timestamp"
    )
    args = ap.parse_args()
    import traceback

    try:
        if args.generate:
            return cmd_generate(args)
        if args.invalidate:
            return cmd_invalidate(args)
        return cmd_check_freshness(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level CI error surface
        print(f"::error::csrc_snapshot_manifest failed unexpectedly: {exc!r}", flush=True)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
