#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Split a pytest-cov data dir into per-test (nodeid) coverage directories.

pytest-cov ``--cov-context=test`` stores one context per pytest nodeid inside
a single coverage database. This script combines parallel fragments, then
writes one ``tests/outputs/<sanitized_nodeid>/covdata/coverage`` per context
so artifact layout is test-case granular.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coverage
from coverage.sqldata import CoverageData


def sanitize_target(target: str) -> str:
    """Convert a pytest target/nodeid to an outputs directory name.

    ``tests/e2e/.../test_sampler.py::test_foo`` ->
    ``tests__e2e__...__test_sampler--test_foo``
    """
    target = target.replace("\\", "/")
    if "::" in target:
        path, _, node = target.partition("::")
        if path.endswith(".py"):
            path = path[:-3]
        target = f"{path}::{node}"
    elif target.endswith(".py"):
        target = target[:-3]
    return target.replace("/", "__").replace("::", "--")


def _combine_parallel(covdata_dir: Path, rcfile: Path) -> Path | None:
    """Combine ``coverage.<pid>`` fragments into ``coverage``. Return path or None."""
    data_file = covdata_dir / "coverage"
    fragments = sorted(covdata_dir.glob("coverage.*"))
    # Ignore already-combined / temp names that are not parallel pid files.
    fragments = [p for p in fragments if p.name != "coverage.combined"]
    if not data_file.exists() and not fragments:
        return None

    cov = coverage.Coverage(data_file=str(data_file), config_file=str(rcfile))
    if fragments:
        try:
            cov.combine(keep=False)
        except coverage.CoverageException as exc:
            print(f"WARNING: coverage combine failed: {exc}", file=sys.stderr)
            # Fall back to reading whatever basename exists.
    if not data_file.exists():
        # combine may have nothing to do if only a non-parallel file exists
        # under a different name; pick the newest fragment.
        if fragments:
            return fragments[0]
        return None
    return data_file


def _copy_context(src_path: Path, context: str, dest_path: Path) -> None:
    src = CoverageData(basename=str(src_path))
    src.read()
    src.set_query_contexts([context])

    if dest_path.exists():
        dest_path.unlink()

    dest = CoverageData(basename=str(dest_path))
    dest.set_context(context)

    lines_map: dict[str, set[int]] = {}
    arcs_map: dict[str, set[tuple[int, int]]] = {}
    for filename in src.measured_files():
        lines = src.lines(filename)
        if lines:
            lines_map[filename] = set(lines)
        arcs = src.arcs(filename)
        if arcs:
            arcs_map[filename] = {tuple(a) for a in arcs}  # type: ignore[misc]

    if lines_map:
        dest.add_lines(lines_map)
    if arcs_map:
        dest.add_arcs(arcs_map)
    dest.write()

    # Record the original nodeid for downstream consumers / debugging.
    (dest_path.parent / "context.txt").write_text(context + "\n", encoding="utf-8")


def split_covdata(covdata_dir: Path, outputs_root: Path, rcfile: Path) -> int:
    covdata_dir = covdata_dir.resolve()
    outputs_root = outputs_root.resolve()
    if not covdata_dir.is_dir():
        print(f"WARNING: covdata dir missing: {covdata_dir}", file=sys.stderr)
        return 0

    data_file = _combine_parallel(covdata_dir, rcfile)
    if data_file is None:
        print(f"WARNING: no coverage data in {covdata_dir}", file=sys.stderr)
        return 0

    src = CoverageData(basename=str(data_file))
    src.read()
    contexts = sorted(c for c in src.measured_contexts() if c)
    if not contexts:
        print(
            f"WARNING: no per-test contexts in {data_file}; leaving staging dir as-is",
            file=sys.stderr,
        )
        return 0

    staging_key = covdata_dir.parent.name
    written_keys: set[str] = set()
    for context in contexts:
        key = sanitize_target(context)
        out_dir = outputs_root / key / "covdata"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "coverage"
        _copy_context(data_file, context, out_file)
        written_keys.add(key)
        print(f"  coverage context -> {key}/covdata/coverage")

    # Drop file/batch-level staging when it is not itself a test-case dir.
    if staging_key not in written_keys:
        staging_parent = covdata_dir.parent
        print(f"  removing staging dir: {staging_parent}")
        shutil.rmtree(staging_parent, ignore_errors=True)
    else:
        # Staging is already a single test-case dir; remove parallel leftovers.
        for leftover in covdata_dir.glob("coverage.*"):
            leftover.unlink(missing_ok=True)

    return len(written_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covdata-dir", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--rcfile", type=Path, required=True)
    args = parser.parse_args()

    print(f"Splitting coverage by test context under {args.covdata_dir}")
    count = split_covdata(args.covdata_dir, args.outputs_root, args.rcfile)
    print(f"Wrote {count} per-test coverage director{'y' if count == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
