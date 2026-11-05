#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_EXCLUDES = [
    "libplatform.so",
    "libregister.so",
    "libge_common_base.so",
    "libc10.so",
    "libc_sec.so",
    "libnnopbase.so",
    "libprofapi.so",
    "libgraph_base.so",
    "libgraph.so",
    "libexe_graph.so",
    "libascend*.so",
    "libAscend*.so",
    "libtorch*.so",
    "libopapi*.so",
    "libops*.so",
    "liboptiling.so",
    "liberror_manager.so",
    "libruntime.so",
    "libmmpa.so",
    "libunified_dlog.so",
    "libstdc++.so",
    "libgcc_s.so",
    "libc.so",
    "libm.so",
    "libdl.so",
    "libpthread.so",
    "librt.so",
]


def _is_lib_available(lib_name: str) -> bool:
    result = subprocess.run(
        ["ldconfig", "-p"],
        capture_output=True,
        text=True,
    )
    return lib_name in result.stdout


def _matches_pattern(lib_name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(lib_name, pattern):
            return True
        if lib_name == pattern:
            return True
    return False


def _extract_wheel_so_files(wheel_path: str) -> list[str]:
    so_files = []
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".so") or ".so." in name:
                so_files.append(name)
    return so_files


def _get_ldd_deps(so_path: str) -> list[str]:
    result = subprocess.run(
        ["ldd", so_path],
        capture_output=True,
        text=True,
    )
    deps = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=>" in line:
            lib_name = line.split()[0]
            if lib_name.endswith(".so") or ".so." in lib_name:
                deps.append(lib_name)
        elif line.endswith(".so") and "not found" in line:
            lib_name = line.split()[0]
            deps.append(lib_name)
    return deps


def _discover_all_deps(wheel_path: str, tmp_dir: str) -> list[str]:
    all_deps: set[str] = set()
    so_files = _extract_wheel_so_files(wheel_path)
    if not so_files:
        return []

    with zipfile.ZipFile(wheel_path, "r") as zf:
        for so_file in so_files:
            try:
                extracted = zf.extract(so_file, tmp_dir)
                deps = _get_ldd_deps(extracted)
                all_deps.update(deps)
            except Exception:
                continue

    result = subprocess.run(
        ["auditwheel", "show", wheel_path],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("INFO:auditwheel.lddtree:Excluding"):
            parts = stripped.split()
            if parts:
                lib_name = parts[-1]
                if lib_name.endswith(".so") or ".so." in lib_name:
                    all_deps.add(lib_name)
        if stripped.startswith("INFO:auditwheel.lddtree:") and "=>" in stripped:
            for part in stripped.split():
                if part.endswith(".so") or ".so." in part:
                    all_deps.add(part)

    return sorted(all_deps)


def repair_wheels(wheel_dir: str) -> int:
    excludes = list(BASE_EXCLUDES)
    extra: list[str] = []

    whl_files = sorted(Path(wheel_dir).glob("*.whl"))
    if not whl_files:
        print(f"No wheel files found in {wheel_dir}", file=sys.stderr)
        return 1

    tmp_dir = Path(wheel_dir) / "_tmp_extract"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for whl in whl_files:
        print(f"Scanning dependencies of {whl.name}...", file=sys.stderr)
        deps = _discover_all_deps(str(whl), str(tmp_dir))
        for dep in deps:
            if _matches_pattern(dep, excludes + extra):
                continue
            if not _is_lib_available(dep):
                print(f"Auto-excluding missing library: {dep}", file=sys.stderr)
                extra.append(dep)
            else:
                print(f"Library available on system: {dep}", file=sys.stderr)

    shutil.rmtree(tmp_dir)

    all_excludes = excludes + extra
    print(f"Base excludes ({len(excludes)}): {excludes}", file=sys.stderr)
    print(f"Auto-discovered missing excludes ({len(extra)}): {extra}", file=sys.stderr)
    print(f"Total excludes ({len(all_excludes)}): {all_excludes}", file=sys.stderr)
    exclude_args: list[str] = []
    for exc in all_excludes:
        exclude_args.extend(["--exclude", exc])

    repaired_dir = Path(wheel_dir) / "repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    for whl in whl_files:
        cmd = [
            "auditwheel",
            "repair",
            str(whl),
            "-w",
            str(repaired_dir),
        ] + exclude_args
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}", file=sys.stderr)
            print(f"STDERR:\n{result.stderr}", file=sys.stderr)
            return result.returncode
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    for whl in repaired_dir.glob("*.whl"):
        dest = Path(wheel_dir) / whl.name
        whl.rename(dest)

    for old_whl in whl_files:
        if old_whl.exists():
            old_whl.unlink()

    repaired_dir.rmdir()

    remaining = sorted(Path(wheel_dir).glob("*.whl"))
    print(f"Repaired wheels: {[w.name for w in remaining]}", file=sys.stderr)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: auto_exclude.py <wheel_dir>", file=sys.stderr)
        return 1
    return repair_wheels(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
