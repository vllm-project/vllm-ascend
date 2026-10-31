#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Identity and integrity checks for nightly L0 artifacts in GitHub Cache."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_csrc_l1_restore import _tracked_csrc_hash

MANIFEST = ".nightly-csrc-l0.json"
ARTIFACTS = ("_cann_ops_custom", "lib", "include")


def package_identity(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if not record:
        raise ValueError(f"No installed artifact record for {name}")
    return {"version": distribution.version, "record": hashlib.sha256(record.encode()).hexdigest()}


def cache_key(source: Path, architecture: str, soc: str) -> str:
    # The same runtime identity as L1, with the actual Python/Torch ABI added.
    # A mutable nightly tag alone does not identify the installed toolchain.
    source_hash = _tracked_csrc_hash(source)
    if source_hash == "adhoc":
        raise ValueError("No tracked csrc inputs")
    response = subprocess.run(
        [
            sys.executable,
            str(source / "csrc/scripts/build_cache.py"),
            "snapshot-key",
            "--architecture",
            architecture,
            "--soc-version",
            soc,
            "--csrc-hash",
            source_hash,
            "--unique-suffix",
            "l0",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(response) != 3 or not all(response):
        raise ValueError("Invalid csrc runtime identity")
    identity = {
        "runtime_source": response[1],
        "python": sys.implementation.cache_tag,
        "packages": {name: package_identity(name) for name in ("torch", "torch-npu")},
        "build_config": {
            name: fingerprint(source / name)
            for name in (
                "pyproject.toml",
                "vllm_ascend/envs.py",
                "packages.txt",
                "requirements.txt",
                "requirements-dev.txt",
                "requirements-nightly.txt",
            )
            if (source / name).is_file()
        },
        "compilers": {
            name: subprocess.run([name, "--version"], check=True, capture_output=True, text=True).stdout
            for name in ("cc", "c++", "cmake")
        },
        "flags": {
            name: os.environ.get(name, "")
            for name in ("CFLAGS", "CXXFLAGS", "CMAKE_BUILD_TYPE", "VLLM_ASCEND_ENABLE_BATCH_MEMCPY")
        },
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"vllm-ascend-nightly-build-v1-{digest}"


def artifact_paths(root: Path) -> list[Path]:
    package = root / "vllm_ascend"
    paths = list(package.glob("*.so"))
    for name in ARTIFACTS:
        paths.extend(path for path in (package / name).rglob("*") if path.is_file())
    return sorted(paths)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, key: str) -> None:
    if not any(path.stat().st_size for path in (root / "vllm_ascend").glob("vllm_ascend_C*.so")):
        raise ValueError("Missing native extension")
    files = {path.relative_to(root).as_posix(): fingerprint(path) for path in artifact_paths(root)}
    (root / MANIFEST).write_text(json.dumps({"key": key, "files": files}, sort_keys=True), encoding="utf-8")


def validate_and_copy(root: Path, source: Path, key: str) -> bool:
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        paths = artifact_paths(root)
        actual = {path.relative_to(root).as_posix(): fingerprint(path) for path in paths}
        if manifest["key"] != key or not actual or actual != manifest["files"]:
            raise ValueError("L0 identity or artifact checksum mismatch")
        if not any(path.stat().st_size for path in (root / "vllm_ascend").glob("vllm_ascend_C*.so")):
            raise ValueError("Missing native extension")
        if source.resolve() != root.resolve():
            for path in paths:
                destination = source / path.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"::notice::Reject nightly L0 ({exc}); compile from source.")
        # Cached outputs must not leak into the producer's fallback build.
        for path in artifact_paths(root):
            path.unlink(missing_ok=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("key", "record", "restore"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--architecture", default="")
    parser.add_argument("--soc-version", default="")
    parser.add_argument("--key", default="")
    args = parser.parse_args()
    values = {}
    if args.operation == "key":
        try:
            values["key"] = cache_key(args.source_root.resolve(), args.architecture, args.soc_version)
        except (OSError, ValueError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as exc:
            print(f"::notice::Nightly L0 unavailable ({exc}); retain source-build fallback.")
    elif args.operation == "record":
        write_manifest(Path.cwd(), args.key)
    else:
        values["cache-hit"] = str(validate_and_copy(Path.cwd(), args.source_root, args.key)).lower()
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for name, value in values.items():
                print(f"{name}={value}", file=output)


if __name__ == "__main__":
    main()
