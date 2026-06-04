#!/usr/bin/env python3
"""
Generate the ninja cache build matrix for schedule_cache_csrc_build.yaml.

Reads Dockerfile list and release branches from
schedule_image_build_and_push.yaml, then reads each Dockerfile's FROM line
to extract the CANN image tag.  Produces a JSON matrix consumed by the
build-cache job.

Usage:
    IMAGE_REGISTRY=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann \
    python3 generate_cache_matrix.py
"""

import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


def soc_version(df: str) -> str:
    """Derive Ascend SoC version from Dockerfile name."""
    lower = df.lower()
    if "310p" in lower:
        return "ascend310p1"
    if "a3" in lower:
        return "ascend910_9391"
    if "a5" in lower:
        return "ascend950dt_9582"
    return "ascend910b1"


def os_type(df: str) -> str:
    return "openeuler" if "openeuler" in df.lower() else "ubuntu"


def image_tag(df: str) -> str | None:
    """Read CANN image tag from a Dockerfile's FROM line."""
    try:
        out = subprocess.check_output(["grep", "-m1", "^FROM", df], stderr=subprocess.DEVNULL).decode().strip()
        return out.split(":")[-1]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def main() -> None:
    image_registry = os.environ.get(
        "IMAGE_REGISTRY",
        "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann",
    )

    # ── 1. discover Dockerfiles ──────────────────────────────────────────
    # Primary source: schedule_image_build_and_push.yaml
    dockerfiles: dict[str, bool] = {}
    config_path = Path(".github/workflows/schedule_image_build_and_push.yaml")
    data = None

    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            for job_name, job in (data or {}).get("jobs", {}).items():
                for meta in job.get("strategy", {}).get("matrix", {}).get("build_meta", []):
                    df = meta.get("dockerfile", "")
                    if df:
                        dockerfiles[df] = True
        except (yaml.YAMLError, KeyError, TypeError) as e:
            warn(f"Cannot parse {config_path}: {e}")
    else:
        warn(f"{config_path} not found; falling back to scanning root Dockerfiles.")

    # Fallback: scan root Dockerfiles
    if not dockerfiles:
        warn(
            "No Dockerfiles found in schedule_image_build_and_push.yaml; "
            "falling back to scanning root Dockerfile[suffix] pattern."
        )
        for f in sorted(glob("Dockerfile*")):
            if "buildwheel" not in f and ".github" not in f:
                dockerfiles[f] = True

    if not dockerfiles:
        warn("No Dockerfiles discovered; matrix will be empty.")

    # ── 2. discover branches ─────────────────────────────────────────────
    branches = ["main"]

    if data is not None or (config_path.exists()):
        try:
            if data is None:
                with open(config_path) as f:
                    data = yaml.safe_load(f)
            for job_name, job in (data or {}).get("jobs", {}).items():
                for meta in job.get("strategy", {}).get("matrix", {}).get("branch_meta", []):
                    br = meta.get("branch_ref", "")
                    if br and br not in branches:
                        branches.append(br)
        except Exception as e:
            warn(f"Cannot read release branches from {config_path}: {e}")

    # ── 3. arch → runner mapping ─────────────────────────────────────────
    arch_runners = {
        "X64": "linux-amd64-cpu-16-hk",
        "ARM64": "linux-arm64-cpu-16",
    }

    # ── 4. build matrix ──────────────────────────────────────────────────
    include: list[dict] = []
    for branch in branches:
        for df in sorted(dockerfiles.keys()):
            tag = image_tag(df)
            if tag is None:
                warn(f"Skipping {df}: cannot read FROM line")
                continue
            for arch, runner in arch_runners.items():
                include.append(
                    {
                        "branch": branch,
                        "branch_normalized": branch.replace("/", "-"),
                        "dockerfile": df,
                        "image": f"{image_registry}:{tag}",
                        "image_tag": tag,
                        "soc_version": soc_version(df),
                        "os_type": os_type(df),
                        "arch": arch,
                        "runner": runner,
                    }
                )

    if not include:
        warn("Matrix is empty — no cache will be produced this run.")

    matrix = {"include": include}
    # Compact JSON — single line for GitHub Actions $GITHUB_OUTPUT compatibility.
    # Use | python3 -m json.tool for pretty-printing when running locally.
    print(json.dumps(matrix, separators=(",", ":")))
    print(
        f"\nGenerated {len(include)} matrix entries "
        f"({len(branches)} branches × {len(dockerfiles)} Dockerfiles × {len(arch_runners)} archs)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
