#!/usr/bin/env python3
"""
Generate the ccache build matrix for schedule_cache_csrc_build.yaml.

Reads Dockerfile list and release branches from
schedule_image_build_and_push.yaml, then reads each Dockerfile's FROM line
to extract the CANN image tag.  Matches each tag against runner_label.json
to discover which CI runners use that image, deduplicating by architecture.
Produces a JSON matrix consumed by the build-cache job.

Usage:
    IMAGE_REGISTRY=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann \
    python3 generate_cache_matrix.py
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).parent
_RUNNER_LABEL_PATH = _SCRIPT_DIR / "runner_label.json"


@dataclass
class _RunnerInfo:
    """A runner entry from runner_label.json."""
    label: str
    chip: str
    npu_num: int
    image_tag: str


def _derive_arch(label: str) -> str:
    """Derive architecture (ARM64 / X64) from a runner label.

    Mirrors the consumer-side get_arch step in _selected_tests.yaml:
        uname -m | case aarch64→ARM64, x86_64→X64.
    """
    if "aarch64" in label:
        return "ARM64"
    if "amd64" in label or "x86_64" in label:
        return "X64"
    return "UNKNOWN"


def _load_runners() -> list[_RunnerInfo]:
    """Load all runners from runner_label.json."""
    try:
        with open(_RUNNER_LABEL_PATH) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        warn(f"Cannot load {_RUNNER_LABEL_PATH}: {e}")
        return []
    return [
        _RunnerInfo(
            label=label,
            chip=info["chip"],
            npu_num=info["npu_num"],
            image_tag=info.get("image_tag", ""),
        )
        for label, info in raw.items()
    ]


def _build_runner_index(
    runners: list[_RunnerInfo],
) -> dict[str, dict[str, _RunnerInfo]]:
    """Build an image_tag → {arch: RunnerInfo} index.

    Deduplicates by arch so that multiple runners sharing the same image_tag
    (e.g. linux-aarch64-a2b3-1/2/4 all use 9.0.0-910b-ubuntu22.04-py3.12)
    produce only one matrix entry per architecture.
    """
    idx: dict[str, dict[str, _RunnerInfo]] = {}
    for r in runners:
        if not r.image_tag:
            continue
        arch = _derive_arch(r.label)
        idx.setdefault(r.image_tag, {})[arch] = r
    return idx


def warn(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


def normalize_branch(branch: str) -> str:
    """Normalize a branch name for cache-key use.

    Mirrors the consumer-side logic in _selected_tests.yaml:
        tr '/:@ ' '----' | tr -cd 'A-Za-z0-9._-'
    Characters ``/``, ``:``, ``@``, `` `` are each replaced by a single
    ``-``.  Any remaining character outside ``[A-Za-z0-9._-]`` is
    removed.  An empty string after filtering falls back to
    ``"unknown-branch"`` (matching the consumer's fallback).
    """
    for ch in "/:@ ":
        branch = branch.replace(ch, "-")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    branch = "".join(c for c in branch if c in allowed)
    return branch or "unknown-branch"


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
    df_path = str(REPO_ROOT / df)
    try:
        out = subprocess.check_output(["grep", "-m1", "^FROM", df_path], stderr=subprocess.DEVNULL).decode().strip()
        return out.split(":")[-1]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ninja cache build matrix")
    parser.add_argument(
        "--branch-filter",
        metavar="BRANCH",
        help="Only include entries whose branch_ref or normalized branch "
        "matches this value (e.g. 'main').  When omitted, all "
        "branches from the source config are included.",
    )
    args = parser.parse_args()

    image_registry = os.environ.get(
        "IMAGE_REGISTRY",
        "swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann",
    )

    # ── 1. discover Dockerfiles ──────────────────────────────────────────
    # Primary source: schedule_image_build_and_push.yaml
    dockerfiles: dict[str, bool] = {}
    config_path = REPO_ROOT / ".github/workflows/schedule_image_build_and_push.yaml"
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
        os.chdir(str(REPO_ROOT))
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

    # ── branch filter (--branch-filter) ─────────────────────────────────
    if args.branch_filter:
        raw_filter = args.branch_filter
        normalized_filter = normalize_branch(raw_filter)
        before = list(branches)
        branches = [b for b in branches if b == raw_filter or normalize_branch(b) == normalized_filter]
        if not branches:
            print(
                f"Error: --branch-filter '{raw_filter}' matched no branches. Candidates: {sorted(before)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── 3. load runner catalogue ─────────────────────────────────────────
    runners = _load_runners()
    if not runners:
        warn(
            f"No runners loaded from {_RUNNER_LABEL_PATH}; "
            "matrix will be empty."
        )

    runner_index = _build_runner_index(runners)

    # ── 4. build matrix ──────────────────────────────────────────────────
    include: list[dict] = []
    for branch in branches:
        for df in sorted(dockerfiles.keys()):
            tag = image_tag(df)
            if tag is None:
                warn(f"Skipping {df}: cannot read FROM line")
                continue

            tag_runners = runner_index.get(tag, {})
            if not tag_runners:
                warn(
                    f"Skipping {df}: no runner in {_RUNNER_LABEL_PATH.name} "
                    f"uses image tag {tag}"
                )
                continue

            for arch, runner_info in tag_runners.items():
                include.append(
                    {
                        "branch": branch,
                        "branch_normalized": normalize_branch(branch),
                        "dockerfile": df,
                        "image": f"{image_registry}:{tag}",
                        "image_tag": tag,
                        "soc_version": soc_version(df),
                        "os_type": os_type(df),
                        "arch": arch,
                        "runner": runner_info.label,
                    }
                )

    if not include:
        warn("Matrix is empty — no cache will be produced this run.")

    matrix = {"include": include}
    # Compact JSON — single line for GitHub Actions $GITHUB_OUTPUT compatibility.
    # Use | python3 -m json.tool for pretty-printing when running locally.
    print(json.dumps(matrix, separators=(",", ":")))
    unique_runners = len({e["runner"] for e in include})
    print(
        f"\nGenerated {len(include)} matrix entries "
        f"({len(branches)} branches x {len(dockerfiles)} Dockerfiles -> "
        f"{unique_runners} runners)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
