#!/usr/bin/env python
"""Export torch-npu service profiles into text reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch_npu.profiler.profiler import analyse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Profile roots containing *_ascend_pt directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = sorted(profile for root in args.roots for profile in root.rglob("*_ascend_pt") if profile.is_dir())
    if not profiles:
        raise SystemExit("No *_ascend_pt directories found")

    for profile in profiles:
        print(f"analyse: {profile}")
        analyse(str(profile))


if __name__ == "__main__":
    main()
