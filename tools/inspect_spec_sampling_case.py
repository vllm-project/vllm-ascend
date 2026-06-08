#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import torch

from vllm_ascend.sample.spec_sampling_poc import load_spec_sampling_case


def _describe(prefix: str, value: Any) -> list[str]:
    lines: list[str] = []
    if torch.is_tensor(value):
        lines.append(f"{prefix}: tensor shape={list(value.shape)} dtype={value.dtype} device={value.device}")
        return lines
    if hasattr(value, "__dataclass_fields__"):
        lines.append(f"{prefix}: {type(value).__name__}")
        for name in value.__dataclass_fields__:
            lines.extend(_describe(f"{prefix}.{name}", getattr(value, name)))
        return lines
    if isinstance(value, list):
        lines.append(f"{prefix}: list len={len(value)}")
        return lines
    if isinstance(value, dict):
        lines.append(f"{prefix}: dict len={len(value)}")
        return lines
    lines.append(f"{prefix}: {type(value).__name__} value={repr(value)[:200]}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one dumped spec-sampling case.")
    parser.add_argument("case_path")
    args = parser.parse_args()

    case = load_spec_sampling_case(args.case_path)
    for key in sorted(case.keys()):
        for line in _describe(key, case[key]):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
