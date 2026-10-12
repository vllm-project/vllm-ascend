# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one singlecard pytest target with exactly one visible Ascend device."""

import importlib
import os
import sys


def single_device_mask(value: str | None) -> str:
    """Narrow an existing assignment, or use the runner's logical device zero."""
    if value is None:
        return "0"
    devices = value.split(",")
    if not devices or any(not device.strip().isdigit() for device in devices):
        raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES assignment: {value!r}")
    return devices[0].strip()


def main() -> int:
    # This must happen before importing torch, torch_npu, vLLM, or conftest.
    # Never replace an existing assignment with a different physical card.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = single_device_mask(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    import torch
    import torch_npu  # noqa: F401

    count = torch.npu.device_count()
    if count != 1:
        raise RuntimeError(f"Singlecard CI requires exactly one visible NPU, got {count}")
    torch.npu.set_device(0)
    print(f"Singlecard CI: {torch.npu.get_device_name(0)}, visible devices={count}", flush=True)

    # A5's global enable_custom_op() is intentionally disabled. Operator tests
    # need the installed local binding on every chip, including A5.
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")

    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
