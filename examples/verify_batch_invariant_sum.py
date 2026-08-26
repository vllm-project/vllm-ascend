# SPDX-License-Identifier: Apache-2.0
"""Verify which ATen overload implements sum and its NPU replacement.

Run this script in an environment where vllm-ascend and batch_invariant_ops
were built from the source tree being tested.
"""

import argparse
import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.batch_invariant import enable_batch_invariant_mode


class SumOpTrace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_name = str(func)
        if "sum" in op_name or "batch_invariant" in op_name:
            print(f"  dispatch: {op_name}")
        return func(*args, **(kwargs or {}))


def run_and_check(name: str, operation, expected: torch.Tensor) -> None:
    print(f"\n{name}")
    with SumOpTrace():
        result = operation()

    torch.testing.assert_close(result, expected)
    print(f"  shape: {tuple(result.shape)}")
    print("  result matches npu_reduce_sum_batch_invariant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    print("Available aten::sum overloads:")
    print(torch.ops.aten.sum.overloads())
    print("\nDefault schema:")
    print(torch.ops.aten.sum.default._schema)
    print("\nDimensioned schema:")
    print(torch.ops.aten.sum.dim_IntList._schema)

    enable_batch_invariant_mode()

    x = torch.arange(12, dtype=torch.float32, device=args.device).reshape(3, 4)
    expected_dim = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, -1, True)
    expected_all = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, 1, False)
    expected_all = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(expected_all, 0, False)
    x_fp16 = x.to(torch.float16)
    expected_dtype = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x_fp16.to(torch.float32), -1, True)

    run_and_check(
        "Tensor.sum(dim=-1, keepdim=True)",
        lambda: x.sum(dim=-1, keepdim=True),
        expected_dim,
    )
    run_and_check(
        "torch.sum(dim=-1, keepdim=True)",
        lambda: torch.sum(x, dim=-1, keepdim=True),
        expected_dim,
    )
    run_and_check(
        "Tensor.sum(dim=-1, keepdim=True, dtype=float32)",
        lambda: x_fp16.sum(dim=-1, keepdim=True, dtype=torch.float32),
        expected_dtype,
    )
    run_and_check("Tensor.sum()", lambda: x.sum(), expected_all)


if __name__ == "__main__":
    main()
