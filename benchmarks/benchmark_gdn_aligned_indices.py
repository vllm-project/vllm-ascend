# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare per-group Torch, upstream multi-group, and NPU SIMD-window gathers.

Run on an NPU: python benchmarks/benchmark_gdn_aligned_indices.py --num-groups 24
Reports synchronized wall time including dispatch; does not measure model TPOT.
"""

import argparse
from time import perf_counter
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.mamba_utils import get_aligned_state_indices_multi_group_kernel

from vllm_ascend.worker.v2.model_states.mamba_hybrid import _compute_aligned_state_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=24)
    parser.add_argument("--num-reqs", type=int, default=8)
    parser.add_argument("--state-slots", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if min(args.num_groups, args.num_reqs, args.state_slots, args.iterations) < 1 or args.warmup < 0:
        parser.error("Counts must be positive and warmup must be non-negative")
    torch.npu.set_device(0)
    columns = 64 + args.state_slots
    tables = torch.arange(args.num_groups * args.num_reqs * columns, dtype=torch.int32, device="npu").view(
        args.num_groups, args.num_reqs, columns
    )
    seq_lens = (torch.arange(args.num_reqs, dtype=torch.int32, device="npu") % 64) * 128 + 1
    spec = MambaSpec(
        block_size=128,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=args.state_slots - 1,
    )
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([bt.data_ptr() for bt in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=tables.stride(1),
        block_size=spec.block_size,
        num_groups=args.num_groups,
        aligned_state_indices=torch.empty(
            (args.num_groups, args.num_reqs, args.state_slots), dtype=torch.int32, device="npu"
        ),
    )

    def per_group():
        return [mamba_get_block_table_tensor(bt, seq_lens, spec, "align") for bt in tables]

    def multi_group():
        return _compute_aligned_state_indices(ctx, seq_lens, args.num_reqs, columns)

    def upstream_multi_group():
        output = ctx.aligned_state_indices
        get_aligned_state_indices_multi_group_kernel[(triton.cdiv(args.num_reqs, 32),)](
            ctx.block_table_ptrs,
            seq_lens,
            output,
            ctx.block_table_stride_req,
            seq_lens.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            args.num_reqs,
            CACHE_BLOCK_SIZE=spec.block_size,
            NUM_GROUPS=args.num_groups,
            BLOCK_GROUPS=triton.next_power_of_2(args.num_groups),
            NUM_STATE_SLOTS=args.state_slots,
            BLOCK_STATE_SLOTS=triton.next_power_of_2(args.state_slots),
            BLOCK_ROWS=32,
            num_warps=1,
        )
        return output

    expected = torch.stack(per_group())
    torch.testing.assert_close(upstream_multi_group(), expected, rtol=0, atol=0)
    torch.testing.assert_close(multi_group(), expected, rtol=0, atol=0)
    for name, run in (
        ("per-group Torch", per_group),
        ("upstream multi-group (v5)", upstream_multi_group),
        ("NPU SIMD-window (v6)", multi_group),
    ):
        for _ in range(args.warmup):
            run()
        torch.npu.synchronize()
        start = perf_counter()
        for _ in range(args.iterations):
            run()
        torch.npu.synchronize()
        elapsed_us = (perf_counter() - start) * 1e6 / args.iterations
        print(f"{name}: {elapsed_us:.2f} us/step ({args.num_groups} groups, {args.num_reqs} requests)")


if __name__ == "__main__":
    main()
