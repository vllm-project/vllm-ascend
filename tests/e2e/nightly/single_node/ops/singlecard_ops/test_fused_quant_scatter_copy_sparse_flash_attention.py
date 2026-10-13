# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal functional test for FusedQuantScatterCopySparseFlashAttention."""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

from .fused_quant_scatter_copy_sparse_flash_attention_golden import (  # noqa: E402
    MAX_CACHE_TOKENS,
    NOPE_DIM,
    TOPK,
    make_test_data,
)


@pytest.fixture(scope="module", autouse=True)
def load_a5_custom_operator() -> None:
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,PLC0415

    assert hasattr(torch.ops._C_ascend, "npu_fused_quant_scatter_copy_sparse_flash_attention")


def swapped_from_cpu(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required for the DRAM input")
    swapped = torch_npu.empty_with_swapped_memory(cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device)
    swapped.zero_()
    swapped.add_(cpu_tensor.to(device))
    torch.npu.synchronize()
    return swapped


def check_attention(actual: torch.Tensor, golden: torch.Tensor) -> None:
    actual = actual.detach().cpu().float()
    golden = golden.float()
    if actual.shape != golden.shape or not bool(torch.isfinite(actual).all()):
        raise AssertionError("attention output has an invalid shape or non-finite value")
    relative_l2 = (actual - golden).flatten(1).norm(dim=1) / golden.flatten(1).norm(dim=1).clamp_min(1e-7)
    if not torch.allclose(actual, golden, atol=0.008, rtol=0.05) or float(relative_l2.max()) > 0.08:
        raise AssertionError(f"attention output differs from CPU golden, max_rel_l2={float(relative_l2.max()):.6f}")


@pytest.mark.parametrize(
    ("miss_count", "cache_tokens"),
    [(0, TOPK), (8, TOPK), (513, TOPK), (8, MAX_CACHE_TOKENS), (513, MAX_CACHE_TOKENS)],
    ids=[
        "non_offload",
        "fused_miss",
        "union_copy_fallback",
        "high_slot_fused_miss",
        "high_slot_union_copy_fallback",
    ],
)
def test_fused_quant_scatter_copy_sparse_flash_attention(miss_count: int, cache_tokens: int) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    op = torch.ops._C_ascend.npu_fused_quant_scatter_copy_sparse_flash_attention.default

    data = make_test_data(miss_count, cache_tokens)
    hbm_kv_bytes = data.initial_hbm.to(device)
    dram_kv_bytes = swapped_from_cpu(data.dram, device)
    inputs = [
        data.query,
        data.hbm_block_table,
        data.dram_block_table,
        data.topk_dst_slots,
        data.topk_src_ids,
        data.topk_miss_counts,
        data.actual_seq_lengths_query,
        data.actual_seq_lengths_kv,
        data.num_cache_tokens,
        data.miss_src_ids,
        data.miss_dst_slots,
        data.miss_counts,
    ]
    device_inputs = [tensor.to(device) for tensor in inputs]

    attention_out = torch.empty((*data.query.shape[:-1], NOPE_DIM), dtype=data.query.dtype, device=device)
    op(
        device_inputs[0],
        hbm_kv_bytes,
        dram_kv_bytes,
        *device_inputs[1:],
        float(data.scale_value),
        attention_out,
    )
    torch.npu.synchronize()
    if not torch.equal(hbm_kv_bytes.cpu(), data.expected_hbm):
        raise AssertionError("hbm_kv_bytes was not updated in place")
    check_attention(attention_out, data.attention_golden)
