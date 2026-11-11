# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc

import torch


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _assert_close(name, actual, expected, rtol=5e-2, atol=5e-2):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol, msg=name)


def _gate_cumsum_reference(g, chunk_size, cu_seqlens=None):
    g_cpu = g.detach().cpu().to(torch.float32)
    ref = torch.zeros_like(g_cpu)
    rcp_ln2 = 1.4426950408889634
    if cu_seqlens is None:
        cu_seqlens = [0, g_cpu.shape[1]]
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            ref[:, chunk_start:chunk_end] = torch.cumsum(g_cpu[:, chunk_start:chunk_end] * rcp_ln2, dim=1)
    return ref


def test_kda_gate_cumsum_vllm_output_metadata():
    raw_gate = torch.zeros((1, 64, 2, 128), device="npu", dtype=torch.bfloat16)
    gk = torch.ops._C_ascend.kda_gate_cumsum_vllm(raw_gate, 64, layout="BSND")
    assert gk.shape == raw_gate.shape
    assert gk.dtype == torch.float32


@torch.inference_mode()
def test_kda_gate_cumsum_vllm_matches_reference():
    torch.manual_seed(20260720)

    g = (-torch.rand(1, 96, 2, 128, dtype=torch.float32) * 0.05).npu()
    cu_seqlens = [0, 31, 96]
    out = torch.ops._C_ascend.kda_gate_cumsum_vllm(g, 64, cu_seqlens=cu_seqlens, layout="BSND")
    ref = _gate_cumsum_reference(g, 64, cu_seqlens)

    assert torch.isfinite(out).all().item()
    _assert_close("gk", out, ref, rtol=2e-3, atol=2e-3)
    _cleanup_npu()
