# SPDX-License-Identifier: Apache-2.0
"""Compare fused arithmetic with the installed native merge, including its rounding order."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch_npu


def load_kernel():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/mla_dcp_exchange.py"
    spec = importlib.util.spec_from_file_location("fused_mla_test_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ranks", [2, 8, 16])
@pytest.mark.parametrize("tokens", [1, 8, 12])
@pytest.mark.parametrize("case", ["finite", "mixed_empty", "all_empty", "wide_lse", "positive_inf"])
def test_merge_matches_native_bits(ranks, tokens, case):
    torch.npu.set_device(0)
    module = load_kernel()
    torch.manual_seed(1909)
    rows = tokens * 6
    values = torch.randn(ranks, rows, 512, device="npu", dtype=torch.bfloat16)
    lse = torch.randn(ranks, rows, device="npu", dtype=torch.float32)
    if case == "wide_lse":
        lse.mul_(100)
    if case in ("mixed_empty", "positive_inf"):
        values[::2].zero_()
        lse[::2] = float("inf") if case == "positive_inf" else float("-inf")
    if case == "all_empty":
        values.zero_()
        lse.fill_(float("-inf"))
    recv = torch.empty(ranks, rows * 257, device="npu", dtype=torch.int32)
    recv[:, : rows * 256].copy_(values.view(torch.int32).reshape(ranks, -1))
    recv[:, rows * 256 :].copy_(lse.view(torch.int32))
    result = torch.empty(rows, 512, device="npu", dtype=torch.float32)
    module._merge_packed[(24,)](recv, recv.view(torch.bfloat16), result, rows, ranks, 24, 128, enable_fp_fusion=False)
    expected = torch_npu.npu_attention_update(
        lse.clamp_min(torch.finfo(torch.float32).min).unbind(), values.float().unbind(), 0
    )[0]
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))
    assert torch.isfinite(result).all()
    if case == "all_empty":
        assert torch.count_nonzero(result).item() == 0


@pytest.mark.parametrize("layout", ["contiguous", "output_stride", "lse_stride", "output_offset", "lse_offset"])
def test_wire_bytes_and_stride_fallback(monkeypatch, layout):
    torch.npu.set_device(0)
    module = load_kernel()
    output = torch.randn(1, 96, 512, device="npu", dtype=torch.bfloat16)
    lse = torch.randn(1, 96, 1, device="npu", dtype=torch.float32)
    if layout == "output_stride":
        output = torch.randn(1, 96, 1024, device="npu", dtype=torch.bfloat16)[..., :512]
    if layout == "lse_stride":
        lse = torch.randn(1, 96, 2, device="npu", dtype=torch.float32)[..., :1]
    if layout == "output_offset":
        output = torch.randn(96 * 512 + 1, device="npu", dtype=torch.bfloat16)[1:].view(1, 96, 512)
    if layout == "lse_offset":
        lse = torch.randn(97, device="npu", dtype=torch.float32)[1:].view(1, 96, 1)
    lse[0, 0, 0], lse[0, 1, 0], lse[0, 2, 0] = float("-inf"), float("inf"), float("nan")
    # This test isolates serialization; real HCCL is covered by the distributed benchmark.
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(module.dist, "all_to_all_single", lambda recv, send, group: recv.copy_(send))
    recv, ranks, tokens = module._exchange_buffers(output, lse, None)
    assert (ranks, tokens) == (16, 1)
    for peer in range(ranks):
        values = output[0, peer * 6 : (peer + 1) * 6].clone(memory_format=torch.contiguous_format)
        stats = lse[0, peer * 6 : (peer + 1) * 6].clone(memory_format=torch.contiguous_format)
        assert torch.equal(recv[peer, :1536], values.view(torch.int32).flatten())
        assert torch.equal(recv[peer, 1536:], stats.view(torch.int32).flatten())
