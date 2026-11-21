# SPDX-License-Identifier: Apache-2.0
"""NPU quantized Down numerical and graph tests for front padding."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertParallelMode


def _shared():
    shared = AscendSharedExperts.__new__(AscendSharedExperts)
    shared.layer = SimpleNamespace(expert_gate=None)
    shared.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=16))
    return shared


def _down(x, scale, weight, weight_scale):
    return torch_npu.npu_quant_matmul(
        x, weight, weight_scale, pertoken_scale=scale, bias=None, output_dtype=torch.bfloat16
    )


def _inputs(rows):
    x = torch.randint(-127, 128, (rows, 384), dtype=torch.int8, device="npu")
    scale = torch.rand(rows, dtype=torch.float32, device="npu") / 128
    weight = torch.randint(-127, 128, (384, 7168), dtype=torch.int8, device="npu")
    weight = torch_npu.npu_format_cast(weight, 29)  # FRACTAL_NZ, as on the deployed path.
    weight_scale = torch.rand(7168, dtype=torch.bfloat16, device="npu") / 128
    return x, scale, weight, weight_scale


@pytest.mark.parametrize("rows", [1, 8, 15, 16, 17, 31, 32, 64])
def test_down_input_padding_matches_output_padding(rows):
    torch.manual_seed(2026 + rows)
    x, scale, weight, weight_scale = _inputs(rows)
    pad_rows = (-rows) % 16
    reference = _down(x, scale, weight, weight_scale)
    if pad_rows:
        reference = F.pad(reference, (0, 0, 0, pad_rows))
    padded_x, padded_scale = _shared()._pad_a8_down_input(x, scale, SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY)
    actual = _down(padded_x, padded_scale, weight, weight_scale)
    torch.testing.assert_close(actual.cpu(), reference.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("warm", [False, True])
def test_down_input_padding_graph_replay_changes_inputs(monkeypatch, warm):
    cache = {}
    monkeypatch.setattr(moe_utils, "_PAD_ZERO_BLOCKS", cache)
    x, scale, weight, weight_scale = _inputs(8)
    shared = _shared()
    mode = SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    if warm:
        shared._pad_a8_down_input(x, scale, mode)
    else:
        # Warm F.pad without populating the candidate cache.
        F.pad(x, (0, 0, 0, 8))
        F.pad(scale, (0, 8))
    _down(x, scale, weight, weight_scale)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        padded_x, padded_scale = shared._pad_a8_down_input(x, scale, mode)
        actual = _down(padded_x, padded_scale, weight, weight_scale)
    assert bool(cache) is warm
    for iteration in range(3):
        x.fill_(iteration + 1)
        scale.fill_(0.01 * (iteration + 1))
        torch.npu.synchronize()
        graph.replay()
        reference = F.pad(_down(x, scale, weight, weight_scale), (0, 0, 0, 8))
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference.cpu(), atol=0, rtol=0)
