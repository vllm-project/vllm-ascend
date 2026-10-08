# SPDX-License-Identifier: Apache-2.0
"""Runtime dimensions/strides retain generic local-merge behavior."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sfa_dcp_merge import fused_merge  # noqa: E402


@pytest.mark.parametrize("ranks", [2, 8])
@pytest.mark.parametrize("token_dim", [1, 2])
@pytest.mark.parametrize("head_dim", [127, 496, 512, 513])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_runtime_merge_dimensions_and_changed_graph_inputs(ranks, token_dim, head_dim, dtype):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(16350)
    values = torch.randn(ranks, 3, 3, head_dim, generator=generator).to(dtype)
    lses = torch.randn(ranks, 3, 3, generator=generator)
    device_values, device_lses = values.npu(), lses.npu()
    for _ in range(3):
        fused_merge(device_values, device_lses, token_dim)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = fused_merge(device_values, device_lses, token_dim)
    for step in range(3):
        current = (values.float() + step / 16).to(dtype)
        stats = lses + torch.arange(ranks).view(ranks, 1, 1) * step
        stats[:, 0, 0] = -torch.inf
        current[:, 0, 0] = torch.nan
        device_values.copy_(current)
        device_lses.copy_(stats)
        graph.replay()
        valid = torch.isfinite(stats)
        weights = torch.nan_to_num(torch.softmax(stats.double().masked_fill(~valid, -torch.inf), dim=0), nan=0.0)
        terms = current.double().masked_fill(~valid.unsqueeze(-1), 0.0) * weights.unsqueeze(-1)
        expected = terms.sum(0).movedim(token_dim - 1, 0)
        budget = (1e-6 + 8 * torch.finfo(torch.float32).eps * terms.abs().sum(0)).movedim(token_dim - 1, 0)
        actual = result.cpu().double()
        assert torch.isfinite(actual).all()
        assert ((actual - expected).abs() <= budget).all()
        assert torch.count_nonzero(actual[0, 0]) == 0
    torch.npu.synchronize()
