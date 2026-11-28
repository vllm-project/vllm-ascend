# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Real NPU tests; never substitute CUDA execution for Ascend evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.models.qwen3_vl import pos_embed_interpolate_native

from vllm_ascend.device.device_config import is_310p
from vllm_ascend.ops.triton.vision_pos_embed import vision_pos_embed_interpolate
from vllm_ascend.patch.worker import patch_qwen3vl


def _compare(actual, expected):
    rtol, atol, max_error = {
        torch.float32: (2**-10, 2**-16, 0.01),
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
    }[expected.dtype]
    error = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert ((error <= atol + rtol * expected.float().abs()).float().mean() >= 0.99).item()
    assert error.max().item() <= max_error
    assert actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device


@pytest.fixture(autouse=True)
def require_npu():
    if not torch.npu.is_available() or is_310p():
        pytest.skip("requires Triton-capable Ascend NPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "grid,side,hidden,merge",
    [
        ((1, 1, 1), 1, 7, 1),
        ((2, 1, 7), 4, 31, 1),
        ((1, 8, 2), 4, 65, 2),
        ((3, 4, 6), 48, 1152, 2),
        ((1, 32, 32), 48, 1024, 2),
        ((1, 64, 96), 48, 1152, 2),
    ],
)
def test_native_accuracy(dtype, grid, side, hidden, merge):
    torch.manual_seed(37948)
    weight = torch.randn(side * side, hidden, device="npu", dtype=dtype)
    saved = weight.clone()
    actual = vision_pos_embed_interpolate(weight, *grid, side, merge, dtype)
    expected = pos_embed_interpolate_native(weight, *grid, side, merge, dtype)
    _compare(actual, expected)
    assert torch.equal(weight, saved)
    frame = grid[1] * grid[2]
    for i in range(1, grid[0]):
        assert torch.equal(actual[:frame], actual[i * frame : (i + 1) * frame])


def test_exact_reorder_and_endpoint():
    # Identity grid: no interpolation error can conceal a permutation bug.
    weight = torch.arange(16 * 7, device="npu", dtype=torch.float32).reshape(16, 7)
    actual = vision_pos_embed_interpolate(weight, 2, 4, 4, 4, 2, weight.dtype)
    order = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15], device="npu")
    assert torch.equal(actual, weight[order].repeat(2, 1))
    assert torch.equal(actual[-1], weight[-1])


def test_dispatch_alternating_sizes_and_graph_replay():
    tower = SimpleNamespace(
        pos_embed=SimpleNamespace(weight=torch.randn(48 * 48, 1152, device="npu", dtype=torch.bfloat16)),
        num_grid_per_side=48,
        spatial_merge_size=2,
        dtype=torch.bfloat16,
    )
    grids = [[1, 8, 12], [2, 4, 6]]
    interpolate = patch_qwen3vl._fast_pos_embed_interpolate
    # This captures a consumer, with metadata regenerated outside the graph,
    # matching the encoder replay preparation boundary (not a full tower test).
    staging = interpolate(tower, grids)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        consumed = staging + 1
    for current in (list(reversed(grids)), grids, list(reversed(grids))):
        tower.pos_embed.weight.add_(0.125)
        with patch.object(patch_qwen3vl, "vision_pos_embed_interpolate", wraps=vision_pos_embed_interpolate) as fused:
            staging.copy_(interpolate(tower, current))
            assert fused.call_count == len(current)
        expected = torch.cat(
            [pos_embed_interpolate_native(tower.pos_embed.weight, *g, 48, 2, tower.dtype) for g in current]
        )
        _compare(staging, expected)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(consumed, staging + 1)
