# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import rotary_embedding as rope


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_cache", [False, True])
def test_rope_cache_graph_replay(monkeypatch, dtype, use_cache):
    """Changing positions must update graph outputs without changing addresses."""
    for name in ("_cos_cache", "_sin_cache", "_cos_mla", "_sin_mla"):
        monkeypatch.setattr(rope, name, None)
    table = torch.randn(257, 64, dtype=dtype, device="npu")
    rope._record_cos_and_sin_cache_interleaved(table)
    assert rope._cos_cache.is_contiguous()
    assert rope._sin_cache.is_contiguous()
    rope._cos_mla = torch.empty(4, 1, 1, 64, dtype=dtype, device="npu")
    rope._sin_mla = torch.empty_like(rope._cos_mla)
    positions = torch.tensor([0, 1, 99, 256], device="npu")
    expected = [half.repeat(1, 2) for half in table.chunk(2, dim=-1)]
    for _ in range(3):
        rope.get_cos_and_sin_mla(positions, use_cache=use_cache)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = rope.get_cos_and_sin_mla(positions, use_cache=use_cache)
    pointers = [tensor.data_ptr() for tensor in outputs]
    if use_cache:
        assert pointers == [rope._cos_mla.data_ptr(), rope._sin_mla.data_ptr()]
    for indices in ([256, 0, 99, 99], [1, 17, 129, 255]):
        positions.copy_(torch.tensor(indices, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        assert pointers == [tensor.data_ptr() for tensor in outputs]
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference[positions, None, None], rtol=0, atol=0)
