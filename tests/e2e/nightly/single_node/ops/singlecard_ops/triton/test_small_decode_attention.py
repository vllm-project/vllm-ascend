# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paged decode and changing graph inputs against CPU attention."""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.small_decode_attention import small_decode_attention
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def initialize_device():
    torch.npu.set_device(0)
    init_device_properties_triton()


def make_cache(layout):
    shape = (16, 128, 8, 128)
    if layout == "contiguous":
        key = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        return key, torch.randn_like(key)
    block_elements = math.prod(shape[1:])
    padding = 32 if layout == "padded_pages" else 0
    page_stride = 2 * block_elements + padding
    storage = torch.randn(shape[0] * page_stride, device="npu", dtype=torch.bfloat16)
    strides = (page_stride, 8 * 128, 128, 1)
    key = storage.as_strided(shape, strides)
    value = storage.as_strided(shape, strides, storage_offset=block_elements)
    return key, value


def reference(query, key, value, table, length, scale):
    if length == 0:
        return torch.zeros_like(query, dtype=torch.float32)
    keys = key[table[0].long()].reshape(-1, 8, 128)[:length].repeat_interleave(2, dim=1)
    values = value[table[0].long()].reshape(-1, 8, 128)[:length].repeat_interleave(2, dim=1)
    probabilities = torch.einsum("hd,shd->hs", query[0].float(), keys.float()).mul(scale).softmax(-1)
    return torch.einsum("hs,shd->hd", probabilities, values.float()).unsqueeze(0)


@pytest.mark.parametrize("layout", ["contiguous", "interleaved_kv", "padded_pages"])
@pytest.mark.parametrize("magnitude", [1, 8])
@torch.inference_mode()
def test_small_decode_graph_replay(layout, magnitude):
    torch.manual_seed(123 + magnitude)
    query = torch.randn(1, 16, 128, device="npu", dtype=torch.bfloat16)
    key, value = make_cache(layout)
    value.mul_(magnitude)
    table = torch.arange(8, device="npu", dtype=torch.int32).reshape(1, 8)
    lengths = torch.tensor([128], device="npu", dtype=torch.int32)
    output = torch.empty(1, 16 * 128, device="npu", dtype=torch.bfloat16)
    scale = 1 / math.sqrt(128)
    args = (query, key, value, table, lengths, scale, output)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            small_decode_attention(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        small_decode_attention(*args)
    key_cpu, value_cpu = key.cpu(), value.cpu()
    # Reuse the same graph while crossing page/tile boundaries and remapping
    # physical pages. A dense-block assumption passes contiguous microtests
    # but reads the wrong data in the model's combined K/V allocation.
    for length in [0, 1, 127, 128, 129, 160, 255, 256, 511, 512, 513, 1023, 1024]:
        query.normal_(std=magnitude)
        permutation = torch.randperm(16)[:8].int().reshape(1, 8)
        table.copy_(permutation)
        lengths.fill_(length)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        actual = output.view_as(query).cpu().float()
        expected = reference(query.cpu(), key_cpu, value_cpu, permutation, length, scale)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0.004 * magnitude, rtol=0.01)
