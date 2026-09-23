# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.ops.output_writeback import write_recurrent_output


@torch.inference_mode()
@pytest.mark.parametrize(
    "source_tokens,padding,heads,head_dim",
    [
        (0, 0, 8, 128),
        (0, 3, 8, 128),
        (1, 1, 1, 128),
        (3, 5, 3, 128),
        (7, 1, 4, 128),
        (17, 14, 8, 128),
        (31, 1, 16, 128),
        (63, 2, 32, 128),
        (127, 1, 64, 128),
        (128, 1, 8, 128),
        (129, 127, 16, 128),
        (255, 2, 32, 128),
        (256, 0, 64, 128),
        (257, 255, 8, 128),
        (512, 1, 16, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_graph", [False, True])
def test_recurrent_writeback_shape_boundaries(source_tokens, padding, heads, head_dim, dtype, use_graph):
    """Cover empty and ragged launches, DP shard widths, padding, and storage guards."""
    torch.manual_seed(1024)
    shape = (1, source_tokens + padding, heads, head_dim)
    source_shape = (1, source_tokens, heads, head_dim)
    elements = (source_tokens + padding) * heads * head_dim
    source_backing = torch.full((source_tokens * heads * head_dim + 34,), -9, device="npu", dtype=dtype)
    source = source_backing[17:-17].view(source_shape)
    backing = torch.full((elements + 34,), -7, device="npu", dtype=dtype)
    output = backing[17:-17].view(shape)
    query_backing = torch.tensor([-91, 0, 0, source_tokens, -93], device="npu", dtype=torch.int32)
    ends = query_backing[1:-1]
    if use_graph and elements:
        source.normal_()
        write_recurrent_output(source, output, ends)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write_recurrent_output(source, output, ends)
    for valid in [source_tokens, source_tokens // 2, 0, source_tokens]:
        source.normal_()
        source[:, valid:] = float("nan")
        ends[-1:].fill_(valid)
        output.fill_(float("nan"))
        if use_graph and elements:
            graph.replay()
        else:
            write_recurrent_output(source, output, ends)
        expected = torch.zeros(shape, device="npu", dtype=dtype)
        expected[:, :valid].copy_(source[:, :valid])
        assert torch.equal(output, expected)
        assert (backing[:17] == -7).all() and (backing[-17:] == -7).all()
        assert (source_backing[:17] == -9).all() and (source_backing[-17:] == -9).all()
        assert torch.equal(query_backing.cpu(), torch.tensor([-91, 0, 0, valid, -93], dtype=torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize(
    "source_tokens,output_tokens,heads",
    [(1, 2048, 16), (15, 533, 16), (62, 793, 16), (63, 1771, 16), (127, 2048, 32), (31, 2048, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_graph", [False, True])
def test_recurrent_writeback_dp_prefill_padding(source_tokens, output_tokens, heads, dtype, use_graph):
    """A decode DP rank can be padded to another rank's much larger prefill."""
    output = torch.empty((1, output_tokens, heads, 128), device="npu", dtype=dtype)
    expected = torch.zeros_like(output)
    ends = torch.tensor([0, source_tokens], device="npu", dtype=torch.int32)
    # Put the source at the end of an allocation. Small padding tests can hide
    # invalid fully masked loads when the out-of-range address is still mapped.
    backing = torch.full((1024 * 1024,), -9, device="npu", dtype=dtype)
    source_elements = source_tokens * heads * 128
    source = backing[-source_elements:].view(1, source_tokens, heads, 128)
    source.normal_()
    expected[:, :source_tokens].copy_(source)
    write_recurrent_output(source, output, ends)
    torch.npu.synchronize()
    assert torch.equal(output, expected)
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write_recurrent_output(source, output, ends)
    for valid in [0, source_tokens // 2, source_tokens]:
        source.normal_()
        ends[-1:].fill_(valid)
        expected.zero_()
        expected[:, :valid].copy_(source[:, :valid])
        output.fill_(float("nan"))
        if use_graph:
            graph.replay()
        else:
            write_recurrent_output(source, output, ends)
        assert torch.equal(output, expected)
        assert (backing[:-source_elements] == -9).all()


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_recurrent_writeback_dynamic_batch_sizes(dtype):
    """Reuse a kernel across changing query counts and DP padding lengths."""
    for source_tokens, output_tokens, query_count in [
        (1, 3, 1),
        (16, 32, 16),
        (63, 1771, 63),
        (128, 409, 128),
        (128, 404, 127),
        (125, 383, 64),
        (125, 506, 17),
        (119, 735, 2),
    ]:
        source = torch.randn((1, source_tokens, 16, 128), device="npu", dtype=dtype)
        output = torch.empty((1, output_tokens, 16, 128), device="npu", dtype=dtype)
        ends = torch.zeros(query_count + 1, device="npu", dtype=torch.int32)
        ends[-1:].fill_(source_tokens)
        write_recurrent_output(source, output, ends)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write_recurrent_output(source, output, ends)
        for valid in [source_tokens, 0, source_tokens // 2]:
            source.normal_()
            ends[-1:].fill_(valid)
            output.fill_(float("nan"))
            graph.replay()
            expected = torch.zeros_like(output)
            expected[:, :valid].copy_(source[:, :valid])
            assert torch.equal(output, expected)
