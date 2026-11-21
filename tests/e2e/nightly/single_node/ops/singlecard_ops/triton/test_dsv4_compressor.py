# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.compressor.dsv4_compressor import compressor_from_projected


def projected_reference(kv, scores, state, ape, norm, sin, cos, table, cumulative, starts, ratio, dtype):
    """Independent CPU reference for paged C4/C128 pooling and state writes."""
    width, head_dim = kv.shape[1], norm.numel()
    coefficient = width // head_dim
    block_size = state.shape[1]
    updated = state.clone()
    out = torch.zeros(cos.shape[0], head_dim, dtype=dtype)
    out_row = 0
    for request, start in enumerate(starts.tolist()):
        begin, end = cumulative[request : request + 2].tolist()
        if start < 0:
            continue
        for group in range(start // ratio, (start + end - begin) // ratio):
            keys, gates = [], []
            for position in range((group - coefficient + 1) * ratio, (group + 1) * ratio):
                if position < 0:
                    continue
                head = 0 if coefficient == 1 or position < group * ratio else head_dim
                if position >= start:
                    row = begin + position - start
                    keys.append(kv[row, head : head + head_dim])
                    gates.append(scores[row, head : head + head_dim] + ape[position % ratio, head : head + head_dim])
                else:
                    page = table[request, position // block_size]
                    cached = state[page, position % block_size]
                    keys.append(cached[head : head + head_dim])
                    gates.append(cached[width + head : width + head + head_dim])
            pooled = (torch.stack(gates).softmax(0) * torch.stack(keys)).sum(0)
            normalized = pooled * torch.rsqrt(pooled.square().mean() + 1e-6) * norm
            rope_dim = cos.shape[-1]
            rope = normalized[-rope_dim:].clone()
            rotated = rope.reshape(-1, 2).flip(-1) * torch.tensor([-1.0, 1.0])
            normalized[-rope_dim:] = rope * cos[out_row] + rotated.flatten() * sin[out_row]
            out[out_row] = normalized.to(dtype)
            out_row += 1
        for row in range(begin, end):
            position = start + row - begin
            page = table[request, position // block_size]
            if page > 0:
                updated[page, position % block_size, :width] = kv[row]
                updated[page, position % block_size, width:] = scores[row] + ape[position % ratio]
    return out, updated


@pytest.mark.parametrize("ratio,head_dim", [(4, 128), (4, 512), (128, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_projected_compressor_paged_chunks(ratio, head_dim, dtype, use_graph, prefill_length=None):
    """Real C4 indexer/main and C128 shapes, history, padding and graph replay."""
    generator = torch.Generator().manual_seed(64880)
    width = (2 if ratio == 4 else 1) * head_dim
    lengths = [max(2 * ratio + 3, prefill_length or 0), ratio + 2, 0, 3]
    starts = torch.tensor([0, ratio - 1, ratio + 1, -1], dtype=torch.int32)
    cumulative = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    tokens = sum(lengths)
    block_size = 8 if ratio == 4 else 32
    max_position = max(start + length for start, length in zip(starts.tolist(), lengths))
    blocks_per_request = (max_position + block_size - 1) // block_size + 1
    table = torch.arange(1, 1 + len(lengths) * blocks_per_request, dtype=torch.int32).reshape(len(lengths), -1)
    # A current-token page can be evicted; its input projections still pool.
    table[0, 0] = 0
    page_count = table.max().item() + 1
    page_elements = block_size * 2 * width
    backing = torch.randn(page_count, page_elements + 32, generator=generator)
    state = backing[:, :page_elements].view(page_count, block_size, 2 * width)
    kv = torch.randn(tokens, width, generator=generator)
    scores = torch.randn(tokens, width, generator=generator)
    ape = torch.randn(ratio, width, generator=generator)
    norm = torch.randn(head_dim, generator=generator)
    output_rows = min(tokens, tokens // ratio + len(lengths))
    angles = torch.randn(output_rows, 32, generator=generator).repeat_interleave(2, -1)
    sin, cos = angles.sin(), angles.cos()
    device_backing = backing.npu()
    device_state = device_backing[:, :page_elements].view_as(state)
    arguments = [tensor.npu() for tensor in (kv, scores, ape, norm, sin, cos, table, cumulative, starts)]

    def run():
        device_kv, device_scores, device_ape, device_norm, device_sin, device_cos, device_table, cu, pos = arguments
        return compressor_from_projected(
            device_kv,
            device_scores,
            device_state,
            device_ape,
            device_norm,
            device_sin,
            device_cos,
            state_block_table=device_table,
            cu_seqlens=cu,
            start_pos=pos,
            compress_ratio=ratio,
            norm_eps=1e-6,
            output_dtype=dtype,
        )

    if use_graph:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        # Replay consumes updated projections; static capture values must not leak.
        kv.add_(0.3)
        arguments[0].copy_(kv)
        device_backing.copy_(backing)
        graph.replay()
    else:
        actual = run()
    expected, expected_state = projected_reference(
        kv, scores, state, ape, norm, sin, cos, table, cumulative, starts, ratio, dtype
    )
    tolerance = 2**-6 if dtype == torch.bfloat16 else 2**-9
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(device_state.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(device_backing[:, page_elements:].cpu(), backing[:, page_elements:], rtol=0, atol=0)
    torch.testing.assert_close(device_state[0].cpu(), state[0], rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("use_graph", [False, True])
def test_projected_compressor_long_c4(head_dim, use_graph):
    # Exceed the hardware core count so each core processes several cache rows.
    test_projected_compressor_paged_chunks(4, head_dim, torch.bfloat16, use_graph, prefill_length=384)


@torch.inference_mode()
def test_projected_compressor_empty_input():
    kv = torch.empty(0, 256, dtype=torch.float32, device="npu")
    state = torch.zeros(2, 8, 512, dtype=torch.float32, device="npu")
    output = compressor_from_projected(
        kv,
        kv,
        state,
        torch.zeros(4, 256, device="npu"),
        torch.ones(128, device="npu"),
        torch.empty(0, 64, device="npu"),
        torch.empty(0, 64, device="npu"),
        state_block_table=torch.ones(1, 1, dtype=torch.int32, device="npu"),
        cu_seqlens=torch.zeros(2, dtype=torch.int32, device="npu"),
        start_pos=torch.zeros(1, dtype=torch.int32, device="npu"),
        compress_ratio=4,
        norm_eps=1e-6,
        output_dtype=torch.bfloat16,
    )
    assert output.shape == (0, 128)
    assert torch.count_nonzero(state).item() == 0
