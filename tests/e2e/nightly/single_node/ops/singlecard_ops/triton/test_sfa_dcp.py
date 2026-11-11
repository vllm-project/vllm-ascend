# SPDX-License-Identifier: Apache-2.0
"""Single-NPU precision and changing-input graph tests for SFA Triton helpers."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import _pack_token_peer_tiled  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_merge import fused_merge  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_query import prepare_query_head_major, unpack_query  # noqa: E402
from vllm_ascend.ops.triton.sfa_indexer_store import store_indexer_key_scale  # noqa: E402
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton  # noqa: E402


@pytest.mark.parametrize("tokens,local_heads,head_dim", [(513, 32, 512), (257, 64, 512), (1, 64, 1024)])
@pytest.mark.parametrize("head_major_lse", [False, True])
def test_tiled_pack_preserves_raw_bits(tokens, local_heads, head_dim, head_major_lse):
    ranks, heads = 2, 2 * local_heads
    bits = torch.tensor([0, -32768, 1, 16256, 32640, -128, 32705, -63], dtype=torch.int16)
    owner_cpu = bits[torch.arange(heads * tokens * head_dim) % bits.numel()].view(heads, tokens, head_dim)
    output = owner_cpu.npu().view(torch.bfloat16).transpose(0, 1)
    lse_bits = torch.tensor([0, -2147483648, 1065353216, 2139095040, -8388608, 2143289345], dtype=torch.int32)
    stats_cpu = lse_bits[torch.arange(tokens * heads) % lse_bits.numel()].view(tokens, heads, 1)
    lse = stats_cpu.npu().view(torch.float32)
    if head_major_lse:
        lse = lse.transpose(0, 1).contiguous().transpose(0, 1)
    words = output.view(torch.int32)
    output_size = tokens * local_heads * (head_dim // 2)
    send = torch.empty((ranks, output_size + tokens * local_heads), dtype=torch.int32, device="npu")
    _pack_token_peer_tiled[(ranks,)](
        words,
        lse,
        send,
        words.stride(0),
        words.stride(1),
        lse.stride(0),
        lse.stride(1),
        tokens,
        local_heads,
        head_dim // 2,
        4096,
    )
    actual = send.cpu()
    expected_words = owner_cpu.transpose(0, 1).view(torch.int32)
    expected_words = expected_words.reshape(tokens, ranks, local_heads, head_dim // 2).permute(1, 0, 2, 3)
    expected_stats = stats_cpu.reshape(tokens, ranks, local_heads).permute(1, 0, 2)
    assert torch.equal(actual[:, :output_size], expected_words.reshape(ranks, -1))
    assert torch.equal(actual[:, output_size:], expected_stats.reshape(ranks, -1))


@pytest.mark.parametrize("widths", [(512, 64), (256, 128), (64, 512)])
@pytest.mark.parametrize("tokens,heads", [(1, 3), (6, 8), (13, 1), (192, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("column_stride", [1, 2])
def test_query_bytes_and_changing_graph_inputs(widths, tokens, heads, dtype, column_stride):
    torch.npu.set_device(0)
    owners = [torch.empty(heads, tokens, width * column_stride, dtype=dtype, device="npu") for width in widths]
    qn, qr = [owner.transpose(0, 1)[..., ::column_stride] for owner in owners]

    def invoke():
        packed = prepare_query_head_major(qn, qr)
        return packed, unpack_query(packed, widths)

    for _ in range(3):
        invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed, fragments = invoke()
    bit_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    for step in range(3):
        for owner in owners:
            bits = torch.arange(owner.numel()).mul(17).add(step * 4099).to(bit_dtype).reshape(owner.shape)
            patterns = (
                [0, -2147483648, 2139095040, 2143289345] if bit_dtype == torch.int32 else [0, -32768, 32640, 32705]
            )
            bits.view(-1)[:4] = torch.tensor(patterns, dtype=bit_dtype)
            owner.copy_(bits.view(dtype))
        graph.replay()
        expected = torch.cat((qn.cpu(), qr.cpu()), dim=-1).transpose(0, 1).contiguous()
        assert torch.equal(packed.cpu().view(bit_dtype), expected.view(bit_dtype))
        for actual, source in zip(fragments, (qn, qr)):
            assert actual.is_contiguous()
            assert torch.equal(actual.cpu().view(bit_dtype), source.cpu().view(bit_dtype))
    torch.npu.synchronize()


@pytest.mark.parametrize("ranks", [2, 4, 8])
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


def oracle(indices, rank, ranks=8, interleave=128):
    expected = torch.full_like(indices, -1)
    # CPU int64 arithmetic is independent of NPU vector integer division.
    count = indices.shape[-1]
    for source, target in zip(indices.long().reshape(-1, count), expected.reshape(-1, count)):
        owned = source[(source >= 0) & ((source // interleave) % ranks == rank)]
        target[: owned.numel()] = (owned // (ranks * interleave) * interleave + owned % interleave).to(target.dtype)
    return expected


@pytest.mark.parametrize("rows", [1, 6, 12])
@pytest.mark.parametrize("ranks,rank", [(ranks, rank) for ranks in (2, 4, 8) for rank in range(ranks)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("interleave", [1, 64, 128, 256])
def test_remap_exact_integer_boundaries_and_changed_graph_inputs(rows, ranks, rank, dtype, interleave):
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(20260911 + rows + rank + interleave)
    base = torch.randint(0, 2**31 - 1, (rows, 2048), generator=rng, dtype=dtype)
    boundaries = torch.tensor(
        [
            -1,
            0,
            interleave - 1,
            interleave,
            ranks * interleave - 1,
            ranks * interleave,
            2**24 - 1,
            2**24,
            2**24 + interleave - 1,
            2**31 - 1,
        ],
        dtype=dtype,
    )
    if dtype == torch.int64:
        base[:, ::3] += 2**40
        boundaries = torch.cat((boundaries, torch.tensor([2**31, 2**40, 2**62 - 1], dtype=dtype)))
    device_input = base.npu()
    for _ in range(5):
        remap_sparse_indices_triton(device_input, ranks, rank, interleave)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = remap_sparse_indices_triton(device_input, ranks, rank, interleave)
    for step in range(8):
        current = torch.roll(base, step, dims=-1)
        current[:, : boundaries.numel()] = boundaries
        if step == 0:
            current.fill_(-1)
        device_input.copy_(current)
        graph.replay()
        assert torch.equal(output.cpu(), oracle(current, rank, ranks, interleave))
    torch.npu.synchronize()


@pytest.mark.parametrize("shape", [(0, 2048), (1, 0), (2, 3, 0)])
def test_remap_empty_input(shape):
    value = torch.empty(shape, dtype=torch.int32, device="npu")
    assert remap_sparse_indices_triton(value, 8, 0, 128) is value


@pytest.mark.parametrize("ranks,interleave", [(4, 128), (8, 64), (2, 1), (3, 127)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_remap_integer_layout_boundaries_and_graph(ranks, interleave, dtype):
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(16350)
    base = torch.randint(0, 2**31 - 1, (2, 3, 257), generator=rng, dtype=dtype)
    boundary = torch.tensor([-1, 0, 127, 128, 2**24 - 1, 2**24, 2**24 + 1, 2**31 - 1], dtype=dtype)
    base[..., : boundary.numel()] = boundary
    for rank in range(ranks):
        # Exercise noncontiguous input and arbitrary leading dimensions.
        storage = base.transpose(-1, -2).contiguous().npu()
        value = storage.transpose(-1, -2)
        for _ in range(3):
            remap_sparse_indices_triton(value, ranks, rank, interleave)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = remap_sparse_indices_triton(value, ranks, rank, interleave)
        for step in range(3):
            current = torch.roll(base, step, dims=-1) if step else torch.full_like(base, -1)
            storage.copy_(current.transpose(-1, -2).contiguous())
            graph.replay()
            assert torch.equal(result.cpu(), oracle(current, rank, ranks, interleave))
    torch.npu.synchronize()


@pytest.mark.parametrize("capacity", [256, 512])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_native_quantized_cache_bytes_and_changing_graph_slots(capacity, slot_dtype, width):
    torch.npu.set_device(0)
    retained = []
    for rows in (1, 6, 12, 13, 48, 192):
        x = torch.randn(rows, width, dtype=torch.bfloat16, device="npu")
        keys, scales = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        slots = torch.arange(rows, dtype=slot_dtype, device="npu") + 7
        backing = torch.full((capacity * (width + 2) + 64,), 19, dtype=torch.int8, device="npu")
        cache = backing[: capacity * width].view(capacity // 128, 128, 1, width)
        scale_cache = backing[capacity * width : -64].view(torch.float16).view(capacity // 128, 128, 1, 1)
        for _ in range(5):
            store_indexer_key_scale(keys, scales, slots, cache, scale_cache)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            store_indexer_key_scale(keys, scales, slots, cache, scale_cache)
        for step in range(16):
            fresh_keys, fresh_scales = torch_npu.npu_dynamic_quant(torch.randn_like(x), dst_type=torch.int8)
            keys.copy_(fresh_keys)
            scales.copy_(fresh_scales)
            selected = torch.arange(rows, dtype=torch.int64) + 3 + step
            if step % 2:
                selected[-1] = -1
            elif step % 4 == 0:
                selected[-1] = capacity - 1
            else:
                selected[-1] = capacity
            slots.copy_(selected)
            backing.fill_(19)
            scale_cache.fill_(3.25)
            graph.replay()
            torch.npu.synchronize()
            expected_keys = torch.full((capacity, width), 19, dtype=torch.int8)
            expected_scales = torch.full((capacity, 1), 3.25, dtype=torch.float16)
            valid = (selected >= 0) & (selected < capacity)
            expected_keys[selected[valid]] = fresh_keys.cpu()[valid]
            expected_scales[selected[valid], 0] = fresh_scales.to(torch.float16).cpu()[valid]
            assert torch.equal(cache.view(capacity, width).cpu(), expected_keys)
            assert torch.equal(scale_cache.view(capacity, 1).view(torch.int16).cpu(), expected_scales.view(torch.int16))
            assert bool((backing[-64:] == 19).all())
        retained.append((graph, backing, keys, scales, slots))
    torch.npu.synchronize()
