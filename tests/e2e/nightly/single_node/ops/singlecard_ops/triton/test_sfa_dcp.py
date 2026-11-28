# SPDX-License-Identifier: Apache-2.0
"""Single-NPU precision and changing-input graph tests for SFA Triton helpers."""

from statistics import median

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import _pack_token_peer_tiled  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_merge import fused_merge  # noqa: E402
from vllm_ascend.ops.triton.query_gather_prep import (  # noqa: E402
    prep_query_head_major,
    prepare_sfa_kv_pair,
    prepare_sfa_query_pair,
)
from vllm_ascend.ops.triton.sfa_indexer_store import store_indexer_key_scale  # noqa: E402
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402


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
        lse,
        send,
        words.stride(0),
        words.stride(1),
        lse.stride(0),
        lse.stride(1),
        lse.stride(0),
        lse.stride(1),
        tokens,
        local_heads,
        head_dim // 2,
        4096,
        FROM_STATS=False,
    )
    actual = send.cpu()
    expected_words = owner_cpu.transpose(0, 1).view(torch.int32)
    expected_words = expected_words.reshape(tokens, ranks, local_heads, head_dim // 2).permute(1, 0, 2, 3)
    expected_stats = stats_cpu.reshape(tokens, ranks, local_heads).permute(1, 0, 2)
    assert torch.equal(actual[:, :output_size], expected_words.reshape(ranks, -1))
    assert torch.equal(actual[:, output_size:], expected_stats.reshape(ranks, -1))


@pytest.mark.parametrize(
    "widths,tokens,heads,dtype,column_stride",
    [
        ((512, 64), 1, 3, torch.bfloat16, 1),
        ((256, 128), 6, 8, torch.float16, 2),
        ((64, 512), 13, 1, torch.float32, 1),
        ((512, 64), 192, 1, torch.bfloat16, 2),
        ((256, 128), 13, 3, torch.float32, 2),
        ((64, 512), 6, 8, torch.float16, 1),
    ],
)
def test_query_bytes_and_changing_graph_inputs(widths, tokens, heads, dtype, column_stride):
    torch.npu.set_device(0)
    owners = [torch.empty(heads, tokens, width * column_stride, dtype=dtype, device="npu") for width in widths]
    qn, qr = [owner.transpose(0, 1)[..., ::column_stride] for owner in owners]

    def invoke():
        packed = prep_query_head_major(qn, qr)
        return packed, packed.transpose(0, 1).split(widths, dim=-1)

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
            assert torch.equal(actual.cpu().view(bit_dtype), source.cpu().view(bit_dtype))
    torch.npu.synchronize()


@pytest.mark.parametrize("tokens,heads,output_heads", [(7, 32, 32), (17, 96, 128), (128, 96, 128)])
def test_prepare_sfa_query_pair_gathered_views_and_graph_replay(tokens, heads, output_heads):
    widths = (512, 64)
    owner = torch.zeros(heads + 1, tokens, sum(widths), dtype=torch.bfloat16, device="npu")
    query, rope = owner[1:].transpose(0, 1).split(widths, dim=-1)
    assert not query.is_contiguous() and not rope.is_contiguous()
    assert 0 < query.storage_offset() < rope.storage_offset()

    for _ in range(3):
        prepare_sfa_query_pair(query, rope, output_heads)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = prepare_sfa_query_pair(query, rope, output_heads)

    bits = torch.arange(owner.numel(), dtype=torch.int32).reshape(owner.shape)
    for step in (0, 1):
        pattern = (bits * 17 + step * 4099).to(torch.int16)
        # Include -0, infinity and a NaN payload in both real fragments.
        special = torch.tensor([-32768, 32640, 32705], dtype=torch.int16)
        pattern[1, 0, :3] = special
        pattern[1, 0, widths[0] : widths[0] + 3] = special
        owner.view(torch.int16).copy_(pattern)
        for out in outputs:
            out.fill_(3.0)  # Replay must also overwrite every padding lane.
        graph.replay()
        for out, original in zip(outputs, pattern[1:].transpose(0, 1).split(widths, -1), strict=True):
            assert out.is_contiguous() and out.dtype == torch.bfloat16
            expected = torch.nn.functional.pad(original, (0, 0, 0, output_heads - heads))
            assert torch.equal(out.cpu().view(torch.int16), expected)
    graph.reset()


def _sfa_cache_planes(owner, block_size):
    # Each physical page holds all latent rows, then all positional rows.
    # Skip a prefix in the backing allocation. The page pitch may include padding.
    pages = owner[1:]
    split = block_size * 512
    kv = pages[:, :split].unflatten(1, (block_size, 1, 512))
    key_rope = pages[:, split : split + block_size * 64].unflatten(1, (block_size, 1, 64))
    return kv, key_rope


@pytest.mark.parametrize("pages,block_size", [(5, 128), (9, 17)])
def test_prepare_sfa_kv_pair_page_ids_and_graph_replay(pages, block_size):
    owner = torch.zeros(pages + 1, block_size * 576 + 64, dtype=torch.bfloat16, device="npu")
    kv, key_rope = _sfa_cache_planes(owner, block_size)
    assert not kv.is_contiguous() and not key_rope.is_contiguous()
    assert 0 < kv.storage_offset() < key_rope.storage_offset()

    for _ in range(3):
        prepare_sfa_kv_pair(kv, key_rope)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = prepare_sfa_kv_pair(kv, key_rope)

    counters = torch.arange(owner.numel(), dtype=torch.int32).reshape(owner.shape)
    for step in (0, 1):
        bits = (counters * 13 + step * 8191).to(torch.int16)
        expected_planes = _sfa_cache_planes(bits, block_size)
        for expected in expected_planes:
            expected[:, 0, 0, :3] = torch.tensor([-32768, 32640, 32705], dtype=torch.int16)
        owner.view(torch.int16).copy_(bits)
        graph.replay()
        for out, expected in zip(outputs, expected_planes, strict=True):
            assert out.ndim == 4 and out.shape == expected.shape and out.is_contiguous()
            assert out.dtype == torch.bfloat16
            assert torch.equal(out.cpu().view(torch.int16), expected)
    graph.reset()


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="A5 BF16 sparse attention")
def test_prepare_sfa_kv_pair_native_attention():
    enable_custom_op()
    generator = torch.Generator().manual_seed(928)
    pages, block_size, heads = 4, 128, 64
    owner = torch.randn(pages + 1, block_size * 576 + 64, dtype=torch.bfloat16, generator=generator).npu()
    kv, key_rope = _sfa_cache_planes(owner, block_size)
    gathered = torch.randn(heads, 1, 576, dtype=torch.bfloat16, generator=generator).npu()
    query, query_rope = gathered.transpose(0, 1).split((512, 64), -1)
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    indices[0, 0, :7] = torch.tensor([0, 127, 128, 255, 256, 384, 511])
    kwargs = dict(
        sparse_indices=indices.npu(),
        scale_value=1 / 24,
        sparse_block_size=1,
        block_table=torch.tensor([[2, 0, 3, 1]], dtype=torch.int32).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([pages * block_size], dtype=torch.int32).npu(),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    def attend(q, qr, k, kr):
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q, query_rope=qr, key=k, value=k, key_rope=kr, **kwargs
        )

    reference = attend(query.contiguous(), query_rope.contiguous(), kv.contiguous(), key_rope.contiguous())
    actual = attend(*prepare_sfa_query_pair(query, query_rope, heads), *prepare_sfa_kv_pair(kv, key_rope))
    for value, expected in zip(actual, reference, strict=True):
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(value, expected, atol=0, rtol=0)


def _capture_sfa_pair(calls):
    graphs = []
    results = []
    for prepare in calls:
        for _ in range(3):
            prepare()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            results.append(prepare())
        graphs.append(graph)
        for _ in range(3):
            graph.replay()
    torch.npu.synchronize()
    return graphs, results


def _sfa_pair_replay_us(graphs):
    # Alternate replay order to reduce frequency and cache-order bias.
    # Timing events bracket only work submitted to the NPU stream.
    samples = ([], [])
    for sample in range(8):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / 10)
    return [median(values) for values in samples]


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="A5 BF16 sparse attention")
@pytest.mark.parametrize(
    "block_size,pages,requests,table_width",
    [(128, 64, 8, 19), (896, 64, 8, 19), (128, 5208, 32, 133)],
    ids=["small-128", "small-896", "dcp8-decode-128-queries"],
)
def test_sfa_native_paired_pages_decode_replay_and_latency(block_size, pages, requests, table_width, record_property):
    enable_custom_op()
    generator = torch.Generator().manual_seed(928)
    # One local shard. Pad the 96 gathered Kimi heads just as DCP8 does.
    q_per_request, heads = 4, 96
    tokens = requests * q_per_request
    owner = torch.randn(pages + 1, block_size * 576, dtype=torch.bfloat16, generator=generator).npu()
    kv, key_rope = _sfa_cache_planes(owner, block_size)
    packed = owner[1:].view(pages, block_size, 1, 576)
    gathered = torch.randn(heads, tokens, 576, dtype=torch.bfloat16, generator=generator).npu()
    q, qr = prepare_sfa_query_pair(*gathered.transpose(0, 1).split((512, 64), -1), 128)
    # Leave one unmapped logical page and skip the reserved physical page zero.
    table = torch.stack(
        [torch.randperm(pages - 1, generator=generator)[:table_width] + 1 for _ in range(requests)]
    ).int()
    table[:, 1] = -1
    visible = table_width * block_size - 7
    indices = torch.full((tokens, 1, 2048), -1, dtype=torch.int32)
    for row in range(2, tokens):
        selected = torch.randperm(visible, generator=generator)
        selected = selected[selected // block_size != 1][:257].sort().values
        indices[row, 0, : selected.numel()] = selected.int()
    indices[0, 0, 0] = 0  # One value permits an exact independent check after a cache write.
    kwargs = dict(
        query=q,
        query_rope=qr,
        sparse_indices=indices.npu(),
        scale_value=1 / 24,
        sparse_block_size=1,
        block_table=table.npu(),
        actual_seq_lengths_query=torch.arange(q_per_request, tokens + 1, q_per_request, dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.full((requests,), visible, dtype=torch.int32).npu(),
        layout_query="TND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    def copy_and_attend():
        k, kr = prepare_sfa_kv_pair(kv, key_rope)
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            key=k, value=k, key_rope=kr, layout_kv="PA_BSND", **kwargs
        )

    def attend_packed():
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            key=packed, value=packed, key_rope=packed, layout_kv="PA_KR_BSND", **kwargs
        )

    graphs, (reference, actual) = _capture_sfa_pair((copy_and_attend, attend_packed))
    for step in range(2):
        # Update only the selected row. Both captured graphs must observe the live backing and the remapped page.
        if step:
            table[:, [0, 2]] = table[:, [2, 0]]
            kwargs["block_table"].copy_(table)
        page = int(table[0, 0])
        new_value = torch.randn((512,), dtype=torch.bfloat16, generator=generator)
        kv[page, 0, 0].copy_(new_value)
        key_rope[page, 0, 0].fill_(step + 1)
        for graph in graphs:
            graph.replay()
        for out, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        out, maximum, total = actual
        assert out.shape == (tokens, 128, 512) and out.dtype == torch.bfloat16
        assert maximum.shape == total.shape == (1, tokens, 128) and maximum.dtype == total.dtype == torch.float32
        assert torch.equal(out[0].cpu(), new_value.expand(128, -1))
        assert torch.count_nonzero(out[1]) == 0 and torch.count_nonzero(total[:, 1]) == 0
        assert torch.isfinite(maximum[:, 2:]).all() and (total[:, 2:] > 0).all()

    copy_sfa_us, paired_sfa_us = _sfa_pair_replay_us(graphs)
    record_property("copy_sfa_us", copy_sfa_us)
    record_property("paired_sfa_us", paired_sfa_us)
    print(
        f"SFA decode pages={pages} block_size={block_size} copy_sfa_us={copy_sfa_us:.3f} "
        f"paired_sfa_us={paired_sfa_us:.3f}"
    )
    for graph in graphs:
        graph.reset()


@pytest.mark.parametrize("stage", ["query", "kv"])
def test_sfa_pair_layout_latency(stage, record_property):
    tokens, heads, output_heads = 128, 96, 128
    pages, block_size = 256, 128
    if stage == "query":
        owner = torch.randn(heads, tokens, 576, dtype=torch.bfloat16, device="npu")
        q, qr = owner.transpose(0, 1).split((512, 64), -1)

        def separate():
            padding = (0, 0, 0, output_heads - heads)
            return torch.nn.functional.pad(q, padding), torch.nn.functional.pad(qr, padding)

        def fused():
            return prepare_sfa_query_pair(q, qr, output_heads)
    else:
        owner = torch.randn(pages + 1, block_size * 576 + 64, dtype=torch.bfloat16, device="npu")
        kv, kr = _sfa_cache_planes(owner, block_size)

        def separate():
            return kv.contiguous(), kr.contiguous()

        def fused():
            return prepare_sfa_kv_pair(kv, kr)

    graphs, results = _capture_sfa_pair((separate, fused))
    for reference, actual in zip(*results, strict=True):
        assert torch.equal(reference, actual)
    separate_us, fused_us = _sfa_pair_replay_us(graphs)
    record_property("separate_us", separate_us)
    record_property("fused_us", fused_us)
    print(f"SFA layout stage={stage} separate_us={separate_us:.3f} fused_us={fused_us:.3f}")
    for graph in graphs:
        graph.reset()


@pytest.mark.parametrize(
    "ranks,token_dim,head_dim,dtype",
    [
        (2, 1, 127, torch.float16),
        (4, 2, 496, torch.bfloat16),
        (8, 1, 512, torch.float32),
        (2, 2, 513, torch.float32),
        (8, 2, 127, torch.bfloat16),
        (4, 1, 513, torch.float16),
    ],
)
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


@pytest.mark.parametrize(
    "rows,ranks,rank,dtype,interleave",
    [
        (1, 2, 0, torch.int32, 1),
        (6, 4, 1, torch.int64, 64),
        (12, 8, 7, torch.int32, 128),
        (1, 8, 0, torch.int64, 256),
        (6, 2, 1, torch.int64, 128),
        (12, 4, 3, torch.int32, 64),
    ],
)
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
    for step in range(3):
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


@pytest.mark.parametrize(
    "capacity,slot_dtype,width",
    [(256, torch.int32, 64), (512, torch.int64, 128), (256, torch.int64, 256), (512, torch.int32, 256)],
)
def test_native_quantized_cache_bytes_and_changing_graph_slots(capacity, slot_dtype, width):
    torch.npu.set_device(0)
    retained = []
    for rows in (1, 13, 192):
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
        for step in range(4):
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
