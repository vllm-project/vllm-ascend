# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer

# GLM-5.3-Flash indexer parameters, including its actual top-k width.
HEAD_DIM = 128
NUM_HEADS = 32
POOL_SIZE = 4
INDEX_TOPK = 2048
CACHE_BLOCK_SIZE = 16
OUTPUT_WIDTH = INDEX_TOPK + POOL_SIZE - 1


def _assert_selection(result, query, cache, weights, ends, pool_lens, table, positions, pack_tail=False):
    """CPU reference: score each query head before applying the head weights."""
    result, query, cache, weights = (x.cpu() for x in (result, query, cache, weights))
    start = 0
    for req, end in enumerate(ends.tolist()):
        for row in range(start, end):
            pos = int(positions[row])
            count = min((pos + 1) // POOL_SIZE, int(pool_lens[req]))
            ids = torch.arange(count)
            keys = cache[table[req, ids // CACHE_BLOCK_SIZE].long(), ids % CACHE_BLOCK_SIZE, 0].float()
            per_head_scores = query[row].float() @ keys.T
            scores = (per_head_scores * weights[row].float()[:, None]).sum(0)
            selected = torch.topk(scores, min(INDEX_TOPK // POOL_SIZE, count)).indices
            history = (selected[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten()
            expected = torch.full((OUTPUT_WIDTH,), -1, dtype=torch.int32)
            expected[: history.numel()] = history.to(torch.int32)
            tail = torch.arange((pos + 1) // POOL_SIZE * POOL_SIZE, pos + 1, dtype=torch.int32)
            tail_column = min((pos + 1) // POOL_SIZE * POOL_SIZE, INDEX_TOPK) if pack_tail else INDEX_TOPK
            expected[tail_column : tail_column + tail.numel()] = tail
            actual = result[row, 0]
            # Top-k ordering may differ for tied scores; token membership,
            # multiplicity, fixed tail columns, and every padding lane must match.
            torch.testing.assert_close(
                actual[:INDEX_TOPK].sort().values,
                expected[:INDEX_TOPK].sort().values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual[INDEX_TOPK:], expected[INDEX_TOPK:], rtol=0, atol=0)
            if pack_tail:
                torch.testing.assert_close(actual[tail_column : tail_column + tail.numel()], tail, rtol=0, atol=0)
        start = end


@pytest.mark.parametrize("max_pool_seq_len", [0, 4, 512, 2050])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_pool_selection_real_shape_paging_and_causal_tail(max_pool_seq_len, use_graph, monkeypatch):
    generator = torch.Generator().manual_seed(19)
    # Three requests exercise non-power-of-two bucketization. Two extra rows
    # model graph padding, which the fused expansion clears.
    ends = torch.tensor([3, 5, 8], dtype=torch.int32)
    num_tokens = 10
    pages = (max_pool_seq_len + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
    num_blocks = max(1, 3 * pages)
    table = torch.randperm(num_blocks, generator=generator)[: 3 * pages].reshape(3, pages).to(torch.int32)
    backing = torch.randn(num_blocks, CACHE_BLOCK_SIZE + 2, 1, HEAD_DIM, generator=generator).bfloat16()
    cache = backing.npu()[:, :CACHE_BLOCK_SIZE]
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, generator=generator).bfloat16().npu()
    weights = torch.randn(num_tokens, NUM_HEADS, generator=generator).bfloat16().npu()
    pool_lens = torch.tensor([0, min(4, max_pool_seq_len), max_pool_seq_len], dtype=torch.int32)
    last_pos = max(2, max_pool_seq_len * POOL_SIZE + 2)
    positions = torch.tensor([0, 1, 2, 1, 2, last_pos - 2, last_pos - 1, last_pos, 0, 0])
    device_lens, device_positions = pool_lens.npu(), positions.npu()
    args = (query, cache, weights, ends.npu(), device_lens, table.npu(), device_positions)
    kwargs = dict(index_topk=INDEX_TOPK, index_kpool=POOL_SIZE, max_pool_seq_len=max_pool_seq_len)
    if use_graph:
        # Exercise the model's packed-tail path and a strided SFA output view.
        backing_output = torch.full((2 * num_tokens, 2080), 77, dtype=torch.int32, device="npu")
        output_buffer = backing_output[::2]
        kwargs.update(output_buffer=output_buffer, pack_tail=True)
    # Force several token chunks without a large scratch allocation. This also
    # checks that request lookup uses the batch-global token offset.
    monkeypatch.setattr(indexer, "TRITON_SCORES_CHUNK_BYTES", max(1, max_pool_seq_len) * 4 * 3)

    if use_graph:
        indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
    for step in range(2):
        if step:
            query.copy_(torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, generator=generator).bfloat16().npu())
            weights.copy_(torch.randn(num_tokens, NUM_HEADS, generator=generator).bfloat16().npu())
            pool_lens[2] //= 2
            positions[3:5] = torch.tensor([15, 16])
            device_lens.copy_(pool_lens)
            device_positions.copy_(positions)
        if use_graph:
            graph.replay()
        else:
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        assert result.shape == (num_tokens, 1, OUTPUT_WIDTH)
        assert result.dtype == torch.int32
        _assert_selection(result, query, cache, weights, ends, pool_lens, table, positions, pack_tail=use_graph)
        assert (result[int(ends[-1]) :] == -1).all()
        if use_graph:
            assert result.data_ptr() == output_buffer.data_ptr()
            assert (output_buffer[:, OUTPUT_WIDTH:] == -1).all()
            assert (backing_output[1::2] == 77).all()
        torch.testing.assert_close(cache.cpu(), backing[:, :CACHE_BLOCK_SIZE], rtol=0, atol=0)


def test_empty_query():
    result = indexer.glm5_next_lightning_indexer_triton(
        torch.empty(0, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu"),
        torch.empty(1, CACHE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu"),
        torch.empty(0, NUM_HEADS, dtype=torch.bfloat16, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.empty(0, 0, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int64, device="npu"),
        index_topk=INDEX_TOPK,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=0,
    )
    assert result.shape == (0, 1, OUTPUT_WIDTH)
    assert result.dtype == torch.int32


def _monotonic_paged_cache(capacity, generator):
    """Encode logical pool IDs exactly with three BF16 base-128 digits."""
    pages = (capacity + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
    physical_pages = torch.randperm(pages, generator=generator)
    logical = torch.arange(pages * CACHE_BLOCK_SIZE)
    keys = torch.zeros(pages * CACHE_BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    keys[:, 0] = logical // (128 * 128)
    keys[:, 1] = logical // 128 % 128
    keys[:, 2] = logical % 128
    cache = torch.empty(pages, CACHE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    cache[physical_pages, :, 0] = keys.view(pages, CACHE_BLOCK_SIZE, HEAD_DIM)
    table = physical_pages.to(torch.int32).repeat(3, 1)
    return cache.npu(), table.npu()


@pytest.mark.parametrize("capacity", [8193, 262208])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_streaming_topk_length_growth_and_shrink_reuses_output(capacity, use_graph, monkeypatch):
    generator = torch.Generator().manual_seed(41)
    num_tokens = 6
    cache, table = _monotonic_paged_cache(capacity, generator)
    query = torch.zeros(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    query[:, 0, :3] = torch.tensor([128 * 128, 128, 1], dtype=torch.bfloat16, device="npu")
    weights = torch.zeros(num_tokens, NUM_HEADS, dtype=torch.bfloat16, device="npu")
    weights[:, 0] = torch.tensor([1, -1, 1, -1, 1, 1], dtype=torch.bfloat16, device="npu")
    ends = torch.tensor([1, 3, 4], dtype=torch.int32)
    lengths = torch.tensor([0, 1024, 1024], dtype=torch.int32)
    positions = torch.full((num_tokens,), 4098, dtype=torch.int64)
    device_ends, device_lengths, device_positions = ends.npu(), lengths.npu(), positions.npu()
    args = (query, cache, weights, device_ends, device_lengths, table, device_positions)
    backing = torch.full((2 * (num_tokens + 1), 2080), 77, dtype=torch.int32, device="npu")
    buffer = backing[::2]
    kwargs = dict(
        index_topk=INDEX_TOPK,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=capacity,
        output_buffer=buffer,
        pack_tail=True,
        allow_cache_packing=False,
        use_streaming_topk=True,
    )
    # Exercise global request lookup across multiple token chunks.
    monkeypatch.setattr(indexer, "TRITON_SCORES_CHUNK_BYTES", 2 * 4 * 512 * 8)
    topk_shapes = []
    original_topk = torch.topk

    def record_topk(scores, *args, **kwargs):
        topk_shapes.append(scores.shape)
        return original_topk(scores, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", record_topk)
    if use_graph:
        indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)

    for step, length in enumerate([1024, 4099, capacity, 511, 0, 2051]):
        ends[:] = torch.tensor([0, 2, 5] if step % 2 else [1, 3, 4])
        lengths[:] = torch.tensor([min(length, 4), length, length])
        positions[:] = max(2, length * POOL_SIZE + 2)
        positions[1] = min(int(positions[1]), 1537 * POOL_SIZE + 1)
        device_ends.copy_(ends)
        device_lengths.copy_(lengths)
        device_positions.copy_(positions)
        if use_graph:
            graph.replay()
        else:
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        actual = result.cpu()[:, 0]
        expected = torch.full_like(actual, -1)
        start = 0
        for request, end in enumerate(ends.tolist()):
            for row in range(start, end):
                position = int(positions[row])
                visible = min(int(lengths[request]), (position + 1) // POOL_SIZE)
                count = min(visible, INDEX_TOPK // POOL_SIZE)
                pools = torch.arange(visible - count, visible) if row % 2 == 0 else torch.arange(count)
                history = (pools[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten()
                expected[row, : history.numel()] = history.to(torch.int32)
                tail_start = (position + 1) // POOL_SIZE * POOL_SIZE
                tail = torch.arange(tail_start, position + 1, dtype=torch.int32)
                tail_column = min(tail_start, INDEX_TOPK)
                expected[row, tail_column : tail_column + tail.numel()] = tail
                torch.testing.assert_close(actual[row, tail_column : tail_column + tail.numel()], tail, rtol=0, atol=0)
            start = end
        torch.testing.assert_close(actual.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0)
        assert result.data_ptr() == buffer.data_ptr()
        assert (buffer[:num_tokens, OUTPUT_WIDTH:] == -1).all()
        assert (buffer[num_tokens:] == 77).all()
        assert (backing[1::2] == 77).all()
    assert topk_shapes and all(shape[0] <= 2 and shape[1] == 2048 for shape in topk_shapes)


@pytest.mark.parametrize("special_scores", [False, True])
@torch.inference_mode()
def test_streaming_topk_handles_ties_and_nonfinite_scores(special_scores):
    capacity = 4099
    generator = torch.Generator().manual_seed(43)
    cache, table = _monotonic_paged_cache(capacity, generator)
    query = torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    weights = torch.zeros(1, NUM_HEADS, dtype=torch.bfloat16, device="npu")
    weights[:, 0] = 1
    if special_scores:
        query[:, 0, :3] = torch.tensor([128 * 128, 128, 1], dtype=torch.bfloat16, device="npu")
        for logical, value in [(3, float("nan")), (5, float("inf")), (7, -float("inf"))]:
            physical = int(table[0, logical // CACHE_BLOCK_SIZE].cpu())
            cache[physical, logical % CACHE_BLOCK_SIZE, 0, 0] = value
    args = (
        query,
        cache,
        weights,
        torch.tensor([1], dtype=torch.int32, device="npu"),
        torch.tensor([capacity], dtype=torch.int32, device="npu"),
        table[:1],
        torch.tensor([capacity * POOL_SIZE - 1], dtype=torch.int64, device="npu"),
    )
    result = indexer.glm5_next_lightning_indexer_triton(
        *args,
        index_topk=INDEX_TOPK,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=capacity,
        allow_cache_packing=False,
        use_streaming_topk=True,
    ).cpu()[0, 0]
    pools = result[:INDEX_TOPK].view(-1, POOL_SIZE)
    assert ((pools[:, 0] >= 0) & (pools[:, 0] < capacity * POOL_SIZE)).all()
    assert torch.unique(pools[:, 0]).numel() == INDEX_TOPK // POOL_SIZE
    torch.testing.assert_close(pools, pools[:, :1] + torch.arange(POOL_SIZE, dtype=pools.dtype), rtol=0, atol=0)
    assert (result[INDEX_TOPK:] == -1).all()
    if special_scores:
        expected = torch.cat((torch.tensor([5]), torch.arange(capacity - 511, capacity))) * POOL_SIZE
        torch.testing.assert_close(pools[:, 0].sort().values, expected.sort().values.to(torch.int32), rtol=0, atol=0)


def _pool_topk_case(capacity):
    generator = torch.Generator().manual_seed(47)
    cache, table = _monotonic_paged_cache(capacity, generator)
    query = torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    query[:, 0, :3] = torch.tensor([128 * 128, 128, 1], dtype=torch.bfloat16, device="npu")
    weights = torch.zeros(1, NUM_HEADS, dtype=torch.bfloat16, device="npu")
    weights[:, 0] = 1
    return (
        query,
        cache,
        weights,
        torch.tensor([1], dtype=torch.int32, device="npu"),
        torch.tensor([capacity], dtype=torch.int32, device="npu"),
        table[:1],
        torch.tensor([capacity * POOL_SIZE + 2], dtype=torch.int64, device="npu"),
    )


def _run_pool_topk_case(args, selected, capacity, use_graph):
    kwargs = dict(
        index_topk=selected * POOL_SIZE,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=capacity,
        allow_cache_packing=False,
        use_streaming_topk=True,
    )
    if use_graph:
        indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
        graph.replay()
    else:
        result = indexer.glm5_next_lightning_indexer_triton(*args, **kwargs)
    return result.cpu()[0, 0]


def _assert_pool_ids_and_tail(result, expected_ids, selected, position):
    topk = selected * POOL_SIZE
    history = torch.full((topk,), -1, dtype=torch.int32)
    tokens = (expected_ids[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten()
    history[: tokens.numel()] = tokens.to(torch.int32)
    torch.testing.assert_close(result[:topk].sort().values, history.sort().values, rtol=0, atol=0)
    tail = torch.arange((position + 1) // POOL_SIZE * POOL_SIZE, position + 1, dtype=torch.int32)
    expected_tail = torch.full((POOL_SIZE - 1,), -1, dtype=torch.int32)
    expected_tail[: tail.numel()] = tail
    torch.testing.assert_close(result[topk:], expected_tail, rtol=0, atol=0)


@pytest.mark.parametrize(
    "selected,capacity,streaming",
    [
        (127, 512, False),
        (127, 513, True),
        (129, 1024, False),
        (129, 1025, True),
        (511, 2048, False),
        (511, 2049, True),
        (513, 4097, False),
    ],
)
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_pool_topk_non_power_of_two_dispatch_boundary(selected, capacity, streaming, use_graph, monkeypatch):
    args = _pool_topk_case(capacity)
    widths = []
    original_topk = torch.topk

    def record_topk(scores, *args, **kwargs):
        widths.append(scores.shape[1])
        return original_topk(scores, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", record_topk)
    result = _run_pool_topk_case(args, selected, capacity, use_graph)
    expected_width = indexer.TRITON_TOPK_PARTITIONS * (1 << (selected - 1).bit_length()) if streaming else capacity
    assert widths and all(width == expected_width for width in widths)
    _assert_pool_ids_and_tail(result, torch.arange(capacity - selected, capacity), selected, capacity * POOL_SIZE + 2)


@pytest.mark.parametrize("kind", ["empty_visible", "nan", "negative_inf", "few_valid"])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_streaming_topk_all_invalid_or_fewer_valid_than_k(kind, use_graph):
    capacity, selected = 4099, 129
    args = _pool_topk_case(capacity)
    cache, lengths, table, positions = args[1], args[4], args[5], args[6]
    expected_ids = torch.empty(0, dtype=torch.long)
    position = capacity * POOL_SIZE + 2
    if kind == "empty_visible":
        lengths.zero_()
        position = 2
        positions.fill_(position)
    else:
        data = cache.cpu()
        data[:, :, 0, 0] = -float("inf") if kind == "negative_inf" else float("nan")
        if kind == "few_valid":
            expected_ids = torch.tensor([3, 1091, 4098])
            physical = table[0].cpu()[expected_ids // CACHE_BLOCK_SIZE].long()
            data[physical, expected_ids % CACHE_BLOCK_SIZE, 0, 0] = 0
        cache.copy_(data)
    result = _run_pool_topk_case(args, selected, capacity, use_graph)
    _assert_pool_ids_and_tail(result, expected_ids, selected, position)


@pytest.mark.parametrize("selected", [129, 511])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_streaming_topk_all_winners_in_one_partition(selected, use_graph):
    capacity = 8193
    args = _pool_topk_case(capacity)
    tile = 1 << (selected - 1).bit_length()
    split = selected // 2
    # Both tiles belong to partition zero. Winners in the later tile force a
    # merge; no other partition may supply any of the selected candidates.
    later = indexer.TRITON_TOPK_PARTITIONS * tile
    expected_ids = torch.cat((torch.arange(split), torch.arange(later, later + selected - split)))
    cache, table = args[1], args[5]
    data = cache.cpu()
    physical = table[0].cpu()[expected_ids // CACHE_BLOCK_SIZE].long()
    data[physical, expected_ids % CACHE_BLOCK_SIZE, 0, 0] = 1
    cache.copy_(data)
    result = _run_pool_topk_case(args, selected, capacity, use_graph)
    _assert_pool_ids_and_tail(result, expected_ids, selected, capacity * POOL_SIZE + 2)


@torch.inference_mode()
def test_streaming_topk_requires_explicit_opt_in(monkeypatch):
    capacity, selected = 4099, 129
    args = _pool_topk_case(capacity)
    widths = []
    original_topk = torch.topk

    def record_topk(scores, *args, **kwargs):
        widths.append(scores.shape[1])
        return original_topk(scores, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", record_topk)
    result = indexer.glm5_next_lightning_indexer_triton(
        *args,
        index_topk=selected * POOL_SIZE,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=capacity,
        allow_cache_packing=False,
    ).cpu()[0, 0]
    _assert_pool_ids_and_tail(result, torch.arange(capacity - selected, capacity), selected, capacity * POOL_SIZE + 2)
    assert widths == [capacity]
