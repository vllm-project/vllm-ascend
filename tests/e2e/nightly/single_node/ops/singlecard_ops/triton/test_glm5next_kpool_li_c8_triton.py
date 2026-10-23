# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LI C8 packed-row layout tests for the GLM-Next KPool Triton kernels.

Every pooled row stores HEAD_DIM quantized bytes plus one little-endian FP32
scale. The write kernel derives both from the row absmax; the score kernel
factors the scale out of the dot product.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer_triton as score,
)

HEAD_DIM = 128
ROW_WIDTH = HEAD_DIM + 4
POOL_SIZE = 4
SENTINEL = 0x55


def _quantize_rows(rows: torch.Tensor) -> torch.Tensor:
    """CPU mirror of the write kernel's packed-row quantization."""
    abs_max = rows.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(abs_max > 0, abs_max / 127.0, torch.zeros_like(abs_max))
    normalized = rows / torch.where(abs_max > 0, scale, torch.ones_like(scale))
    quantized = torch.round(normalized).clamp(-127.0, 127.0).to(torch.int8)
    packed = torch.zeros(rows.shape[0], ROW_WIDTH, dtype=torch.uint8)
    packed[:, :HEAD_DIM] = quantized.view(torch.uint8)
    packed[:, HEAD_DIM:] = scale.view(torch.uint8)
    return packed


def _dequantize_rows(packed: torch.Tensor) -> torch.Tensor:
    quantized = packed[:, :HEAD_DIM].view(torch.int8).float()
    scale = packed[:, HEAD_DIM:].contiguous().view(torch.float32)
    return quantized * scale


@pytest.mark.parametrize("page_rows", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_c8_cann_working_set_masks_short_requests_and_page_padding(page_rows, padded):
    from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_c8_dequant_pool_key

    generator = torch.Generator().manual_seed(83)
    lengths = [257, 13, 0]
    pages = (max(lengths) + page_rows - 1) // page_rows
    blocks = len(lengths) * pages
    storage = torch.full((blocks, page_rows * ROW_WIDTH + (16 if padded else 0)), SENTINEL, dtype=torch.int8)
    cache = storage[:, : page_rows * ROW_WIDTH].view(blocks, page_rows, 1, ROW_WIDTH)
    table = torch.randperm(blocks, generator=generator).view(len(lengths), pages).to(torch.int32)
    expected = torch.zeros(len(lengths), 384, HEAD_DIM, dtype=torch.bfloat16)
    for req, length in enumerate(lengths):
        rows = _quantize_rows(torch.randn(length, HEAD_DIM, generator=generator))
        expected[req, :length] = _dequantize_rows(rows).bfloat16()
        for row in range(length):
            cache[table[req, row // page_rows], row % page_rows, 0] = rows[row].view(torch.int8)
        # Invalid pages must never be read, including the empty DSA-CP shard.
        table[req, (length + page_rows - 1) // page_rows :] = 2**30
    npu_storage = storage.npu()
    npu_cache = npu_storage[:, : page_rows * ROW_WIDTH].view_as(cache)
    args = (npu_cache, table.npu(), torch.tensor(lengths, device="npu"), max(lengths), HEAD_DIM)
    workspace = glm5_next_c8_dequant_pool_key(*args)
    assert workspace is not None
    keys, identity = workspace
    torch.testing.assert_close(keys.cpu().reshape_as(expected), expected, rtol=0, atol=0)
    torch.testing.assert_close(identity.cpu(), torch.arange(9, dtype=torch.int32).view(3, 3))
    keys.fill_(42)
    reused = glm5_next_c8_dequant_pool_key(*args, workspace)
    assert reused[0] is keys and reused[1] is identity
    torch.testing.assert_close(keys.cpu().reshape_as(expected), expected, rtol=0, atol=0)
    torch.testing.assert_close(npu_storage.cpu(), storage, rtol=0, atol=0)


def test_c8_cann_working_set_budget_falls_back_without_allocating(monkeypatch):
    import vllm_ascend.ops.triton.glm5_next_lightning_indexer as module

    monkeypatch.setattr(module, "TRITON_PACKED_CACHE_BYTES", 1)
    assert (
        module.glm5_next_c8_dequant_pool_key(
            torch.empty(1, 32, 1, ROW_WIDTH, dtype=torch.int8),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
            HEAD_DIM,
        )
        is None
    )


@pytest.mark.parametrize("packing", ["paged", "packed"])
@torch.inference_mode()
def test_kpool_li_c8_score_dequantizes_rows(packing):
    generator = torch.Generator().manual_seed(31)
    num_heads, index_topk = 32, 8
    rows_per_req, rows_per_block = 16, 8
    num_tokens = 40 if packing == "packed" else 3

    keys = torch.randn(2, rows_per_req, HEAD_DIM, generator=generator)
    packed_rows = _quantize_rows(keys.reshape(-1, HEAD_DIM))
    backing = torch.full((4, rows_per_block, 1, ROW_WIDTH), SENTINEL, dtype=torch.uint8)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    for req in range(2):
        for row in range(rows_per_req):
            block = int(table[req, row // rows_per_block])
            backing[block, row % rows_per_block, 0] = packed_rows[req * rows_per_req + row]
    cache = backing.npu().view(torch.int8)
    dequantized = _dequantize_rows(packed_rows).reshape(2, rows_per_req, HEAD_DIM)

    last_pos = rows_per_req * POOL_SIZE + 2
    per_req = num_tokens // 2
    positions = [last_pos - 1 if i % 3 == 0 else last_pos for i in range(num_tokens)]
    ends = torch.tensor([per_req, num_tokens], dtype=torch.int32)
    pool_lens = torch.tensor([rows_per_req, rows_per_req], dtype=torch.int32)
    query = torch.randn(num_tokens, num_heads, HEAD_DIM, generator=generator).bfloat16().npu()
    weights = torch.randn(num_tokens, num_heads, generator=generator).bfloat16().npu()

    result = score(
        query,
        cache,
        weights,
        ends.npu(),
        pool_lens.npu(),
        table.npu(),
        torch.tensor(positions, device="npu"),
        index_topk=index_topk,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=rows_per_req,
        pack_tail=True,
    )

    result, query, weights = result.cpu(), query.cpu(), weights.cpu()
    selected = index_topk // POOL_SIZE
    for row in range(num_tokens):
        req = 0 if row < per_req else 1
        pos = positions[row]
        visible = min((pos + 1) // POOL_SIZE, rows_per_req)
        dots = query[row].float() @ dequantized[req, :visible].T
        scores = (dots.relu() * weights[row].float()[:, None]).sum(0)
        top_pools = torch.topk(scores, min(selected, visible)).indices
        expected = torch.full((index_topk,), -1, dtype=torch.int32)
        history = (top_pools[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten()
        expected[: history.numel()] = history.to(torch.int32)
        actual = result[row, 0]
        actual_topk, expected_topk = actual[:index_topk].sort().values, expected[:index_topk].sort().values
        torch.testing.assert_close(actual_topk, expected_topk, rtol=0, atol=0)
        tail_start = (pos + 1) // POOL_SIZE * POOL_SIZE
        tail = torch.arange(tail_start, pos + 1, dtype=torch.int32)
        tail_column = min(tail_start, index_topk)
        torch.testing.assert_close(actual[tail_column : tail_column + tail.numel()], tail, rtol=0, atol=0)
        assert bool((actual[tail_column + tail.numel() :] == -1).all())


@torch.inference_mode()
def test_kpool_li_c8_score_matches_cann_per_head_relu():
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import cann_ops_transformer  # noqa: F401

    generator = torch.Generator().manual_seed(53)
    pools, page_rows, topk = 700, 32, 2048
    # Integer keys and unit scales are exactly representable in both cache
    # formats, so this isolates scoring semantics from quantization error.
    keys = torch.randint(-16, 17, (pools, HEAD_DIM), generator=generator).float()
    half_query = torch.randn(16, HEAD_DIM, generator=generator).bfloat16()
    query = torch.cat((half_query, -half_query)).unsqueeze(0)
    weights = torch.ones(1, 32, dtype=torch.bfloat16)
    # Collapsing heads would cancel every query and assign every pool zero.
    assert torch.count_nonzero((query.float() * weights.unsqueeze(-1)).sum(1)) == 0
    scores = (query[0].float() @ keys.T).relu().sum(0)
    reference = torch.topk(scores, topk // POOL_SIZE).indices
    bf16_cache = torch.zeros(32, page_rows, 1, HEAD_DIM, dtype=torch.bfloat16)
    bf16_cache.view(-1, HEAD_DIM)[:pools] = keys
    packed = torch.zeros(32, page_rows, 1, ROW_WIDTH, dtype=torch.uint8)
    packed.view(-1, ROW_WIDTH)[:pools, :HEAD_DIM] = keys.to(torch.int8).view(torch.uint8)
    packed.view(-1, ROW_WIDTH)[:pools, HEAD_DIM:] = torch.ones(pools, 1).view(torch.uint8)
    query, weights = query.npu(), weights.npu()
    ends = torch.tensor([1], dtype=torch.int32, device="npu")
    lengths = torch.tensor([pools], dtype=torch.int32, device="npu")
    table = torch.arange(32, dtype=torch.int32, device="npu").view(1, 32)
    positions = torch.tensor([pools * POOL_SIZE - 1], device="npu")
    expected, _ = torch.ops.cann_ops_transformer.pool_key_indexer(
        query,
        bf16_cache.npu(),
        weights,
        torch.zeros(1, dtype=torch.int32, device="npu"),
        actual_seq_q=ends,
        actual_seq_k=lengths,
        block_table=table,
        topk=topk,
        pool_size=POOL_SIZE,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
    )
    actual = score(
        query,
        packed.npu().view(torch.int8),
        weights,
        ends,
        lengths,
        table,
        positions,
        index_topk=topk,
        index_kpool=POOL_SIZE,
        max_pool_seq_len=pools,
        pack_tail=True,
    )
    reference_tokens = (reference[:, None] * POOL_SIZE + torch.arange(POOL_SIZE)).flatten().sort().values
    torch.testing.assert_close(actual.cpu()[0, 0, :topk].sort().values.long(), reference_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.cpu().reshape(-1).sort().values, expected.cpu().reshape(-1).sort().values, rtol=0, atol=0
    )


@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_kpool_li_c8_padding_and_empty_requests(graph_mode):
    query = torch.zeros(5, 32, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    weights = torch.ones(5, 32, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(2, 8, 1, ROW_WIDTH, dtype=torch.int8, device="npu")
    ends = torch.tensor([3, 3], dtype=torch.int32, device="npu")
    lengths = torch.zeros(2, dtype=torch.int32, device="npu")
    table = torch.zeros(2, 1, dtype=torch.int32, device="npu")
    positions = torch.tensor([0, 1, 2, 0, 0], device="npu")

    def run():
        return score(
            query,
            cache,
            weights,
            ends,
            lengths,
            table,
            positions,
            index_topk=8,
            index_kpool=POOL_SIZE,
            max_pool_seq_len=8,
            pack_tail=True,
        )

    run()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
    for q, k, p in (
        ([3, 3], [0, 0], [0, 1, 2, 0, 0]),
        ([0, 3], [0, 0], [0, 1, 2, 0, 0]),
        ([0, 0], [0, 0], [0, 0, 0, 0, 0]),
        ([2, 4], [1, 2], [3, 4, 7, 8, 0]),
    ):
        for target, values in ((ends, q), (lengths, k), (positions, p)):
            target.copy_(torch.tensor(values, dtype=target.dtype, device="npu"))
        if graph is None:
            output = run()
        else:
            graph.replay()
        actual = output.cpu()
        for row in range(5):
            expected = torch.full((11,), -1, dtype=torch.int32)
            if row < q[-1]:
                expected[: p[row] + 1] = torch.arange(p[row] + 1, dtype=torch.int32)
            torch.testing.assert_close(actual[row, 0].sort().values, expected.sort().values, rtol=0, atol=0)


@torch.inference_mode()
def test_kpool_li_c8_score_rejects_fp8_rows():
    # The score kernel unpacks quantized rows from INT16 word pairs, which
    # only supports int8 payloads; fp8 rows must be rejected explicitly.
    cache = torch.zeros((2, 8, 1, ROW_WIDTH), dtype=torch.uint8).npu().view(torch.float8_e4m3fn)
    query = torch.zeros((1, 2, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    weights = torch.zeros((1, 2), dtype=torch.bfloat16, device="npu")
    with pytest.raises(NotImplementedError, match="int8"):
        score(
            query,
            cache,
            weights,
            torch.tensor([1], dtype=torch.int32, device="npu"),
            torch.tensor([1], dtype=torch.int32, device="npu"),
            torch.zeros((1, 1), dtype=torch.int32, device="npu"),
            torch.tensor([0], dtype=torch.int64, device="npu"),
            index_topk=4,
            index_kpool=POOL_SIZE,
            max_pool_seq_len=8,
        )


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("tail_padding", [0, 32])
@torch.inference_mode()
def test_cann_pooled_cache_writer_preserves_tail_and_padded_pages(quant, speculative, tail_padding):
    from vllm_ascend.ops.cann_keypool import prepare_keypool_state
    from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

    generator = torch.Generator().manual_seed(41)
    dim, capacity, page_rows = HEAD_DIM, 7, 16
    width = ROW_WIDTH if quant else HEAD_DIM
    dtype = torch.int8 if quant else torch.bfloat16
    page_elements = page_rows * 2 * dim + tail_padding
    tail_storage = torch.full((3 * page_elements,), 55.0, device="npu")
    tail = tail_storage.as_strided((3, capacity, 2 * dim), (page_elements, 2 * dim, 1))
    cache_storage = torch.full((3, 48, 1, width), SENTINEL, dtype=dtype, device="npu")
    cache = cache_storage[:, :32]
    pooled = torch.randn(4, dim, generator=generator).bfloat16()
    pooled[0] = 0
    keys = torch.randn(3, dim, generator=generator)
    gates = torch.randn(3, dim, generator=generator)
    state = torch.randn(4, 4, 2 * dim, generator=generator)
    writes = torch.tensor([[1, page_rows + 2, -1]], dtype=torch.int64, device="npu")
    state_rows = torch.tensor([[1, 6, 0]], dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor([0, 33, -1, 96], device="npu"))
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(
            write_indices=writes,
            state_indices=state_rows,
            page_rows=page_rows,
        )
    )
    write_keypool_caches(
        tail,
        cache,
        keys.npu() if speculative else None,
        gates.npu() if speculative else None,
        pooled.npu(),
        metadata,
        tail_metadata,
        state.npu(),
    )
    expected_tail = torch.full_like(tail_storage.cpu(), 55.0)
    source = torch.cat((keys, gates), dim=-1) if speculative else state.view(-1, 2 * dim)[[1, 6]]
    expected_tail_view = expected_tail.as_strided((3, capacity, 2 * dim), (page_elements, 2 * dim, 1))
    expected_tail_view[0, 1] = source[0]
    expected_tail_view[1, 2] = source[1]
    torch.testing.assert_close(tail_storage.cpu(), expected_tail, rtol=0, atol=0)
    expected_cache = torch.full_like(cache_storage.cpu(), SENTINEL)
    rows = _quantize_rows(pooled.float()).view(torch.int8) if quant else pooled
    expected_cache[0, 0, 0] = rows[0]
    expected_cache[1, 1, 0] = rows[1]
    actual_cache = cache_storage.cpu()
    if quant:
        # BF16 inputs can land exactly on a half-integer quantization tie.
        # CPU and device division may differ by one ULP at that boundary;
        # check the quantization error bound, while keeping scales, padding
        # and every untouched row byte-exact.
        for block, row, source_row in ((0, 0, 0), (1, 1, 1)):
            actual = actual_cache[block, row, 0]
            expected = expected_cache[block, row, 0]
            assert (actual[:dim].to(torch.int16) - expected[:dim].to(torch.int16)).abs().max() <= 1
            scale = actual[dim:].contiguous().view(torch.float32)
            error = (actual[:dim].float() * scale - pooled[source_row].float()).abs()
            assert torch.all(error <= scale * 0.5001 + 1e-7)
            expected[:dim] = actual[:dim]
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    history = torch.tensor([[0, 1, 2, 3], [page_rows + 2, page_rows + 3, page_rows, page_rows + 1]], device="npu")
    tail_metadata.cann_inputs.history_indices = history
    gathered = prepare_keypool_state(tail, torch.zeros(2, device="npu", dtype=torch.int32), tail_metadata, POOL_SIZE)
    expected_history = torch.stack((expected_tail_view[0, [0, 1, 2, 3]], expected_tail_view[1, [2, 3, 0, 1]]))
    torch.testing.assert_close(gathered[1:3].cpu(), expected_history, rtol=0, atol=0)
