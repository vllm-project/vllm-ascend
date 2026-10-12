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

from vllm_ascend.ops.triton.glm5_next_kpool_tail_compress import (
    glm5_next_kpool_tail_compress_and_write_cache_triton as compress,
)
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer_triton as score,
)

HEAD_DIM = 128
ROW_WIDTH = HEAD_DIM + 4
POOL_SIZE = 4
SENTINEL = 0x55

# fp8 e4m3 casts need the A5 toolchain; bishengir on 910_93 cannot lower
# arith::TruncFOp to fp8 ("unsupported datatype ... to hfusion").
REQUIRES_A5_FP8 = pytest.mark.skipif(
    "950" not in torch.npu.get_device_name(0),
    reason="fp8 e4m3 casts require the A5 toolchain",
)


def _quant_max(dtype: torch.dtype) -> float:
    return 127.0 if dtype == torch.int8 else 448.0


def _quantize_rows(rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """CPU mirror of the write kernel's packed-row quantization."""
    abs_max = rows.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(abs_max > 0, abs_max / _quant_max(dtype), torch.zeros_like(abs_max))
    normalized = rows / torch.where(abs_max > 0, scale, torch.ones_like(scale))
    if dtype == torch.int8:
        quantized = torch.round(normalized).clamp(-127.0, 127.0).to(torch.int8)
    else:
        quantized = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    packed = torch.zeros(rows.shape[0], ROW_WIDTH, dtype=torch.uint8)
    packed[:, :HEAD_DIM] = quantized.view(torch.uint8)
    packed[:, HEAD_DIM:] = scale.view(torch.uint8)
    return packed


def _dequantize_rows(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    quantized = packed[:, :HEAD_DIM].view(dtype).float()
    scale = packed[:, HEAD_DIM:].contiguous().view(torch.float32)
    return quantized * scale


@pytest.mark.parametrize("quant_dtype", [torch.int8, pytest.param(torch.float8_e4m3fn, marks=REQUIRES_A5_FP8)])
@torch.inference_mode()
def test_kpool_li_c8_compress_packs_quantized_rows(quant_dtype):
    generator = torch.Generator().manual_seed(23)
    lengths = [20, 32]
    num_blocks, rows_per_block = 3, 8
    tail_ids, cache_ids = [1, 0], [2, 0]

    keys = torch.randn(2, 32, HEAD_DIM, generator=generator)
    keys[0, :POOL_SIZE] = 0.0  # an all-zero pool must land as q=0, scale=0
    gates = torch.randn(2, 32, HEAD_DIM, generator=generator) * 0.1
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=generator) * 0.1

    tail = torch.zeros(2, 2, POOL_SIZE, HEAD_DIM, device="npu")
    backing = torch.full((num_blocks, rows_per_block, 1, ROW_WIDTH), SENTINEL, dtype=torch.uint8, device="npu")
    cache = backing.view(quant_dtype)

    positions, tail_slots, pool_slots, input_k, input_g = [], [], [], [], []
    expected_rows: dict[int, torch.Tensor] = {}
    for req, length in enumerate(lengths):
        for pos in range(length):
            positions.append(pos)
            input_k.append(keys[req, pos])
            input_g.append(gates[req, pos])
            tail_slots.append(tail_ids[req] * POOL_SIZE + pos % POOL_SIZE)
            if (pos + 1) % POOL_SIZE == 0:
                slot = cache_ids[req] * rows_per_block + pos // POOL_SIZE
                window = slice(pos - POOL_SIZE + 1, pos + 1)
                pooled = (torch.softmax(gates[req, window] + ape, dim=0) * keys[req, window]).sum(0)
                expected_rows[slot] = pooled
            else:
                slot = -1
            pool_slots.append(slot)

    compress(
        tail,
        cache,
        torch.stack(input_k).npu(),
        torch.stack(input_g).npu(),
        ape.npu(),
        torch.tensor(positions, device="npu"),
        torch.tensor(lengths, dtype=torch.int32).cumsum(0).int().npu(),
        torch.tensor(lengths, dtype=torch.int32).npu(),
        torch.tensor(tail_slots, device="npu"),
        torch.tensor([[i] for i in tail_ids], dtype=torch.int32).npu(),
        torch.tensor(pool_slots, device="npu"),
        POOL_SIZE,
    )

    packed = backing.cpu()
    quant_max = _quant_max(quant_dtype)
    # Each side rounds to its own grid point: a one-level flip near a rounding
    # boundary stays within one full quantization step of the FP32 reference.
    atol_ratio = 1.05 / quant_max if quant_dtype == torch.int8 else 0.16
    for slot, pooled in expected_rows.items():
        block, row = divmod(slot, rows_per_block)
        packed_row = packed[block, row, 0]
        scale = packed_row[HEAD_DIM:].view(torch.float32)
        expected_scale = pooled.abs().amax() / quant_max
        torch.testing.assert_close(scale, expected_scale.reshape(1), rtol=1e-6, atol=0)
        dequantized = _dequantize_rows(packed_row.unsqueeze(0), quant_dtype)[0]
        torch.testing.assert_close(
            dequantized,
            pooled,
            rtol=0,
            atol=float(expected_scale) * (atol_ratio * quant_max) + 1e-4,
        )
    # Block 1 owns no request and must keep the sentinel pattern.
    assert bool((packed[1] == SENTINEL).all())


@pytest.mark.parametrize("quant_dtype", [torch.int8])
@pytest.mark.parametrize("packing", ["paged", "packed"])
@torch.inference_mode()
def test_kpool_li_c8_score_dequantizes_rows(quant_dtype, packing):
    generator = torch.Generator().manual_seed(31)
    num_heads, index_topk = 32, 8
    rows_per_req, rows_per_block = 16, 8
    num_tokens = 40 if packing == "packed" else 3

    keys = torch.randn(2, rows_per_req, HEAD_DIM, generator=generator)
    packed_rows = _quantize_rows(keys.reshape(-1, HEAD_DIM), quant_dtype)
    backing = torch.full((4, rows_per_block, 1, ROW_WIDTH), SENTINEL, dtype=torch.uint8)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    for req in range(2):
        for row in range(rows_per_req):
            block = int(table[req, row // rows_per_block])
            backing[block, row % rows_per_block, 0] = packed_rows[req * rows_per_req + row]
    cache = backing.npu().view(quant_dtype)
    dequantized = _dequantize_rows(packed_rows, quant_dtype).reshape(2, rows_per_req, HEAD_DIM)

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
        qbar = (query[row].float() * weights[row].float()[:, None]).sum(0)
        scores = dequantized[req, :visible] @ qbar
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
    rows = _quantize_rows(pooled.float(), torch.int8).view(torch.int8) if quant else pooled
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
