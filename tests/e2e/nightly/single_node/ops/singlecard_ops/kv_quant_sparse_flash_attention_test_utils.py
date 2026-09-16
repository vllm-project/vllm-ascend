# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing and independent reference helpers for INT8 sparse attention tests."""

import math
from dataclasses import dataclass
from itertools import accumulate

import torch

from .test_kv_quant_sparse_flash_attention import BF16_ATOL, BF16_RTOL

NOPE_DIM = 512
ROPE_DIM = 64
QUANT_TILE_SIZE = 128
PAGE_SIZE = 128
SCALE_GROUPS = NOPE_DIM // QUANT_TILE_SIZE
FP16_ATOL = 2.5e-5
FP16_RTOL = 5e-3
SCALE_VALUE = (NOPE_DIM + ROPE_DIM) ** -0.5


@dataclass(frozen=True)
class RopeCase:
    name: str
    query_layout: str
    kv_layout: str
    heads: int
    sparse_block_size: int
    max_tokens: int
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]


def _stored_query(tensor, case):
    if case.query_layout == "TND":
        return torch.cat([tensor[b, :length] for b, length in enumerate(case.query_lengths)])
    return tensor


def _make_cpu_case(case, dtype, rope_dim, *, uniform=False, value_sign=1):
    generator = torch.Generator().manual_seed(20260915)
    batch = len(case.query_lengths)
    query_seq = max(case.query_lengths)
    kv_seq = max(case.kv_lengths)
    query_shape = (batch, query_seq, case.heads, NOPE_DIM)
    key_shape = (batch, kv_seq, 1, NOPE_DIM)
    if uniform:
        query = torch.zeros(query_shape, dtype=dtype)
        query[..., 0] = (torch.arange(math.prod(query_shape[:-1])).reshape(query_shape[:-1]) % 7 + 1).to(dtype)
        key = (torch.arange(math.prod(key_shape)).reshape(key_shape) % 31 + 8).to(torch.int8)
        key.mul_(value_sign)
        key[..., 0] = 0
        exponent = (torch.arange(batch * kv_seq)[:, None] + torch.arange(SCALE_GROUPS)[None, :]) % 4 - 3
        scales = torch.pow(2.0, exponent).reshape(batch, kv_seq, 1, SCALE_GROUPS)
    else:
        query = (torch.rand(query_shape, generator=generator) * 2 - 1).to(dtype)
        key = torch.randint(-64, 64, key_shape, generator=generator, dtype=torch.int8)
        scales = torch.rand((batch, kv_seq, 1, SCALE_GROUPS), generator=generator) * 0.095 + 0.005

    query_rope = torch.zeros((*query_shape[:-1], rope_dim), dtype=dtype)
    key_rope = torch.zeros((*key_shape[:-1], rope_dim), dtype=dtype)
    packed = torch.cat((key, key_rope.view(torch.int8), scales.contiguous().view(torch.int8)), dim=-1)
    index_width = case.max_tokens // case.sparse_block_size
    sparse = torch.full((batch, query_seq, 1, index_width), -1, dtype=torch.int32)
    for batch_index, query_length in enumerate(case.query_lengths):
        for query_index in range(query_length):
            threshold = case.kv_lengths[batch_index] - query_length + query_index + 1
            available = math.ceil(threshold / case.sparse_block_size)
            count = min(index_width, available)
            # Always include the causal boundary block, including partially valid blocks.
            sparse[batch_index, query_index, 0, : count - 1] = torch.randperm(
                available - 1, generator=generator, dtype=torch.int32
            )[: count - 1]
            sparse[batch_index, query_index, 0, count - 1] = available - 1

    block_table = None
    if case.kv_layout == "PA_BSND":
        pages_per_sequence = tuple(math.ceil(length / PAGE_SIZE) for length in case.kv_lengths)
        page_count = sum(pages_per_sequence)
        physical_pages = torch.randperm(page_count, generator=generator, dtype=torch.int32)
        block_table = torch.full((batch, math.ceil(kv_seq / PAGE_SIZE)), -1, dtype=torch.int32)
        packed_storage = torch.zeros((page_count, PAGE_SIZE, 1, packed.shape[-1]), dtype=torch.int8)
        value_storage = torch.zeros((page_count, PAGE_SIZE, 1, NOPE_DIM), dtype=torch.int8)
        page_index = 0
        for batch_index, pages in enumerate(pages_per_sequence):
            for logical_page in range(pages):
                physical_page = int(physical_pages[page_index])
                block_table[batch_index, logical_page] = physical_page
                begin = logical_page * PAGE_SIZE
                end = min(begin + PAGE_SIZE, case.kv_lengths[batch_index])
                packed_storage[physical_page, : end - begin] = packed[batch_index, begin:end]
                value_storage[physical_page, : end - begin] = key[batch_index, begin:end]
                page_index += 1
    else:
        packed_storage = torch.cat([packed[b, :length] for b, length in enumerate(case.kv_lengths)])
        value_storage = torch.cat([key[b, :length] for b, length in enumerate(case.kv_lengths)])

    query_ends = tuple(accumulate(case.query_lengths)) if case.query_layout == "TND" else case.query_lengths
    kv_ends = tuple(accumulate(case.kv_lengths)) if case.kv_layout == "TND" else case.kv_lengths
    return {
        "query": _stored_query(torch.cat((query, query_rope), dim=-1), case),
        "key": packed_storage,
        "value": value_storage,
        "sparse_indices": _stored_query(sparse, case),
        "block_table": block_table,
        "actual_seq_lengths_query": torch.tensor(query_ends, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor(kv_ends, dtype=torch.int32),
        "scale_value": SCALE_VALUE,
        "sparse_block_size": case.sparse_block_size,
        "layout_query": case.query_layout,
        "layout_kv": case.kv_layout,
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": QUANT_TILE_SIZE,
        "rope_head_dim": rope_dim,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "cpu": {"query": query, "key": key, "scales": scales, "sparse_indices": sparse},
    }


def _to_npu(inputs):
    return {name: value.npu() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}


def _pad_zero_rope(inputs):
    padded = dict(inputs)
    query = inputs["query"]
    key = inputs["key"]
    padded["query"] = torch.cat((query, query.new_zeros((*query.shape[:-1], ROPE_DIM))), dim=-1)
    padded["key"] = torch.cat(
        (key[..., :NOPE_DIM], key.new_zeros((*key.shape[:-1], ROPE_DIM * 2)), key[..., NOPE_DIM:]), dim=-1
    )
    padded["rope_head_dim"] = ROPE_DIM
    assert torch.equal(padded["key"][..., NOPE_DIM + ROPE_DIM * 2 :], key[..., NOPE_DIM:])
    return padded


def _uniform_reference(inputs, case):
    """Uniform attention is the selected V mean; LSE is log(valid token count)."""
    cpu = inputs["cpu"]
    query = cpu["query"]
    assert torch.all(query[..., 0] != 0)
    assert torch.count_nonzero(query[..., 1:]) == 0
    assert torch.count_nonzero(cpu["key"][..., 0]) == 0
    values = cpu["key"].double() * cpu["scales"].double().repeat_interleave(QUANT_TILE_SIZE, dim=-1)
    assert torch.equal(values, values.to(query.dtype).double())
    expected = torch.zeros_like(query, dtype=torch.float64)
    counts = torch.zeros(query.shape[:-1], dtype=torch.float32)
    for batch_index, query_length in enumerate(case.query_lengths):
        for query_index in range(query_length):
            threshold = case.kv_lengths[batch_index] - query_length + query_index + 1
            token_ids: list[int] = []
            for block_id in cpu["sparse_indices"][batch_index, query_index, 0].tolist():
                if block_id < 0:
                    break
                begin = block_id * case.sparse_block_size
                token_ids.extend(range(begin, min(begin + case.sparse_block_size, threshold)))
            expected[batch_index, query_index] = values[batch_index, token_ids, 0].sum(dim=0) / len(token_ids)
            counts[batch_index, query_index] = len(token_ids)
    expected = _stored_query(expected.to(query.dtype), case)
    counts = _stored_query(counts, case)
    # N2=1: BSND LSE is [B,1,S,N], TND LSE is [1,T,N].
    counts = counts.unsqueeze(0) if case.query_layout == "TND" else counts.unsqueeze(1)
    return expected, counts


def _check_outputs(outputs, inputs, case, return_lse):
    attention, maximum, denominator = outputs
    output_shape = (*inputs["query"].shape[:-1], NOPE_DIM)
    assert attention.shape == output_shape
    assert attention.dtype == inputs["query"].dtype
    assert torch.isfinite(attention).all()
    assert maximum.dtype == denominator.dtype == torch.float32
    if not return_lse:
        assert maximum.numel() == denominator.numel() == 0
        return
    lse_shape: tuple[int, ...]
    if case.query_layout == "TND":
        lse_shape = (1, sum(case.query_lengths), case.heads)
    else:
        lse_shape = (len(case.query_lengths), 1, max(case.query_lengths), case.heads)
    assert maximum.shape == denominator.shape == lse_shape
    valid = _valid_lse_rows(case).to(maximum.device)
    assert torch.isfinite(maximum[valid]).all()
    assert torch.isfinite(denominator[valid]).all()


def _valid_lse_rows(case):
    if case.query_layout == "TND":
        return torch.ones((1, sum(case.query_lengths), case.heads), dtype=torch.bool)
    valid = torch.zeros((len(case.query_lengths), 1, max(case.query_lengths), case.heads), dtype=torch.bool)
    for batch_index, length in enumerate(case.query_lengths):
        valid[batch_index, 0, :length] = True
    return valid


def _assert_outputs_equal(actual, expected, case, return_lse):
    assert torch.equal(actual[0], expected[0])
    if return_lse:
        valid = _valid_lse_rows(case)
        for observed, control in zip(actual[1:], expected[1:]):
            # Padding query rows do not have a specified LSE value.
            assert torch.equal(observed.cpu()[valid], control.cpu()[valid])
    else:
        assert actual[1].numel() == actual[2].numel() == expected[1].numel() == expected[2].numel() == 0


def _check_uniform(outputs, expected, counts, return_lse):
    attention, maximum, denominator = (tensor.cpu() for tensor in outputs)
    atol, rtol = (BF16_ATOL, BF16_RTOL) if attention.dtype == torch.bfloat16 else (FP16_ATOL, FP16_RTOL)
    torch.testing.assert_close(attention.float(), expected.float(), atol=atol, rtol=rtol)
    if return_lse:
        valid = counts > 0
        # Only valid rows have a specified LSE. The maximum is zero and exp-sum is an integer.
        torch.testing.assert_close(maximum[valid], torch.zeros_like(maximum[valid]), atol=0, rtol=0)
        torch.testing.assert_close(denominator[valid], counts[valid], atol=0, rtol=0)
        lse = maximum[valid] + denominator[valid].log()
        torch.testing.assert_close(lse, counts[valid].log(), atol=0, rtol=0)
