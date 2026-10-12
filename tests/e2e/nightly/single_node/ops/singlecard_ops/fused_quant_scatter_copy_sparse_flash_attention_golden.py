# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal independent C8 fixture and CPU golden implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
PACKED_DIM = 656
TOPK = 2048
COPY_CAPACITY = 32768
MAX_CACHE_TOKENS = 32760
POISON = -91


@dataclass
class TestData:
    query: torch.Tensor
    initial_hbm: torch.Tensor
    expected_hbm: torch.Tensor
    dram: torch.Tensor
    hbm_block_table: torch.Tensor
    dram_block_table: torch.Tensor
    topk_dst_slots: torch.Tensor
    topk_src_ids: torch.Tensor
    topk_miss_counts: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    num_cache_tokens: torch.Tensor
    miss_src_ids: torch.Tensor
    miss_dst_slots: torch.Tensor
    miss_counts: torch.Tensor
    attention_golden: torch.Tensor
    scale_value: float


def physical_rows(table: torch.Tensor, logical_tokens: torch.Tensor) -> torch.Tensor:
    logical_tokens = logical_tokens.to(torch.int64)
    return table[0, logical_tokens // BLOCK_SIZE].to(torch.int64) * BLOCK_SIZE + logical_tokens.remainder(BLOCK_SIZE)


def make_packed_bytes(blocks: int, generator: torch.Generator) -> torch.Tensor:
    nope = (
        torch.randint(
            -3,
            4,
            (blocks, BLOCK_SIZE, 1, NOPE_DIM),
            generator=generator,
            dtype=torch.int16,
        )
        .float()
        .to(torch.float8_e4m3fn)
    )
    rope = (
        torch.empty((blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32)
        .uniform_(-0.5, 0.5, generator=generator)
        .to(torch.bfloat16)
    )
    scales = torch.empty((blocks, BLOCK_SIZE, 1, NOPE_DIM // BLOCK_SIZE), dtype=torch.float32).uniform_(
        0.02, 0.08, generator=generator
    )
    packed = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed.shape[-1] != PACKED_DIM:
        raise AssertionError("packed C8 KV row must contain 656 bytes")
    return packed.view(torch.int8).contiguous()


def unpack_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = rows.contiguous().view(torch.uint8)
    nope_c8 = raw[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    rope = raw[:, NOPE_DIM : NOPE_DIM + 2 * ROPE_DIM].contiguous().view(torch.bfloat16)
    scales = raw[:, NOPE_DIM + 2 * ROPE_DIM :].contiguous().view(torch.float32)
    value = (nope_c8.float() * scales.repeat_interleave(BLOCK_SIZE, dim=-1)).to(torch.bfloat16).float()
    key = torch.cat((value, rope.float()), dim=-1)
    return key, value


def make_test_data(miss_count: int, cache_tokens: int = TOPK, seed: int = 2026) -> TestData:
    if not 0 <= miss_count <= TOPK:
        raise ValueError("miss_count must be in [0, 2048]")
    if not TOPK <= cache_tokens <= MAX_CACHE_TOKENS:
        raise ValueError("cache_tokens must be in [2048, 32760]")
    generator = torch.Generator().manual_seed(seed + miss_count + cache_tokens)
    dram_blocks = 18
    hbm_blocks = math.ceil(cache_tokens / BLOCK_SIZE)
    source_tokens = dram_blocks * BLOCK_SIZE
    actual_kv_length = max(source_tokens, cache_tokens)
    hbm_block_table = torch.randperm(hbm_blocks, generator=generator).to(torch.int32).view(1, -1)
    dram_block_table = torch.randperm(dram_blocks, generator=generator).to(torch.int32).view(1, -1)
    dram = make_packed_bytes(dram_blocks, generator)
    initial_hbm = torch.full((hbm_blocks, BLOCK_SIZE, 1, PACKED_DIM), POISON, dtype=torch.int8)
    expected_hbm = initial_hbm.clone()

    selected_sources = torch.randperm(source_tokens, generator=generator)[:TOPK].to(torch.int32)
    destination_slots = torch.arange(cache_tokens - TOPK, cache_tokens, dtype=torch.int32)
    if cache_tokens == MAX_CACHE_TOKENS:
        destination_slots = torch.roll(destination_slots, shifts=1)
    dram_rows = dram.view(-1, PACKED_DIM)
    initial_rows = initial_hbm.view(-1, PACKED_DIM)
    expected_rows = expected_hbm.view(-1, PACKED_DIM)
    source_physical = physical_rows(dram_block_table, selected_sources)
    destination_physical = physical_rows(hbm_block_table, destination_slots)
    initial_rows[destination_physical[miss_count:]] = dram_rows[source_physical[miss_count:]]
    expected_rows[destination_physical] = dram_rows[source_physical]

    topk_dst_slots = destination_slots.view(1, 1, TOPK).contiguous()
    topk_src_ids = selected_sources.view(1, 1, TOPK).contiguous()
    miss_src_ids = torch.full((1, 1, COPY_CAPACITY), -1, dtype=torch.int32)
    miss_dst_slots = torch.full_like(miss_src_ids, -1)
    miss_src_ids[0, 0, :miss_count] = selected_sources[:miss_count]
    miss_dst_slots[0, 0, :miss_count] = destination_slots[:miss_count]

    query = torch.randn((1, 8, NOPE_DIM + ROPE_DIM), generator=generator).to(torch.bfloat16)
    scale_value = 1.0 / math.sqrt(NOPE_DIM + ROPE_DIM)
    selected_rows = dram_rows[source_physical]
    key, value = unpack_rows(selected_rows)
    scores = query[0].float() @ key.T * scale_value
    probability = torch.softmax(scores, dim=-1)
    attention_golden = (probability @ value).unsqueeze(0)

    return TestData(
        query=query,
        initial_hbm=initial_hbm,
        expected_hbm=expected_hbm,
        dram=dram,
        hbm_block_table=hbm_block_table,
        dram_block_table=dram_block_table,
        topk_dst_slots=topk_dst_slots,
        topk_src_ids=topk_src_ids,
        topk_miss_counts=torch.tensor([miss_count], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([actual_kv_length], dtype=torch.int32),
        num_cache_tokens=torch.tensor([0 if miss_count == 0 else cache_tokens], dtype=torch.int32),
        miss_src_ids=miss_src_ids,
        miss_dst_slots=miss_dst_slots,
        miss_counts=torch.tensor([miss_count], dtype=torch.int32),
        attention_golden=attention_golden,
        scale_value=scale_value,
    )
