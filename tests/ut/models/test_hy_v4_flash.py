# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from torch import nn

pytest.importorskip("vllm.models.hy_v4_flash", exc_type=ModuleNotFoundError)


@pytest.mark.parametrize("fp8", [False, True])
def test_engram_lookup_respects_local_shard_and_invalid_rows(fp8):
    from vllm_ascend.models.hy4_flash.model import AscendFlashEmbedding

    table = AscendFlashEmbedding.__new__(AscendFlashEmbedding)
    nn.Module.__init__(table)
    table.head_start, table.head_end = 1, 3
    table.vocab_start = 100
    table.fp8 = fp8
    values = torch.tensor([[2, 4], [1, 3]], dtype=torch.bfloat16)
    table.embedding = nn.Module()
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table.embedding.weight = nn.Parameter(values.to(dtype), requires_grad=False)
    factors = torch.tensor([0.5, 2.0], dtype=torch.bfloat16)
    if fp8:
        table.embedding.weight_scale_inv = nn.Parameter(factors, requires_grad=False)
    hashes = torch.tensor([[7, 100, 101, 3], [8, -1, 102, 4]], dtype=torch.int32)
    expected = torch.cat((values[None].float(), torch.zeros(1, 2, 2)))
    if fp8:
        expected *= factors[None, :, None]
    torch.testing.assert_close(table(hashes), expected)


def test_long_flash_swa_indices_keep_only_the_causal_window():
    from vllm_ascend.models.hy4_flash.long_context import _swa_indices

    assert _swa_indices(torch.tensor([0, 2, 3, 10]), 4).tolist() == [
        [0, -1, -1, -1],
        [0, 1, 2, -1],
        [0, 1, 2, 3],
        [7, 8, 9, 10],
    ]


def test_long_flash_paged_rows_with_remapped_pages():
    from vllm_ascend.models.hy4_flash.long_context import _cache_rows, _write_cache_rows

    cache = torch.zeros(4, 2, 1, 3)
    positions = torch.tensor([0, 1, 2, 3, 4])
    block_table = torch.tensor([3, 1, 0])
    slots = block_table[positions // 2] * 2 + positions % 2
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    _write_cache_rows(cache, slots, values)
    torch.testing.assert_close(_cache_rows(cache, block_table, positions), values)
    torch.testing.assert_close(cache[2], torch.zeros_like(cache[2]))


def test_long_flash_indexpool_ranks_groups_and_unconditionally_includes_tail():
    from types import SimpleNamespace

    from vllm_ascend.models.hy4_flash.long_context import AscendHYV4FlashIndexPool

    indexer = SimpleNamespace(num_heads=1, group_budget=2, ratio=4)
    query = torch.ones((2, 1, 128), dtype=torch.bfloat16)
    weights = torch.ones((2, 1), dtype=torch.float32)
    keys = torch.arange(1, 6, dtype=torch.bfloat16)[:, None].expand(-1, 128)
    output = AscendHYV4FlashIndexPool._select_indices(indexer, query, weights, keys, torch.tensor([1, 18]))
    assert output[0].tolist() == [0, 1] + [-1] * 10
    assert output[1].tolist() == list(range(8, 19)) + [-1]


def test_long_flash_indexpool_completes_group_across_prefill_chunks():
    from types import SimpleNamespace

    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import gated_pool

    from vllm_ascend.models.hy4_flash.long_context import (
        AscendHYV4FlashIndexPool,
        _cache_rows,
        _quantized_vectors,
        _write_cache_rows,
    )

    raw = torch.zeros(4, 2, 1, 128, dtype=torch.bfloat16)
    pool = torch.zeros_like(raw)
    block_table = torch.tensor([3, 1, 0, 2])
    positions = torch.arange(8)
    slots = block_table[positions // 2] * 2 + positions % 2
    keys = torch.arange(1, 9, dtype=torch.bfloat16)[:, None].expand(-1, 128)
    _write_cache_rows(raw, slots, keys)
    indexer = SimpleNamespace(
        ratio=4,
        key_value_projection=lambda x: x,
        key_gate_projection=lambda x: x,
        intra_group_ape=torch.zeros(4, 128),
        key_norm=SimpleNamespace(weight=torch.ones(128)),
        eps=1e-6,
        rope_dim=0,
        rope_parameters={},
    )
    AscendHYV4FlashIndexPool._write_completed_groups(
        indexer, raw, pool, block_table[None], [(0, 0, 4, 3, 7)], raw.device
    )
    assert torch.count_nonzero(_cache_rows(pool, block_table, torch.tensor([1]))) == 0
    AscendHYV4FlashIndexPool._write_completed_groups(
        indexer, raw, pool, block_table[None], [(0, 0, 1, 7, 8)], raw.device
    )
    groups = keys.view(2, 4, 128).float()
    expected = gated_pool(groups, groups, torch.zeros(4, 128), torch.ones(128), 1e-6).bfloat16()
    torch.testing.assert_close(_cache_rows(pool, block_table, torch.arange(2)), _quantized_vectors(expected))
