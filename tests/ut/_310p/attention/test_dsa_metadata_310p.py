# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.attention.dsa_v1 import _can_use_uniform_decode
from vllm_ascend.attention.dsa_v1 import (
    build_dspark_swa_indices,
    classify_fresh_prefill,
    classify_uniform_query_len,
)
from vllm_ascend.ops.dsa import filter_exact_metadata


def test_filter_metadata_selects_exact_swa_namespace() -> None:
    expected = object()
    metadata = {
        "model.layers.2.self_attn.attn": object(),
        "model.layers.2.self_attn.compressor.state_cache": object(),
        "model.layers.2.self_attn.indexer.k_cache": object(),
        "model.layers.2.self_attn.swa_cache": expected,
    }

    assert filter_exact_metadata(metadata, "model.layers.2.self_attn.swa_cache") == [expected]


def test_filter_metadata_rejects_missing_or_ambiguous_namespace() -> None:
    with pytest.raises(ValueError, match="Expected exactly one"):
        filter_exact_metadata({}, "model.layers.2.self_attn.swa_cache")

    metadata = {
        "model.layers.2.self_attn.swa_cache": object(),
        "model.layers.2.self_attn.swa_cache.extra": object(),
    }
    with pytest.raises(ValueError, match="got 2"):
        filter_exact_metadata(metadata, "model.layers.2.self_attn.swa_cache")


def test_dspark_swa_indices_decode_hybrid_logical_blocks() -> None:
    # A physical 32-token page is split into 16 logical 2-token entries.
    # Logical IDs 160..223 represent physical blocks 10..13.
    block_table = torch.arange(160, 224, dtype=torch.int64).view(1, 64)
    slots, visible_lens = build_dspark_swa_indices(
        block_table=block_table,
        num_speculative_tokens=5,
        window_size=128,
        block_size=32,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([19], dtype=torch.int32),
        num_decode_tokens=5,
        index_width=133,
        blocks_per_phys_block=16,
    )

    assert visible_lens.tolist() == [19, 19, 19, 19, 19]
    expected = torch.arange(320, 339, dtype=torch.int32)
    for token_row in slots[:, 0]:
        torch.testing.assert_close(token_row[:19], expected)
        assert torch.all(token_row[19:] == -1)


def test_uniform_decode_classification_rejects_mixed_lengths() -> None:
    mixed = torch.tensor([0, 1, 4], dtype=torch.int32)
    uniform = torch.tensor([0, 2, 4], dtype=torch.int32)

    assert classify_uniform_query_len(mixed) is None
    assert classify_uniform_query_len(uniform) == 2
    assert not _can_use_uniform_decode(
        SimpleNamespace(uniform_query_len=None, seq_lens_list=[8, 8]),
        num_query_tokens=4,
    )
    assert _can_use_uniform_decode(
        SimpleNamespace(uniform_query_len=2, seq_lens_list=[8, 8]),
        num_query_tokens=4,
    )


def test_fresh_prefill_classification_uses_cpu_metadata() -> None:
    query_lens = torch.tensor([3, 2], dtype=torch.int32)

    assert classify_fresh_prefill(torch.tensor([3, 2], dtype=torch.int32), query_lens)
    assert not classify_fresh_prefill(torch.tensor([5, 2], dtype=torch.int32), query_lens)
