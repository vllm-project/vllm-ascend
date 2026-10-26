# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the DSV4.1 indexer warmup wrappers."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config, make_registry
from vllm_ascend.ops.triton import prepare_indexer_indices as pi
from vllm_ascend.ops.triton import quantize_indexer_query as qi


def test_block_rows_formula_capped_by_sort_budget_and_cores():
    # topk=8 -> block_cols=8; sort budget: 128 KiB // (8 * 4 * 8) = 512;
    # core cap: next_pow2(cdiv(8192, 4)) = 2048 -> min = 512.
    assert pi.collect_indexer_warmup_block_rows(8, 4, 8192) == [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    ]
    # A small max_tokens clamps via the core cap instead.
    assert pi.collect_indexer_warmup_block_rows(8, 4, 64) == [1, 2, 4, 8, 16]


def test_keys_cover_both_write_lengths_variants():
    config = make_mock_vllm_config(max_num_batched_tokens=8192)
    with patch.object(pi, "get_vectorcore_num", return_value=4):
        keys = pi._PREPARE_INDEXER_INDICES_KERNEL.get_warmup_keys((8, 4, config))

    assert {key.write_lengths for key in keys} == {False, True}
    # WRITE_LENGTHS bumps BLOCK_COLS to 32, which halves the row budget the
    # runtime enforces: wide-column keys only exist up to that tighter cap.
    base_rows = pi.collect_indexer_warmup_block_rows(8, 4, 8192)
    wl_cap = pi._indexer_max_block_rows(32)
    assert len(keys) == len(base_rows) + len([r for r in base_rows if r <= wl_cap])
    assert max(key.block_rows for key in keys if key.write_lengths) == wl_cap
    for key in keys:
        assert key.topk == 8
        assert key.compress_ratio == 4
        # write_lengths widens BLOCK_COLS to the reduction width.
        assert key.block_cols == (32 if key.write_lengths else 8)
        assert key.positions_dtype == torch.int64


def test_quantize_head_dim_from_hf_text_config():
    config = make_mock_vllm_config(dtype=torch.bfloat16, index_head_dim=192)
    (key,) = qi._QUANTIZE_INDEXER_QUERY_KERNEL.get_warmup_keys(config)
    assert key.dtype == torch.bfloat16
    assert key.head_dim == 192

    config = make_mock_vllm_config(
        dtype=torch.bfloat16,
    )
    config.model_config.hf_text_config = SimpleNamespace()  # attribute missing
    (key,) = qi._QUANTIZE_INDEXER_QUERY_KERNEL.get_warmup_keys(config)
    assert key.head_dim == 128


def test_register_indexer_warmup_window_semantics():
    config = make_mock_vllm_config()
    registry = make_registry(config)

    pi.register_indexer_warmup(8, 4, config)
    assert len(registry) == 0

    with registry.activate():
        pi.register_indexer_warmup(8, 4, config)
        qi._QUANTIZE_INDEXER_QUERY_KERNEL.register_warmup(config)
    assert len(registry) == 2
