# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the DSV4.1 indexer warmup wrappers."""

from unittest.mock import patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config
from vllm_ascend.ops.triton import prepare_indexer_indices as pi

# The registration gates and key spaces under test are only meaningful
# with a real Triton kernel behind them; the without-device CI leg has no
# triton installed.
pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")


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
