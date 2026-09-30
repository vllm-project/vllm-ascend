# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm_ascend.ops.triton.pcp_kv_cache import _get_pcp_kv_cache_rows


@pytest.mark.parametrize(
    "num_tokens,block_cols,element_size,num_caches,ub_kib,expected",
    [
        (32, 1024, 1, 1, 192, 1),  # Keep small batches spread across cores.
        (128, 1024, 1, 1, 192, 2),
        (4096, 256, 2, 1, 192, 8),  # Retain the maximum when the tile fits.
        (4096, 1024, 1, 1, 192, 4),  # C8 includes address and cast temporaries.
        (4096, 512, 2, 2, 192, 4),  # Budget both latent and RoPE cache tiles.
        (4096, 1024, 1, 1, 128, 2),  # Smaller devices reduce row batching.
        (4096, 2048, 1, 1, 192, 2),  # Wider rows reduce row batching.
        (4096, 1024, 4, 1, 192, 2),  # Account for larger element sizes.
        (4096, 1024, 1, 1, 160, 2),  # Round down when the budget fits three rows.
        (4096, 8192, 1, 1, 192, 1),  # Preserve the baseline for very wide rows.
    ],
)
def test_pcp_kv_row_budget(num_tokens, block_cols, element_size, num_caches, ub_kib, expected):
    with patch("vllm_ascend.ops.triton.pcp_kv_cache.get_ub_size_bytes", return_value=ub_kib * 1024):
        assert _get_pcp_kv_cache_rows(num_tokens, 48, block_cols, element_size, num_caches) == expected
