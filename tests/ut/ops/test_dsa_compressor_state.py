# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.dsa_compressor import (
    can_use_triton_compressor_state_gather,
    triton_compressor_state_gather,
)


def test_state_gather_accepts_page_padded_layout() -> None:
    storage = torch.empty(80, dtype=torch.float32)
    state = torch.as_strided(
        storage,
        size=(4, 2, 8),
        stride=(20, 8, 1),
    )
    block_indices = torch.tensor([1, 1, 3], dtype=torch.int64)
    offset_indices = torch.tensor([0, 1, 0], dtype=torch.int64)
    output = torch.empty((3, 8), dtype=torch.float32)

    assert (
        can_use_triton_compressor_state_gather(
            state,
            block_indices,
            offset_indices,
            output,
        )
        is HAS_TRITON
    )


@pytest.mark.parametrize(
    "block_dtype,offset_dtype",
    [
        (torch.float32, torch.int64),
        (torch.int64, torch.float32),
    ],
)
def test_state_gather_rejects_non_integer_indices(
    block_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> None:
    state = torch.empty((4, 2, 8), dtype=torch.float32)
    block_indices = torch.zeros(3, dtype=block_dtype)
    offset_indices = torch.zeros(3, dtype=offset_dtype)
    output = torch.empty((3, 8), dtype=torch.float32)

    assert not can_use_triton_compressor_state_gather(
        state,
        block_indices,
        offset_indices,
        output,
    )


def test_state_gather_rejects_mismatched_output() -> None:
    state = torch.empty((4, 2, 8), dtype=torch.float32)
    block_indices = torch.zeros(3, dtype=torch.int64)
    offset_indices = torch.zeros(3, dtype=torch.int64)
    output = torch.empty((3, 4), dtype=torch.float32)

    assert not can_use_triton_compressor_state_gather(
        state,
        block_indices,
        offset_indices,
        output,
    )


def test_state_gather_rejects_non_contiguous_state_rows() -> None:
    state = torch.empty((4, 2, 16), dtype=torch.float32)[..., ::2]
    block_indices = torch.zeros(3, dtype=torch.int64)
    offset_indices = torch.zeros(3, dtype=torch.int64)
    output = torch.empty((3, 8), dtype=torch.float32)

    assert not can_use_triton_compressor_state_gather(
        state,
        block_indices,
        offset_indices,
        output,
    )


def test_state_gather_accepts_unpadded_cache() -> None:
    state = torch.empty((4, 2, 8), dtype=torch.float32)
    block_indices = torch.zeros(3, dtype=torch.int64)
    offset_indices = torch.zeros(3, dtype=torch.int64)
    output = torch.empty((3, 8), dtype=torch.float32)

    assert (
        can_use_triton_compressor_state_gather(
            state,
            block_indices,
            offset_indices,
            output,
        )
        is HAS_TRITON
    )


def test_state_gather_rejects_i32_offset_overflow() -> None:
    state = torch.empty_strided(
        (2, 2, 8),
        ((1 << 31), 8, 1),
        dtype=torch.float32,
        device="meta",
    )
    block_indices = torch.zeros(3, dtype=torch.int64)
    offset_indices = torch.zeros(3, dtype=torch.int64)
    output = torch.empty((3, 8), dtype=torch.float32)

    assert not can_use_triton_compressor_state_gather(
        state,
        block_indices,
        offset_indices,
        output,
    )


@pytest.mark.skipif(not HAS_TRITON, reason="triton unavailable on host")
def test_state_gather_launches_triton_kernel() -> None:
    state = torch.empty((4, 2, 8), dtype=torch.float32)
    block_indices = torch.tensor([1, 3], dtype=torch.int64)
    offset_indices = torch.tensor([0, 1], dtype=torch.int64)
    output = torch.empty((2, 8), dtype=torch.float32)

    with patch(
        "vllm_ascend.ops.triton.dsa_compressor._compressor_state_gather_kernel",
    ) as kernel:
        triton_compressor_state_gather(
            state,
            block_indices,
            offset_indices,
            output,
        )

    kernel.__getitem__.assert_called_once_with((2,))
    assert kernel.__getitem__.return_value.call_count == 1
