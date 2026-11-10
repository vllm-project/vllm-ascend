#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for ``_rms_block_m``.

``BLOCK_M`` is a Triton constexpr, so every distinct value is one JIT
compile. ``_rms_block_m`` derives the per-core row budget adaptively from the
runtime Unified Buffer (UB) size, the hidden ``dim`` and the element width
(``dtype``), then floors ``BLOCK_M`` to a power of two; leftover rows stay
masked either way.
"""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.rms_norm import _rms_block_m

# 192 KB, matching ``get_ub_size_bytes()``'s documented fallback for 910B/A3.
_UB_FALLBACK_BYTES = 192 * 1024


def _row_block_size(dim, dtype):
    """Mirror of the adaptive row budget inside ``_rms_block_m``."""
    element_size = torch.empty(1, dtype=dtype).element_size()
    data_multiplier = 5 if element_size == 4 else 7
    return int((_UB_FALLBACK_BYTES - 6144) / (dim * element_size * data_multiplier))


def _block_m(total_batch, num_vectorcore, dim, dtype):
    with patch(
        "vllm_ascend.ops.triton.rms_norm.get_ub_size_bytes",
        return_value=_UB_FALLBACK_BYTES,
    ):
        return _rms_block_m(total_batch, num_vectorcore, dim, dtype)


# (dim, dtype, row_block_size, capped BLOCK_M): different hidden dims / dtypes
# map to different row budgets and therefore different BLOCK_M caps. fp16 uses
# the same 2-byte element size as bf16 and is not repeated here.
_DIM_DTYPE_CASES = [
    (64, torch.bfloat16, 212, 128),
    (128, torch.bfloat16, 106, 64),
    (256, torch.bfloat16, 53, 32),
    (512, torch.bfloat16, 26, 16),
    (1024, torch.bfloat16, 13, 8),
    (2048, torch.bfloat16, 6, 4),
    (64, torch.float32, 148, 128),
    (128, torch.float32, 74, 64),
    (256, torch.float32, 37, 32),
    (512, torch.float32, 18, 16),
    (1024, torch.float32, 9, 8),
    (2048, torch.float32, 4, 4),
]


@pytest.mark.parametrize(("dim", "dtype", "row_block_size", "expected"), _DIM_DTYPE_CASES)
def test_block_m_caps_at_adaptive_row_block_size(dim, dtype, row_block_size, expected):
    """A batch larger than any core can hold caps BLOCK_M at the top
    power-of-two of the adaptive row budget."""
    assert _row_block_size(dim, dtype) == row_block_size
    assert _block_m(10**6, 1, dim, dtype) == expected


@pytest.mark.parametrize(
    ("total_batch", "num_vectorcore", "expected"),
    [
        (1, 8, 1),
        (8, 8, 1),
        (9, 8, 2),
        (16, 8, 2),
        (24, 8, 2),
        (32, 8, 4),
        (48, 8, 4),
        (64, 8, 8),
        (128, 8, 16),
        (4096, 8, 64),
        (3, 1, 2),
        (16, 1, 16),
    ],
)
def test_block_m_floors_to_power_of_two(total_batch, num_vectorcore, expected):
    # dim=128 / bf16 gives row_block_size=106, so the cap is 64.
    assert _block_m(total_batch, num_vectorcore, 128, torch.bfloat16) == expected


@pytest.mark.parametrize("total_batch", [1, 7, 15, 63, 257, 1024])
def test_block_m_is_always_a_valid_constexpr(total_batch):
    block_m = _block_m(total_batch, 8, 128, torch.bfloat16)

    assert block_m & (block_m - 1) == 0
    assert 1 <= block_m <= _row_block_size(128, torch.bfloat16)


def test_block_m_never_returns_zero():
    """cdiv can be 0 for an empty batch; the kernel still needs BLOCK_M >= 1."""
    assert _block_m(0, 8, 128, torch.bfloat16) == 1


def test_block_m_is_capped_at_row_block_size():
    assert _block_m(10**6, 1, 128, torch.bfloat16) == 64
    assert _block_m(10**6, 1, 2048, torch.bfloat16) == 4


def test_block_m_scales_down_for_large_dim():
    """Larger hidden dims must shrink the tile to stay within UB."""
    assert _row_block_size(128, torch.bfloat16) > _row_block_size(2048, torch.bfloat16)
    assert _block_m(10**6, 1, 128, torch.bfloat16) > _block_m(10**6, 1, 2048, torch.bfloat16)
