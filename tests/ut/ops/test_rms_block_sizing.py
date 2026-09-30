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
compile. Flooring to a power of two keeps the key set at {1, 2, 4, 8, 16}
instead of all sixteen integers, and leftover rows stay masked either way.
"""

import pytest

from vllm_ascend.ops.triton.rms_norm import _rms_block_m


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
        (4096, 8, 16),
        (3, 1, 2),
        (16, 1, 16),
    ],
)
def test_block_m_floors_to_power_of_two(total_batch, num_vectorcore, expected):
    assert _rms_block_m(total_batch, num_vectorcore) == expected


@pytest.mark.parametrize("total_batch", [1, 7, 15, 63, 257, 1024])
def test_block_m_is_always_a_valid_constexpr(total_batch):
    block_m = _rms_block_m(total_batch, 8)

    assert block_m in (1, 2, 4, 8, 16)
    assert block_m & (block_m - 1) == 0


def test_block_m_never_returns_zero():
    """cdiv can be 0 for an empty batch; the kernel still needs BLOCK_M >= 1."""
    assert _rms_block_m(0, 8) == 1


def test_block_m_is_capped_at_row_block_size():
    assert _rms_block_m(10**6, 1) == 16
