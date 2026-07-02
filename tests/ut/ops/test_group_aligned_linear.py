#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import pytest

from vllm_ascend.ops.group_aligned_linear import group_aligned_partition


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@pytest.mark.parametrize(
    "total, tp_size, group_size, expected_elems, expected_groups",
    [
        # Qwen3-VL vision MLP fc2: 4304 is not a multiple of 64 -> uneven split.
        (4304, 2, 32, [2176, 2128], [68, 67]),
        # attn.proj / aligned dims -> reduces to the plain even split.
        (1152, 2, 32, [576, 576], [18, 18]),
        # merger fc2 -> aligned, even split (each rank owns 72 of the 144 groups).
        (4608, 2, 32, [2304, 2304], [72, 72]),
        # tp=1 -> the full tensor.
        (4304, 1, 32, [4304], [135]),
        # tp=4 over an unaligned dim.
        (4304, 4, 32, [1088, 1088, 1088, 1040], [34, 34, 34, 33]),
    ],
)
def test_group_aligned_partition_expected(total, tp_size, group_size, expected_elems, expected_groups):
    elem_sizes, group_sizes = group_aligned_partition(total, tp_size, group_size)
    assert elem_sizes == expected_elems
    assert group_sizes == expected_groups


# Realistic (non-degenerate) dims: num_groups >= tp_size so every rank owns at
# least one group. The degenerate "more ranks than groups" case never happens
# for a real ViT MLP and is intentionally out of scope.
@pytest.mark.parametrize("total", [4304, 1152, 4608, 4096, 2048, 8192])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_group_aligned_partition_invariants(total, tp_size, group_size):
    elem_sizes, group_sizes = group_aligned_partition(total, tp_size, group_size)

    # Shapes line up with the rank count.
    assert len(elem_sizes) == tp_size
    assert len(group_sizes) == tp_size

    # Partitions tile the whole tensor and the whole group axis.
    assert sum(elem_sizes) == total
    assert sum(group_sizes) == _cdiv(total, group_size)

    # No negative sizes.
    assert all(e >= 0 for e in elem_sizes)
    assert all(g >= 0 for g in group_sizes)

    # The key invariant: the per-rank scale dim that create_weights derives via
    # ceil(input_size_per_partition / group_size) must equal the group count we
    # slice the checkpoint scale by -- otherwise weight and scale would diverge.
    for e, g in zip(elem_sizes, group_sizes):
        assert _cdiv(e, group_size) == g

    # Every boundary but the last lands on a group boundary (multiple of
    # group_size), so no MX group is cut across ranks.
    boundary = 0
    for e in elem_sizes[:-1]:
        boundary += e
        assert boundary % group_size == 0
