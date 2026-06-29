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
"""Group-aligned tensor-parallel partitioning for MX-quantized weights.

MX (microscaling) quantization groups every ``group_size`` (e.g. 32) elements
along the input dimension and stores one shared scale per group. Splitting such
a weight across tensor-parallel ranks is only numerically valid if every
partition boundary lands on a group boundary -- otherwise a single group is cut
across two ranks and its scale can no longer be applied correctly.

Plain even sharding (``size // tp_size``) breaks this when the sharded
dimension is not a multiple of ``group_size * tp_size``. For the Qwen3-VL vision
MLP the intermediate size is ``4304`` which is not a multiple of ``64`` (tp=2),
so the even split ``2152`` falls inside group #67 and the weight_scale load
fails with a ``narrow ... exceeds dimension size`` error.

:func:`group_aligned_partition` instead splits the sharded dimension on **group
boundaries**: groups are distributed as evenly as possible (earlier ranks get
the +1) and the last rank absorbs the trailing partial group. This yields an
uneven but group-aligned partition (e.g. tp=2 -> ``[2176, 2128]`` elements /
``[68, 67]`` groups for 4304) that is correct for the row all-reduce. It is
consumed by ``AscendLinearMethod.create_weights`` (see ``method_adapters.py``).
"""


def group_aligned_partition(total: int, tp_size: int, group_size: int) -> tuple[list[int], list[int]]:
    """Split ``total`` elements (along the sharded dim) across ``tp_size`` ranks
    on ``group_size`` boundaries so no MX group straddles a partition boundary.

    Returns ``(elem_sizes, group_sizes)`` -- per-rank element counts and group
    counts. Groups are distributed as evenly as possible (earlier ranks get the
    +1); the last rank absorbs the trailing partial group, if any.

    The invariant ``ceil(elem_sizes[r] / group_size) == group_sizes[r]`` holds
    for every rank, which keeps the loaded weight (split by elements) consistent
    with the loaded weight_scale (split by groups). For dimensions that are
    already a multiple of ``group_size * tp_size`` the result equals the plain
    even split; for ``tp_size == 1`` it is the full tensor.
    """

    def _cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    if tp_size <= 1:
        return [total], [_cdiv(total, group_size)]

    num_groups = _cdiv(total, group_size)
    base, rem = divmod(num_groups, tp_size)
    group_sizes = [base + (1 if r < rem else 0) for r in range(tp_size)]

    # Each rank owns a contiguous span of groups; its element count is that span
    # clamped to ``total`` so the trailing partial group stays intact and no rank
    # goes negative even in the degenerate ``num_groups < tp_size`` case.
    elem_sizes: list[int] = []
    groups_done = 0
    for r in range(tp_size):
        start = groups_done * group_size
        groups_done += group_sizes[r]
        end = min(groups_done * group_size, total)
        elem_sizes.append(max(0, end - start))
    return elem_sizes, group_sizes
