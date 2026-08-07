# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from enum import IntEnum

import torch
from torch.nn.functional import pad


class GroupListType(IntEnum):
    """Grouped-matmul expert-token metadata layouts."""

    CUMULATIVE = 0
    COUNTS = 1
    SPARSE_COUNTS = 2


def convert_group_list(
    group_list: torch.Tensor,
    src_list_type: int | GroupListType,
    dst_list_type: int | GroupListType,
    active_num: int = 0,
    expert_num: int = 0,
) -> torch.Tensor:
    """Convert a group list at the MLP boundary.

    Kernel implementations must use this helper instead of carrying their
    own cumsum/diff conversions. This keeps the meaning of group-list types
    consistent across GMM1, activation, and GMM2.
    """

    try:
        src_type = GroupListType(src_list_type)
    except ValueError as exc:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}") from exc
    try:
        dst_type = GroupListType(dst_list_type)
    except ValueError as exc:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {dst_list_type}") from exc

    if src_type == dst_type:
        return group_list
    if src_type == GroupListType.COUNTS and dst_type == GroupListType.CUMULATIVE:
        return group_list.cumsum(dim=0)
    if src_type == GroupListType.CUMULATIVE and dst_type == GroupListType.COUNTS:
        group_diff = torch.diff(group_list)
        return torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
    if src_type == GroupListType.SPARSE_COUNTS and dst_type == GroupListType.CUMULATIVE:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumulative_group_list = torch.full(
            size=(expert_num,),
            fill_value=active_num,
            dtype=group_list.dtype,
            device=group_list.device,
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumulative_group_list[start:end] = tokens[i]

        return cumulative_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_type.value} to dst_list_type={dst_type.value} is not implemented yet. "
        "This feature is under development."
    )


__all__ = ["GroupListType", "convert_group_list"]
