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
#

from typing import List, Optional

import torch
import vllm
from vllm.distributed.parallel_state import GroupCoordinator


class GroupCoordinatorPatch(GroupCoordinator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1,
                   scatter_sizes: Optional[List[int]] = None,
                   gather_sizes: Optional[List[int]] = None) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        return self.device_communicator.all_to_all(input_, scatter_dim,
                                                   gather_dim, scatter_sizes,
                                                   gather_sizes)


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch  # Note: check the GroupCoordinator with online serving