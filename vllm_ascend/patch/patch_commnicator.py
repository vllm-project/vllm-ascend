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
# This file is used to monkey patch communicator in vllm to support ascend.
# Remove this file when vllm support by
# https://github.com/vllm-project/vllm/pull/11324.

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.utils import resolve_obj_by_qualname


class GroupCoordinatorPatch(GroupCoordinator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from vllm.platforms import current_platform
        device_comm_cls = resolve_obj_by_qualname(
            current_platform.get_device_communicator_cls())
        # we have checked and ensure that reusing tpu tag here is fine.
        use_custom_device = kwargs.get("use_tpu_communicator", False)
        if use_custom_device and self.world_size > 1:
            self.communicator = device_comm_cls(group=self.device_group,
                                                unique_name=self.unique_name)

    def all_reduce(self, input_):
        # Bypass the function if we are using only 1 device.
        if self.world_size == 1:
            return input_

        return self.communicator.all_reduce(input_)

    def gather(self, input_, dst=0, dim=-1):
        # Bypass the function if we are using only 1 device.
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        return self.communicator.gather(input_, dst, dim)

    def all_gather(self, input_, dim=-1):
        # Bypass the function if we are using only 1 device.
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        return self.communicator.all_gather(input_, dim)


GroupCoordinator = GroupCoordinatorPatch
