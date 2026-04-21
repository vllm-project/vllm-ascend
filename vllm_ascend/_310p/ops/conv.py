#
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
#

import torch

from vllm_ascend.ops.conv import AscendConv3dLayer


class AscendConv3dLayer310(AscendConv3dLayer):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # 310P should avoid the aclnn BatchMatMulV2 Conv3D path used by
        # AscendConv3dLayer and keep vLLM's native Conv3d dispatch behavior.
        return super().forward_native(x)
