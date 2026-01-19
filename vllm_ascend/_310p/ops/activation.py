#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
import torch.nn.functional as F

from vllm_ascend.ops.activation import AscendSiluAndMul as _Base


class AscendSiluAndMul310(_Base):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.ops.vllm.maybe_prefetch_mlp_down_proj(x)
        h = x.shape[-1] // 2
        out = (F.silu(x[..., :h].to(torch.float32)) * x[..., h:].to(torch.float32)).to(torch.float16)
        torch.ops.vllm.maybe_wait_prefetch_done(out)
        return out
