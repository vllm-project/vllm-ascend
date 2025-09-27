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
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul


class AscendQuickGELU(QuickGELU):

    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        import torch_npu

        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        from vllm_ascend.utils import is_310p

        torch.ops.vllm.maybe_prefetch_mlp_down_proj(x)
        if is_310p():
            out = torch_npu.npu_swiglu(x.to(torch.float32)).to(torch.float16)
        else:
            out = torch_npu.npu_swiglu(x)
        torch.ops.vllm.maybe_wait_prefetch_done(out)
        return out
