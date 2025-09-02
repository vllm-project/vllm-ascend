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
from vllm.forward_context import get_forward_context


class AscendQuickGELU(QuickGELU):

    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        import torch_npu

        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    def prefetch_down_proj(self,
                           dependency: torch.Tensor):
        import torch_npu
        forward_context = get_forward_context()
        prefetch_model = forward_context.prefetch_model
        prefetch_stream = forward_context.prefetch_stream
        layer_idx = forward_context.layer_idx

        prefetch_stream.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(prefetch_stream):
            MLP_DOWN_PREFETCH_SIZE = 6 * 1024 * 1024
            torch_npu.npu_prefetch(prefetch_model.model.layers[layer_idx].mlp.down_proj.weight, \
                                dependency, MLP_DOWN_PREFETCH_SIZE)
            forward_context.layer_idx += 1

    def wait_prefetch_done(self):
        forward_context = get_forward_context()
        prefetch_stream = forward_context.prefetch_stream
        torch.npu.current_stream().wait_stream(prefetch_stream)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        from vllm_ascend.utils import is_310p

        if is_310p():
            out = torch_npu.npu_swiglu(x.to(torch.float32)).to(torch.float16)
        else:
            dependency = x
            self.prefetch_down_proj(dependency)

            out = torch_npu.npu_swiglu(x)

            self.wait_prefetch_done()
        return out
