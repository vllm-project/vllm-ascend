#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
from abc import ABC, abstractmethod
import torch
import torch_npu


class MatcherCustomOp(ABC):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        
    @abstractmethod
    def forward(self, *args, **kws):
        pass
        
    def __call__(self, *args, **kws):
        return self.forward(*args, **kws)
    

class MatcherAscendRMSNorm(MatcherCustomOp):

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        x, residual = torch_npu.npu_rms_norm(
            input, weight, self.epsilon
        )
        return x
    
       
class MatcherAscendRMSNormWithBias(MatcherCustomOp):
    
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        x, residual = torch_npu.npu_rms_norm(
            input, weight, self.epsilon
        )
        x.add_(bias)
        return x

