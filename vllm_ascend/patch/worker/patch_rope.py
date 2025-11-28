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

import torch
import torch.nn as nn
from vllm.model_executor.layers.rotary_embedding.base import \
    RotaryEmbeddingBase


class AscendRotaryEmbeddingBase(nn.Module):

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin


# NOTE: These will be removed after vllm-ascend is aligned with vllm latest main.
RotaryEmbeddingBase.get_cos_sin = AscendRotaryEmbeddingBase.get_cos_sin
