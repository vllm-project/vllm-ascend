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

from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)

from vllm_ascend.platform import CUSTOM_OP_ENABLED


def rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    # adopt custom kernel path for rotary_embedding
    if CUSTOM_OP_ENABLED and self.is_neox_style:
        return torch.ops._C.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
    if offsets is not None:
        raise NotImplementedError(
            "Batched rotary embedding is currently not supported on NPU.")
    else:
        # TODO: Remove the contiguous in the future.
        query_shape, key_shape = query.shape, key.shape
        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(key.shape[0], -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
    return query.view(query_shape), key.view(key_shape)


RotaryEmbedding.forward_oot = rope_forward_oot
DeepseekScalingRotaryEmbedding.forward = rope_forward_oot
