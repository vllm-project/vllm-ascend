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
from vllm.attention import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.forward_context import ForwardContext, get_forward_context


def attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    # NOTE: please avoid accessing `kv_cache` and `attn_metadata` arguments
    # directly, use `self.kv_cache` and
    # `get_forward_context().attn_metadata` instead.
    if self.use_output:
        output = torch.empty_like(query)
        hidden_size = query.size(-1)
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size)
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            self.impl.forward(self,
                              query,
                              key,
                              value,
                              self_kv_cache,
                              attn_metadata,
                              output=output)
        else:
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)
    else:
        if self.use_direct_call:
            forward_context = get_forward_context()
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            return self.impl.forward(self, query, key, value, self_kv_cache,
                                     attn_metadata)
        else:
            return torch.ops.vllm.unified_attention(query, key, value,
                                                    self.layer_name)


Attention.forward = attention_forward