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


def fused_qkvzba_split_reshape_cat_pytorch(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """
    PyTorch implementation of fused_qkvzba_split_reshape_cat.
    This is a fallback implementation for 310P without Triton support.
    """
    batch, seq_len = mixed_qkvz.shape[0], 1
    
    # Split mixed_qkvz into components
    # mixed_qkvz shape: [batch, num_heads_qk, head_qk * 2 + num_heads_v // num_heads_qk * head_v * 2]
    value_hidden = (num_heads_v // num_heads_qk) * head_v
    qkvz_dim = head_qk * 2 + value_hidden * 2
    
    # Reshape and extract
    mixed_qkvz_reshaped = mixed_qkvz.view(batch * seq_len, num_heads_qk, qkvz_dim)
    
    query = mixed_qkvz_reshaped[:, :, :head_qk]
    key = mixed_qkvz_reshaped[:, :, head_qk:head_qk*2]
    v_end = head_qk * 2 + value_hidden
    z_end = v_end + value_hidden
    value = mixed_qkvz_reshaped[:, :, head_qk * 2 : v_end]
    z = mixed_qkvz_reshaped[:, :, v_end:z_end]
    
    # Reshape to combine heads
    query = query.reshape(batch * seq_len, -1)
    key = key.reshape(batch * seq_len, -1)
    value = value.reshape(batch * seq_len, -1)
    
    # Concatenate q, k, v
    mixed_qkv = torch.cat([query, key, value], dim=-1)
    
    # Reshape z to [batch * seq_len, num_heads_v, head_v]
    z = z.view(batch * seq_len, num_heads_v, head_v)
    
    # Extract b and a from mixed_ba
    # mixed_ba shape: [batch, num_heads_qk, num_heads_v // num_heads_qk * 2]
    ba_dim = (num_heads_v // num_heads_qk) * 2
    mixed_ba_reshaped = mixed_ba.view(batch * seq_len, num_heads_qk, ba_dim)
    
    b_end = num_heads_v // num_heads_qk
    a_end = b_end + num_heads_v // num_heads_qk
    
    b_part = mixed_ba_reshaped[:, :, :b_end]
    a_part = mixed_ba_reshaped[:, :, b_end:a_end]
    
    # Reshape to [batch * seq_len, num_heads_v]
    b = b_part.reshape(batch * seq_len, num_heads_v)
    a = a_part.reshape(batch * seq_len, num_heads_v)
    
    return mixed_qkv, z, b, a
