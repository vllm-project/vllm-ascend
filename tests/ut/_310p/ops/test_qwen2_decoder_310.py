#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from vllm_ascend._310p.ops.qwen2_decoder import _gqa_attention_bmm_310p


def test_gqa_attention_bmm_310p_matches_sdpa():
    torch.manual_seed(0)
    bsz, num_heads, num_kv_heads, seq_len, head_dim = 2, 14, 2, 32, 16
    q = torch.randn(bsz, num_heads, seq_len, head_dim)
    k = torch.randn(bsz, num_kv_heads, seq_len, head_dim)
    v = torch.randn(bsz, num_kv_heads, seq_len, head_dim)
    scaling = head_dim**-0.5

    half = seq_len // 2
    allow = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    allow[:half, :half] = True
    allow[half:, :half] = True
    allow[half:, half:] = torch.tril(torch.ones((seq_len - half, seq_len - half), dtype=torch.bool))
    mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min).view(1, 1, seq_len, seq_len).expand(bsz, 1, -1, -1)

    k_rep = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_rep = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    expected = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask, scale=scaling).transpose(1, 2)
    actual = _gqa_attention_bmm_310p(q, k, v, mask, scaling)

    assert actual.shape == (bsz, seq_len, num_heads, head_dim)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
