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
from vllm.model_executor.models.deepencoder import add_decomposed_rel_pos

from vllm_ascend._310p.ops.rel_pos_attention import (
    AscendRelPosAttention310,
    _add_decomposed_rel_pos_310p,
    _rel_pos_attention_310p,
)


def test_rel_pos_attention_310_accepts_upstream_kwargs():
    layer = AscendRelPosAttention310(
        dim=32,
        num_heads=4,
        qkv_bias=True,
        use_rel_pos=True,
        use_triton_attention=True,
        rel_pos_zero_init=True,
        input_size=(4, 4),
    )

    assert layer._rel_pos_h_index.shape == (4, 4)
    assert layer._rel_pos_w_index.shape == (4, 4)

    x = torch.randn(2, 4, 4, 32)
    out = layer(x)
    assert out.shape == x.shape


def test_add_decomposed_rel_pos_310p_matches_upstream():
    torch.manual_seed(0)
    q = torch.randn(4, 16, 8)
    rel_pos_h = torch.randn(9, 8)
    rel_pos_w = torch.randn(9, 8)

    expected_h, expected_w = add_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, (4, 4), (4, 4))
    actual_h, actual_w = _add_decomposed_rel_pos_310p(q, rel_pos_h, rel_pos_w, (4, 4), (4, 4), None, None)

    torch.testing.assert_close(actual_h, expected_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_w, expected_w, atol=1e-5, rtol=1e-5)


def test_rel_pos_attention_310p_matches_sdpa():
    torch.manual_seed(0)
    B, num_heads, H, W, head_dim = 2, 4, 3, 5, 8
    q = torch.randn(B, num_heads, H * W, head_dim)
    k = torch.randn(B, num_heads, H * W, head_dim)
    v = torch.randn(B, num_heads, H * W, head_dim)
    rel_h = torch.randn(B, num_heads, H * W, H, 1)
    rel_w = torch.randn(B, num_heads, H * W, 1, W)
    scale = head_dim**-0.5

    attn_bias = (rel_h + rel_w).view(B, num_heads, H * W, H * W)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=scale)
    actual = _rel_pos_attention_310p(q, k, v, rel_h, rel_w, scale)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
