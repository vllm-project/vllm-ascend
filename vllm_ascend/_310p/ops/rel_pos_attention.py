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

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend.ops.rel_pos_attention import AscendRelPosAttention


def _linear_interpolate_1d_310p(rel_pos: torch.Tensor, size: int) -> torch.Tensor:
    src_size = rel_pos.shape[0]
    if src_size == size:
        return rel_pos

    dtype = rel_pos.dtype
    positions = (torch.arange(size, device=rel_pos.device, dtype=torch.float32) + 0.5) * (src_size / size) - 0.5
    positions = torch.clamp(positions, min=0.0, max=src_size - 1)
    left = torch.floor(positions).to(torch.long)
    right = torch.clamp(left + 1, max=src_size - 1)
    weight = (positions - left.to(torch.float32)).unsqueeze(-1)

    rel_pos_float = rel_pos.to(torch.float32)
    resized = rel_pos_float.index_select(0, left) * (1 - weight) + rel_pos_float.index_select(0, right) * weight
    return resized.to(dtype)


def _build_relative_position_index(q_size: int, k_size: int) -> torch.Tensor:
    q_scale = max(k_size / q_size, 1.0)
    k_scale = max(q_size / k_size, 1.0)
    offset = (k_size - 1) * k_scale
    return torch.tensor(
        [[int(q_idx * q_scale - k_idx * k_scale + offset) for k_idx in range(k_size)] for q_idx in range(q_size)],
        dtype=torch.long,
    )


def _get_rel_pos_310p(
    q_size: int,
    k_size: int,
    rel_pos: torch.Tensor,
    relative_position_index: torch.Tensor | None = None,
) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    rel_pos_resized = _linear_interpolate_1d_310p(rel_pos, max_rel_dist)

    if relative_position_index is None or relative_position_index.shape != (q_size, k_size):
        relative_position_index = _build_relative_position_index(q_size, k_size)
    relative_position_index = relative_position_index.to(device=rel_pos.device)
    return rel_pos_resized.index_select(0, relative_position_index.reshape(-1)).reshape(q_size, k_size, -1)


def _add_decomposed_rel_pos_310p(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
    rel_pos_h_index: torch.Tensor | None,
    rel_pos_w_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos_310p(q_h, k_h, rel_pos_h, rel_pos_h_index)
    Rw = _get_rel_pos_310p(q_w, k_w, rel_pos_w, rel_pos_w_index)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1)
    rel_w = rel_w.unsqueeze(-2)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


def _rel_pos_attention_310p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch_size, num_heads, query_tokens, head_dim = q.shape
    k_h = rel_h.shape[-2]
    k_w = rel_w.shape[-1]
    bh = batch_size * num_heads
    q_scaled = q.reshape(bh, query_tokens, head_dim) * scale
    k = k.reshape(bh, k.size(-2), head_dim).transpose(-2, -1)
    v = v.reshape(bh, v.size(-2), head_dim)
    attn = torch.bmm(q_scaled, k)
    attn_4d = attn.view(bh, query_tokens, k_h, k_w)
    attn_4d.add_(rel_h.reshape(bh, query_tokens, k_h, 1))
    attn_4d.add_(rel_w.reshape(bh, query_tokens, 1, k_w))
    attn = torch.softmax(attn, dim=-1)
    x = torch.bmm(attn, v)
    return x.view(batch_size, num_heads, query_tokens, head_dim)


class AscendRelPosAttention310(AscendRelPosAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_rel_pos:
            input_h = (self.rel_pos_h.shape[0] + 1) // 2
            input_w = (self.rel_pos_w.shape[0] + 1) // 2
            self.register_buffer(
                "_rel_pos_h_index",
                _build_relative_position_index(input_h, input_h),
                persistent=False,
            )
            self.register_buffer(
                "_rel_pos_w_index",
                _build_relative_position_index(input_w, input_w),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = _add_decomposed_rel_pos_310p(
                q,
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
                getattr(self, "_rel_pos_h_index", None),
                getattr(self, "_rel_pos_w_index", None),
            )

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            assert rel_h is not None and rel_w is not None
            rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            # 310P rejects npu_prompt_flash_attention with non-null pse_shift,
            # so keep relative-position attention in NPU bmm/softmax/bmm form.
            x = _rel_pos_attention_310p(q, k, v, rel_h, rel_w, self.scale)
        else:
            x = torch_npu.npu_prompt_flash_attention(
                q,
                k,
                v,
                input_layout="BNSD",
                scale_value=self.scale,
                num_heads=self.num_heads,
            )

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
