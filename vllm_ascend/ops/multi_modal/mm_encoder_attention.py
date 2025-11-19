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

import torch.nn.functional as F

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention


MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendMMEncoderAttention(MMEncoderAttention):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            quant_config,
            prefix,
        )

        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.enable_pad = False
        self.finish_pad = False

        # TODO(shen-shanshan): Add verification for env vars (enable unpad).
        if self.hidden_size_per_attention_head > MIN_PAD_SIZE \
            and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
            self.half_origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head // 2
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def pad_qkv_bias(self, bias):
        first_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, :self.half_origin_hidden_size_per_attention_head]
        second_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, self.half_origin_hidden_size_per_attention_head:]
        first_half_padded = F.pad(
            first_half, (0, self.half_pad_hidden_size_per_attention_head))
        second_half_padded = F.pad(
            second_half, (0, self.half_pad_hidden_size_per_attention_head))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(-1)
        return bias_final

    def pad_qkv_weight(self, data):
        qkv_weight_first_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size
        )[:, :, :self.half_origin_hidden_size_per_attention_head, :]
        qkv_weight_second_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size
        )[:, :, self.half_origin_hidden_size_per_attention_head:, :]

        qkv_weight_first_half_padded = F.pad(
            qkv_weight_first_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_second_half_padded = F.pad(
            qkv_weight_second_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_padded = torch.cat(
            [qkv_weight_first_half_padded, qkv_weight_second_half_padded],
            dim=2)
        qkv_weight_final = qkv_weight_padded.reshape(-1, self.hidden_size)

        if is_enable_nz():
            qkv_weight_final_copy = torch.empty_like(qkv_weight_final).copy_(
                qkv_weight_final)
            qkv_weight_final_copy = torch_npu.npu_format_cast(
                qkv_weight_final_copy, ACL_FORMAT_FRACTAL_ND)
            return qkv_weight_final_copy

        return qkv_weight_final

    def pad_proj_weight(self, data):
        out_weight = F.pad(
            data.reshape(self.hidden_size, -1,
                         self.half_origin_hidden_size_per_attention_head),
            (0, self.half_pad_hidden_size_per_attention_head, 0, 0)).reshape(
                self.hidden_size, -1)

        if is_enable_nz():
            out_weight_copy = torch.empty_like(out_weight).copy_(out_weight)
            out_weight_copy = torch_npu.npu_format_cast(
                out_weight_copy, ACL_FORMAT_FRACTAL_ND)
            return out_weight_copy

        return out_weight

    def pad_qkv_weight_scale_offset(self, data):
        reshaped_data = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, 1)
        data1 = reshaped_data[:, :, :self.
                              half_origin_hidden_size_per_attention_head, :]
        data2 = reshaped_data[:, :, self.
                              half_origin_hidden_size_per_attention_head:, :]
        data1_paded = F.pad(
            data1, (0, 0, 0, self.half_pad_hidden_size_per_attention_head, 0,
                    0, 0, 0))
        data2_paded = F.pad(
            data2, (0, 0, 0, self.half_pad_hidden_size_per_attention_head, 0,
                    0, 0, 0))
        res = torch.cat([data1_paded, data2_paded], dim=2)
        res = res.reshape(-1, 1)
        return res

    def pad_qkv_deq_scale_quant_bias(self, data):
        reshaped_data = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head)
        data1 = reshaped_data[:, :, :self.
                              half_origin_hidden_size_per_attention_head]
        data2 = reshaped_data[:, :,
                              self.half_origin_hidden_size_per_attention_head:]

        data1_paded = F.pad(
            data1, (0, self.half_pad_hidden_size_per_attention_head))
        data2_paded = F.pad(
            data2, (0, self.half_pad_hidden_size_per_attention_head))

        res = torch.cat([data1_paded, data2_paded], dim=2)
        res = res.reshape(-1)
        return res

    def pad_cos_sin(self, cos: torch.Tensor, sin: torch.Tensor):
        # cos/sin: [seqlen, rotary_dim / 2] ?
        if self.enable_pad:
            cos = F.pad(
                cos, (0, self.half_pad_hidden_size_per_attention_head))
            sin = F.pad(
                sin, (0, self.half_pad_hidden_size_per_attention_head))
        
        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(torch.stack((cos, cos), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
            sin_new = rearrange(torch.stack((sin, sin), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
        cos_new = cos_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        return cos_new, sin_new
        
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Padding weight
        if self.enable_pad and not self.finish_pad:
            self.qkv.weight.data = self.pad_qkv_weight(self.qkv.weight.data)
            self.qkv.bias.data = self.pad_qkv_bias(self.qkv.bias.data)
            self.proj.weight.data = self.pad_proj_weight(self.proj.weight.data)
            # TODO(shen-shanshan): optimize this to avoid redundant computation.
            cos, sin = self.pad_cos_sin(cos, sin)
            # TODO(shen-shanshan): add padding for quantization.
            self.finish_pad = True

        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)

        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output
