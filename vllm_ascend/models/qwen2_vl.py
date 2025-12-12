#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm/model_executor/models/qwen2_vl.py
# This file is a part of the vllm-ascend project.

from collections.abc import Iterable
from functools import partial
from typing import Callable, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from transformers.models.qwen2_vl.configuration_qwen2_vl import \
    Qwen2VLVisionConfig
from vllm.config import VllmConfig
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionAttention, Qwen2VisionBlock, Qwen2VisionPatchEmbed,
    Qwen2VisionTransformer, Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration, Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, is_enable_nz

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendQwen2VisionAttention(Qwen2VisionAttention):

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
        self.cu_seqlens = None
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
        if self.hidden_size_per_attention_head > MIN_PAD_SIZE and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:

        self.cu_seqlens = cu_seqlens

        # [s, b, c] --> [s, b, 3 * head * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = [
            rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
        ]
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
            seq_len=self.cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)
        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class AscendQwen2VisionBlock(Qwen2VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, act_layer, norm_layer,
                         quant_config, prefix)
        self.attn = AscendQwen2VisionAttention(embed_dim=dim,
                                               num_heads=num_heads,
                                               projection_size=dim,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.attn")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class AscendQwen2VisionPatchEmbed(Qwen2VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.embed_dim, -1).transpose(0, 1))
        return x


class AscendQwen2VisionTransformer(Qwen2VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)

        self.interleaved = interleaved
        self.enable_pad = False
        self.depth = vision_config.depth
        self.hidden_size = vision_config.embed_dim
        self.num_heads = vision_config.num_heads
        self.patch_embed = AscendQwen2VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )

        self.blocks = nn.ModuleList([
            AscendQwen2VisionBlock(dim=self.embed_dim,
                                   num_heads=self.num_heads,
                                   mlp_ratio=vision_config.mlp_ratio,
                                   norm_layer=partial(nn.LayerNorm,
                                                      eps=norm_eps),
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])

        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads)

        if self.hidden_size_per_attention_head > MIN_PAD_SIZE and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
            self.half_origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head // 2
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()
        if self.enable_pad:
            cos = torch.nn.functional.pad(
                cos, (0, self.half_pad_hidden_size_per_attention_head))
            sin = torch.nn.functional.pad(
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

    def pad_qkv_bias(self, bias):
        first_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, :self.half_origin_hidden_size_per_attention_head]
        second_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, self.half_origin_hidden_size_per_attention_head:]
        first_half_padded = torch.nn.functional.pad(
            first_half, (0, self.half_pad_hidden_size_per_attention_head))
        second_half_padded = torch.nn.functional.pad(
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

        qkv_weight_first_half_padded = torch.nn.functional.pad(
            qkv_weight_first_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_second_half_padded = torch.nn.functional.pad(
            qkv_weight_second_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_padded = torch.cat(
            [qkv_weight_first_half_padded, qkv_weight_second_half_padded],
            dim=2)
        qkv_weight_final = qkv_weight_padded.reshape(-1, self.hidden_size)

        if is_enable_nz(qkv_weight_final.dtype):
            qkv_weight_final_copy = torch.empty_like(qkv_weight_final).copy_(
                qkv_weight_final)
            qkv_weight_final_copy = torch_npu.npu_format_cast(
                qkv_weight_final_copy, ACL_FORMAT_FRACTAL_ND)
            return qkv_weight_final_copy

        return qkv_weight_final

    def pad_proj_weight(self, data):
        out_weight = torch.nn.functional.pad(
            data.reshape(self.hidden_size, -1,
                         self.half_origin_hidden_size_per_attention_head),
            (0, self.half_pad_hidden_size_per_attention_head, 0, 0)).reshape(
                self.hidden_size, -1)

        if is_enable_nz(out_weight.dtype):
            out_weight_copy = torch.empty_like(out_weight).copy_(out_weight)
            out_weight_copy = torch_npu.npu_format_cast(
                out_weight_copy, ACL_FORMAT_FRACTAL_ND)
            return out_weight_copy

        return out_weight

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                if ("attn.proj.weight" in name) and self.enable_pad:
                    param.data = self.pad_proj_weight(param.data)
                if ("attn.qkv.weight" in name) and self.enable_pad:
                    param.data = self.pad_qkv_weight(param.data)
                if ("attn.qkv.bias" in name) and self.enable_pad:
                    param.data = self.pad_qkv_bias(param.data)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        # compute cu_seqlens and avoid cumsum to fit operator unpadFA
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:,
                                                      0]).cpu().to(torch.int32)

        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cos, sin = self.cal_cos_sin(rotary_pos_emb)

        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, cos=cos, sin=sin)

        # adapter
        x = self.merger(x)
        return x


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class AscendQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.visual = AscendQwen2VisionTransformer(
            self.config.vision_config,
            norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
