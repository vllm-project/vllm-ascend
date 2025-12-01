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

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import \
    Qwen3VLVisionConfig
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import check_upstream_fa_availability
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_vl import (Qwen3_VisionBlock,
                                                 Qwen3_VisionPatchEmbed,
                                                 Qwen3_VisionPatchMerger,
                                                 Qwen3_VisionTransformer)
from vllm.model_executor.models.vision import get_vit_attn_backend


class AscendQwen3_VisionBlock(nn.Module):

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            max_seqlen: torch.Tensor,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class AscendQwen3_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        nn.Module.__init__(self)

        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.use_data_parallel = use_data_parallel
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes))

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings,
                                      self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel,
        )

        self.deepstack_merger_list = nn.ModuleList([
            Qwen3_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                use_postshuffle_norm=True,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                use_data_parallel=use_data_parallel,
            ) for layer_idx in range(len(self.deepstack_visual_indexes))
        ])

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )
        use_upstream_fa = False
        if (self.attn_backend != AttentionBackendEnum.FLASH_ATTN
                and self.attn_backend != AttentionBackendEnum.ROCM_AITER_FA
                and check_upstream_fa_availability(torch.get_default_dtype())):
            self.attn_backend = AttentionBackendEnum.FLASH_ATTN
            use_upstream_fa = True

        if self.attn_backend not in {
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.XFORMERS,
                AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"Qwen3-VL does not support {self.attn_backend} backend now.")
        self.blocks = nn.ModuleList([
            Qwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_data_parallel=use_data_parallel,
                attn_backend=self.attn_backend,
                use_upstream_fa=use_upstream_fa,
            ) for layer_idx in range(vision_config.depth)
        ])

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size) if t == 1 else
            self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        # (num_tokens, rotary_dim // 2)
        cos_h = cos[pos_ids[:, 0]]  # type: ignore
        cos_w = cos[pos_ids[:, 1]]  # type: ignore
        sin_h = sin[pos_ids[:, 0]]  # type: ignore
        sin_w = sin[pos_ids[:, 1]]  # type: ignore

        cos_combined = torch.cat([cos_h, cos_w], dim=-1)
        sin_combined = torch.cat([sin_h, sin_w], dim=-1)

        return cos_combined, sin_combined

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device,
                             dtype=self.dtype,
                             non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw = grid_thw.to("cpu")
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(
            grid_thw_list)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(hidden_states.device,
                                                   non_blocking=True)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(hidden_states.device,
                                                   non_blocking=True)

        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2],
                               grid_thw[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens)

        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num)
                deepstack_feature = self.deepstack_merger_list[
                    deepstack_merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists,
            dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states


# NOTE: These will be removed after vllm-ascend is aligned with vllm latest main.
Qwen3_VisionBlock.forward = AscendQwen3_VisionBlock.forward
Qwen3_VisionTransformer.__init__ = AscendQwen3_VisionTransformer.__init__
Qwen3_VisionTransformer.rot_pos_emb = AscendQwen3_VisionTransformer.rot_pos_emb
Qwen3_VisionTransformer.forward = AscendQwen3_VisionTransformer.forward
