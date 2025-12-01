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

from functools import lru_cache, partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import \
    Qwen2_5_VLVisionConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import \
    Qwen2VLVisionConfig
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import (check_upstream_fa_availability,
                                  maybe_get_vit_flash_attn_backend)
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import (
    apply_rotary_emb_torch, dispatch_rotary_emb_function)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock, Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionPatchMerger, Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLImageInputs,
    Qwen2_5_VLVideoInputs)
from vllm.model_executor.models.qwen2_vl import (Qwen2VisionAttention,
                                                 Qwen2VisionBlock,
                                                 Qwen2VisionPatchEmbed,
                                                 Qwen2VisionPatchMerger,
                                                 Qwen2VisionTransformer)
from vllm.model_executor.models.utils import cast_overflow_tensors
from vllm.model_executor.models.vision import (
    get_vit_attn_backend, run_dp_sharded_mrope_vision_model)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import set_ascend_forward_context

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendQwen2_5_VisionAttention(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        # Split q k v.
        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        origin_shape = q.shape[-1]

        # Convert cumulative tensor to intervals and move it to cpu.
        cu_seqlens = torch.diff(cu_seqlens).to("cpu")

        cos = torch.cat((rotary_pos_emb_cos, rotary_pos_emb_cos), dim=-1)
        sin = torch.cat((rotary_pos_emb_sin, rotary_pos_emb_sin), dim=-1)
        cos = cos.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        sin = sin.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            einops.rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        enable_pad = (envs_ascend.USE_OPTIMIZED_MODEL
                      and self.hidden_size_per_attention_head > MIN_PAD_SIZE
                      and self.hidden_size_per_attention_head < MAX_PAD_SIZE)

        if enable_pad:
            pad_len = MAX_PAD_SIZE - origin_shape
            # q/k/v: [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer,
        )

        if enable_pad:
            context_layer = context_layer[..., :origin_shape]

        context_layer = einops.rearrange(context_layer,
                                         "(b s) h d -> s b (h d)",
                                         b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class AscendQwen2VisionBlock(nn.Module):

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            max_seqlen: int | None = None,  # Only used for Flash Attention
            seqlens: list[int] | None = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class AscendQwen2VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        nn.Module.__init__(self)

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        in_channels = vision_config.in_channels
        hidden_size = vision_config.hidden_size
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_ratio = vision_config.mlp_ratio

        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.hidden_size

        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.blocks = nn.ModuleList([
            Qwen2VisionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_data_parallel=use_data_parallel,
                attn_backend_override=attn_backend_override,
            ) for layer_idx in range(depth)
        ])
        self.merger = Qwen2VisionPatchMerger(
            d_model=hidden_size,
            context_dim=embed_dim,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel,
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        if (self.attn_backend != AttentionBackendEnum.FLASH_ATTN
                and check_upstream_fa_availability(torch.get_default_dtype())):
            self.attn_backend = AttentionBackendEnum.FLASH_ATTN

    def rot_pos_emb(
            self,
            grid_thw: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten())
            wpos_ids = (wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten())
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
            max_grid_size = max(max_grid_size, h, w)
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
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(
            grid_thw_list)

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        # transformers
        x = x.unsqueeze(1)

        # pre-compute seqlens for attn mask to reduce cuMemcpy operations
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )

        # adapter
        x = self.merger(x)

        return x


class AscendQwen2_5_VisionBlock(nn.Module):

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            max_seqlen: torch.Tensor,  # Only used for Flash Attention
            seqlens: torch.Tensor,  # Only used for xFormers
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class AscendQwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        nn.Module.__init__(self)

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size

        # args for get_window_index_thw
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2
        # TODO[@lucaskabela]: Investigate fixing this usage
        # see https://github.com/vllm-project/vllm/issues/27044
        # DO NOT MOVE THIS IMPORT
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("Qwen2_5_VisionPatchEmbed"):
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                hidden_size=self.hidden_size,
            )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        use_upstream_fa = False
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
                attn_backend_override=attn_backend_override,
            ))

        with set_model_tag("Qwen2_5_VisionBlock"):
            self.blocks = nn.ModuleList([
                Qwen2_5_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                    attn_backend=self.attn_backend,
                    use_upstream_fa=use_upstream_fa,
                    attn_backend_override=attn_backend_override,
                ) for layer_idx in range(depth)
            ])

        with set_model_tag("Qwen2_5_VisionPatchMerger"):
            self.merger = Qwen2_5_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=self.hidden_size,
                norm_layer=norm_layer,
                spatial_merge_size=self.spatial_merge_size,
                quant_config=quant_config,
                prefix=f"{prefix}.merger",
                use_data_parallel=use_data_parallel,
            )

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = (hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten())
        wpos_ids = (wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten())
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_size)

        cos_h = cos[pos_ids[:, 0]]  # (num_tokens, rotary_dim // 2)
        cos_w = cos[pos_ids[:, 1]]
        sin_h = sin[pos_ids[:, 0]]
        sin_w = sin[pos_ids[:, 1]]

        cos_combined = torch.cat([cos_h, cos_w], dim=-1)
        sin_combined = torch.cat([sin_h, sin_w], dim=-1)

        cos_combined = cos_combined.reshape(
            cos_combined.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        sin_combined = sin_combined.reshape(
            sin_combined.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        return cos_combined, sin_combined

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)
        cos_thw, sin_thw = self.rotary_pos_emb_thw(t, h, w)

        cos_thw = cos_thw[window_index_thw, :, :]
        cos_thw = cos_thw.flatten(start_dim=0, end_dim=1)
        sin_thw = sin_thw[window_index_thw, :, :]
        sin_thw = sin_thw.flatten(start_dim=0, end_dim=1)

        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t)
        return (
            cos_thw,
            sin_thw,
            window_index_thw,
            cu_seqlens_window_thw,
            cu_seqlens_thw,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        seq_len, _ = x.size()
        rotary_pos_emb_cos: list = []
        rotary_pos_emb_sin: list = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                cos_thw,
                sin_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb_cos = torch.cat(rotary_pos_emb_cos)
        rotary_pos_emb_sin = torch.cat(rotary_pos_emb_sin)
        window_index = torch.cat(window_index)
        # compute reverse indices
        reverse_indices = self.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        # pre-compute seqlens for window/full attn to reduce cuMemcpy operations
        max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(
            cu_seqlens)
        max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(
            cu_window_seqlens)

        cu_seqlens = cu_seqlens.to(  # type: ignore[attr-defined]
            device=self.device,
            non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(  # type: ignore[attr-defined]
            device=self.device,
            non_blocking=True)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(  # type: ignore[attr-defined]
            device=self.device,
            non_blocking=True)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(  # type: ignore[attr-defined]
            device=self.device,
            non_blocking=True)
        window_index = window_index.to(  # type: ignore[attr-defined]
            device=hidden_states.device,
            non_blocking=True)
        reverse_indices = reverse_indices.to(device=hidden_states.device,
                                             non_blocking=True)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                seqlens_now = seqlens_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                seqlens_now = seqlens_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen_now,
                seqlens=seqlens_now,
            )

        # For Qwen2.5-VL-3B, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


class AscendQwen2_5_VLForConditionalGeneration(nn.Module):

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_ascend_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values,
                        grid_thw_list,
                        rope_type="rope_3d")
                else:
                    image_embeds = self.visual(pixel_values,
                                               grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_ascend_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values_videos,
                        grid_thw_list,
                        rope_type="rope_3d",
                    )
                else:
                    video_embeds = self.visual(pixel_values_videos,
                                               grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)


def _apply_rotary_pos_emb_vision(t: torch.Tensor, cos: torch.Tensor,
                                 sin: torch.Tensor) -> torch.Tensor:
    rotary_emb_function = dispatch_rotary_emb_function(
        default=partial(apply_rotary_emb_torch, is_neox_style=True))
    output = rotary_emb_function(t, cos, sin).type_as(t)
    return output


# NOTE: This will be removed after MMEncoderAttention has been extract as a CustomOp in vllm.
Qwen2VisionAttention.forward = AscendQwen2_5_VisionAttention.forward
Qwen2_5_VisionAttention.forward = AscendQwen2_5_VisionAttention.forward

# NOTE: These will be removed after https://github.com/vllm-project/vllm/pull/29388 is merged.
Qwen2_5_VLForConditionalGeneration._process_image_input = AscendQwen2_5_VLForConditionalGeneration._process_image_input
Qwen2_5_VLForConditionalGeneration._process_video_input = AscendQwen2_5_VLForConditionalGeneration._process_video_input

# NOTE: These will be removed after vllm-ascend is aligned with vllm latest main.
Qwen2VisionBlock.forward = AscendQwen2VisionBlock.forward
Qwen2VisionTransformer.__init__ = AscendQwen2VisionTransformer.__init__
Qwen2VisionTransformer.rot_pos_emb = AscendQwen2VisionTransformer.rot_pos_emb
Qwen2VisionTransformer.forward = AscendQwen2VisionTransformer.forward
Qwen2_5_VisionBlock.forward = AscendQwen2_5_VisionBlock.forward
Qwen2_5_VisionTransformer.__init__ = AscendQwen2_5_VisionTransformer.__init__
Qwen2_5_VisionTransformer.rotary_pos_emb_thw = AscendQwen2_5_VisionTransformer.rotary_pos_emb_thw
Qwen2_5_VisionTransformer.get_rope_by_thw = AscendQwen2_5_VisionTransformer.get_rope_by_thw
Qwen2_5_VisionTransformer.forward = AscendQwen2_5_VisionTransformer.forward
apply_rotary_pos_emb_vision = _apply_rotary_pos_emb_vision
