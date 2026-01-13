from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from itertools import islice
from typing import Any

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig,
)

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import MultiModalConfig
from .patch_qwen2_5_vl import all_gather_2d
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope


from .patch_qwen2_5_vl import (
    get_rank_world,
    Qwen2_5_VisionAttention
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
)

from vllm.model_executor.models.qwen3_vl import(
    Qwen3_VisionMLP,
    Qwen3_VisionBlock,
    Qwen3_VisionPatchMerger,
    Qwen3_VisionTransformer,
    Qwen3_VisionPatchEmbed
)


class AscendQwen3_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super(Qwen3_VisionMLP,self).__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            #disable_tp=use_data_parallel,
            disable_tp=True
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            #disable_tp=use_data_parallel,
            disable_tp=True
        )
        self.act_fn = act_fn

class AscendQwen3_VisionBlock(nn.Module):
    attn: Qwen2_5_VisionAttention

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        true_seq: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            true_seq=true_seq,
        )

        x = x + self.mlp(self.norm2(x))
        return x

class AscendQwen3_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super(Qwen3_VisionPatchMerger,self).__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            #disable_tp=use_data_parallel,
            disable_tp=True,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            #disable_tp=use_data_parallel,
            disable_tp=True,
        )

class AscendQwen3_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super(Qwen3_VisionTransformer,self).__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.merger",
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend if multimodal_config else None
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        if self.attn_backend not in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"Qwen3-VL does not support {self.attn_backend} backend now."
            )
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens)

        seq_len, _ = hidden_states.size()
        tp_rank, tp_world_size, tp_group = get_rank_world()
        if not self.use_data_parallel:
            merge_size = self.spatial_merge_size ** 2
            padding_size = math.ceil(math.ceil(seq_len / tp_world_size) / merge_size
                                 ) * merge_size * tp_world_size - seq_len
            if padding_size > 0:
                padding = torch.zeros(padding_size,
                                  *hidden_states.size()[1:],
                                  dtype=hidden_states.dtype,
                                  device=hidden_states.device)
                hidden_states = torch.cat([hidden_states, padding], dim=0)
            hidden_states = hidden_states.chunk(tp_world_size, dim=0)[tp_rank % tp_world_size]

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
                true_seq=seq_len,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        

        if self.use_data_parallel == False:
            hidden_states = all_gather_2d(hidden_states, world_size=tp_world_size, group=tp_group)
            if padding_size:
                hidden_states = hidden_states[:-padding_size // merge_size]

        return hidden_states


Qwen3_VisionMLP.__init__ = AscendQwen3_VisionMLP.__init__
Qwen3_VisionBlock.forward = AscendQwen3_VisionBlock.forward
Qwen3_VisionPatchMerger.__init__ = AscendQwen3_VisionPatchMerger.__init__
Qwen3_VisionTransformer.__init__ = AscendQwen3_VisionTransformer.__init__
Qwen3_VisionTransformer.forward = AscendQwen3_VisionTransformer.forward