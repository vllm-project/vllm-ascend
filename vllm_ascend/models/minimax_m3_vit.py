# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 vision tower for Ascend bring-up.

This module is adapted from vLLM main's MiniMax-M3 vision implementation, with
the first Ascend version kept intentionally close to the generic vLLM vision
interfaces available in v0.21.0.
"""

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig

from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)


def _get_vision_tp_size_and_disable_tp() -> tuple[int, bool]:
    if is_vit_use_data_parallel():
        return 1, True
    try:
        return parallel_state.get_tensor_model_parallel_world_size(), False
    except AssertionError as exc:
        if "tensor model parallel group is not initialized" not in str(exc):
            raise
        return 1, True


class MiniMaxVLPatchEmbed(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        temporal_patch_size = compression.get("temporal_patch_size", 2)
        patch_size = config.patch_size
        num_channels = config.num_channels

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels

        self.patch_embedding = nn.Conv3d(
            in_channels=num_channels,
            out_channels=config.hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)
        x = pixel_values.reshape(
            pixel_values.shape[0],
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.patch_embedding(x).reshape(x.shape[0], -1)


class MiniMaxVLAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size, disable_tp = _get_vision_tp_size_and_disable_tp()
        self.head_dim = embed_dim // num_heads
        self.num_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=disable_tp,
        )
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=disable_tp,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True, enable_fp32_compute=True
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_qkv, _ = self.qkv_proj(x)
        seq_len, batch_size, _ = x_qkv.shape

        qkv = rearrange(
            x_qkv,
            "s b (three head d) -> b s three head d",
            three=3,
            head=self.num_heads_per_partition,
        )
        qk, v = qkv[:, :, :2], qkv[:, :, 2]

        qk_reshaped = rearrange(qk, "b s two h d -> (two b) s h d", two=2)
        qk_reshaped = qk_reshaped.contiguous()
        rotary_dim = rotary_cos.shape[-1] * 2
        qk_rot, qk_pass = (
            qk_reshaped[..., :rotary_dim],
            qk_reshaped[..., rotary_dim:],
        )
        qk_rot = self.apply_rotary_emb(qk_rot, rotary_cos, rotary_sin)
        qk_rotated = torch.cat((qk_rot, qk_pass), dim=-1)
        qk_rotated = qk_rotated.view(
            2, batch_size, seq_len, self.num_heads_per_partition, self.head_dim
        )
        q, k = qk_rotated.unbind(dim=0)

        context = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        context = rearrange(context, "b s h d -> s b (h d)", b=batch_size)
        output, _ = self.out_proj(context)
        return output


class MiniMaxVLEncoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        _, disable_tp = _get_vision_tp_size_and_disable_tp()

        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn = MiniMaxVLAttention(
            embed_dim=embed_dim,
            num_heads=config.num_attention_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=disable_tp,
        )
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=disable_tp,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.layer_norm1(x),
            cu_seqlens,
            rotary_cos,
            rotary_sin,
            max_seqlen,
            sequence_lengths,
        )
        residual = x
        x, _ = self.fc1(self.layer_norm2(x))
        x = self.act(x)
        x, _ = self.fc2(x)
        return residual + x


class MiniMaxVLEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        num_hidden_layers_override: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        num_layers = (
            config.num_hidden_layers
            if num_hidden_layers_override is None
            else num_hidden_layers_override
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxVLEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                cu_seqlens,
                rotary_cos,
                rotary_sin,
                max_seqlen,
                sequence_lengths,
            )
        return x


class MiniMaxVLVisionTransformer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        self.spatial_merge_size = compression.get("spatial_merge_size", 2)
        self.tp_size, self.disable_tp = _get_vision_tp_size_and_disable_tp()

        embed_dim = config.hidden_size
        head_dim = embed_dim // config.num_attention_heads
        self.hidden_size = embed_dim
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )

        rope_dims = 2 * (head_dim // 2)
        self.t_dim = int(2 * ((rope_dims // 3) // 2))
        self.h_dim = int(2 * ((rope_dims // 3) // 2))
        self.w_dim = int(2 * ((rope_dims // 3) // 2))

        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq_t = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.t_dim, 2, dtype=torch.float32) / self.t_dim)
        )
        inv_freq_h = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        )
        inv_freq_w = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        )
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

        self.embeddings = MiniMaxVLPatchEmbed(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        total_layers = config.num_hidden_layers
        if num_hidden_layers_override is None:
            num_hidden_layers_override = total_layers
        self.encoder = MiniMaxVLEncoder(
            config=config,
            num_hidden_layers_override=num_hidden_layers_override,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )

        if require_post_norm is None:
            require_post_norm = num_hidden_layers_override == total_layers
        self.post_layernorm = (
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            if require_post_norm
            else None
        )
        self.out_hidden_size = embed_dim

    def _get_3d_rope_embed(
        self, grid_t: int, grid_h: int, grid_w: int, spatial_merge_size: int
    ) -> torch.Tensor:
        tokens_per_frame = grid_h * grid_w
        tpos_ids = (
            torch.arange(grid_t, device=self.inv_freq_t.device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .flatten()
        )
        hpos_ids = (
            torch.arange(grid_h, device=self.inv_freq_h.device)
            .unsqueeze(1)
            .expand(-1, grid_w)
            .reshape(
                grid_h // spatial_merge_size,
                spatial_merge_size,
                grid_w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .unsqueeze(0)
            .expand(grid_t, -1, -1, -1, -1)
            .flatten()
        )
        wpos_ids = (
            torch.arange(grid_w, device=self.inv_freq_w.device)
            .unsqueeze(0)
            .expand(grid_h, -1)
            .reshape(
                grid_h // spatial_merge_size,
                spatial_merge_size,
                grid_w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .unsqueeze(0)
            .expand(grid_t, -1, -1, -1, -1)
            .flatten()
        )

        seq_t = torch.arange(
            max(grid_t, 1), device=self.inv_freq_t.device, dtype=self.inv_freq_t.dtype
        )
        seq_hw = torch.arange(
            max(grid_h, grid_w),
            device=self.inv_freq_h.device,
            dtype=self.inv_freq_h.dtype,
        )
        freqs_t = torch.outer(seq_t, self.inv_freq_t)
        freqs_h = torch.outer(seq_hw, self.inv_freq_h)
        freqs_w = torch.outer(seq_hw, self.inv_freq_w)
        return torch.cat(
            [freqs_t[tpos_ids], freqs_h[hpos_ids], freqs_w[wpos_ids]], dim=-1
        )

    def _get_rope_embed_3d(
        self, grid_thw: list[list[int]], spatial_merge_size: int
    ) -> torch.Tensor:
        embeds = [
            self._get_3d_rope_embed(t, h, w, spatial_merge_size) for t, h, w in grid_thw
        ]
        return torch.cat(embeds, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden = self.embeddings(pixel_values)
        hidden = self.pre_layrnorm(hidden)

        lens = [int(t) * int(h) * int(w) for t, h, w in grid_thw]
        cu_seqlens_np = np.zeros(len(lens) + 1, dtype=np.int32)
        np.cumsum(np.array(lens, dtype=np.int32), out=cu_seqlens_np[1:])

        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens_np, hidden.device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens_np),
            dtype=torch.int32,
            device=hidden.device,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            self.hidden_size,
            self.tp_size,
            hidden.device,
        )

        freqs = self._get_rope_embed_3d(grid_thw, self.spatial_merge_size)
        freqs = freqs.to(device=hidden.device)
        rotary_cos, rotary_sin = freqs.cos(), freqs.sin()

        hidden = hidden.unsqueeze(1)
        hidden = self.encoder(
            hidden,
            cu_seqlens,
            rotary_cos,
            rotary_sin,
            max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        hidden = hidden.squeeze(1)

        if self.post_layernorm is not None:
            hidden = self.post_layernorm(hidden)
        return hidden


class MiniMaxVLMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_size: int | None,
        multimodal_projector_bias: bool,
        projector_hidden_act: str = "gelu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        mid = projector_hidden_size if projector_hidden_size else text_hidden_size
        _, disable_tp = _get_vision_tp_size_and_disable_tp()
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            mid,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            disable_tp=disable_tp,
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            disable_tp=disable_tp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_1(x)
        x = self.act(x)
        x, _ = self.linear_2(x)
        return x


class MiniMaxVLPatchMerger(nn.Module):
    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_size: int | None,
        patch_merge_bias: bool,
        projector_hidden_act: str = "gelu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        mid = projector_hidden_size if projector_hidden_size else text_hidden_size
        merge_in = text_hidden_size * spatial_merge_size**2
        _, disable_tp = _get_vision_tp_size_and_disable_tp()
        self.linear_1 = ColumnParallelLinear(
            merge_in,
            mid,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            disable_tp=disable_tp,
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid,
            text_hidden_size,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            disable_tp=disable_tp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merge_area = self.spatial_merge_size**2
        x = x.reshape(x.shape[0] // merge_area, -1)
        x, _ = self.linear_1(x)
        x = self.act(x)
        x, _ = self.linear_2(x)
        return x


class MiniMaxVLVisionModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        text_hidden_size: int,
        projector_hidden_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        spatial_merge_size = compression.get("spatial_merge_size", 2)
        self.spatial_merge_size = spatial_merge_size
        _, self.disable_tp = _get_vision_tp_size_and_disable_tp()

        self.vision_model = MiniMaxVLVisionTransformer(
            config=config,
            require_post_norm=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )
        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            vision_hidden_size=config.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            multimodal_projector_bias=getattr(
                config, "multimodal_projector_bias", True
            ),
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size=spatial_merge_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            patch_merge_bias=getattr(config, "patch_merge_bias", True),
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "patch_merge_mlp"),
        )

        self.dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        self.out_hidden_size = text_hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden = self.vision_model(pixel_values=pixel_values, grid_thw=grid_thw)
        hidden = self.multi_modal_projector(hidden)
        hidden = self.patch_merge_mlp(hidden)
        return hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj.", "q_proj.", "q"),
            ("qkv_proj.", "k_proj.", "k"),
            ("qkv_proj.", "v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = name.replace(".mlp.fc1.", ".fc1.")
            name = name.replace(".mlp.fc2.", ".fc2.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
