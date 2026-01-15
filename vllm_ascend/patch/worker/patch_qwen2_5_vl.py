from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Annotated, Any, Literal, TypeAlias

import einops
import torch
import torch.distributed as dist
import os
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.config import MultiModalConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.distributed.parallel_state import get_tp_group

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionBlock,
    Qwen2_5_VisionPatchMerger,
)


logger = init_logger(__name__)

def all_to_all_4d(input_tensor: torch.Tensor,
                  is_seq_to_head: bool,
                  group=None,
                  use_sync: bool = False) -> torch.Tensor:
    seq_world_size = dist.get_world_size(group)
    if is_seq_to_head:
        # Transfer shape (bs, seqlen/sp, hc, hs) to (bs, seqlen, hc/sp, hs)
        bs, shard_seqlen, hc, hs = input_tensor.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        input_t = (input_tensor.reshape(bs, shard_seqlen, seq_world_size,
                                        shard_hc,
                                        hs).transpose(0, 2).contiguous())

        output = torch.empty_like(input_t)
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                platform = input_tensor.device.type
                sync_func = getattr(torch, f"{platform}.synchronize")
                sync_func()
        else:
            output = input_t

        output = output.reshape(seqlen, bs, shard_hc,
                                hs).transpose(0, 1).contiguous()
        return output
    else:
        bs, seqlen, shard_hc, hs = input_tensor.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        input_t = (input_tensor.reshape(
            bs, seq_world_size, shard_seqlen, shard_hc,
            hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                seq_world_size, shard_hc, shard_seqlen, bs, hs))

        output = torch.empty_like(input_t)
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                platform = input_tensor.device.type
                sync_func = getattr(torch, f"{platform}.synchronize")
                sync_func()
        else:
            output = input_t

        output = output.reshape(hc, shard_seqlen, bs,
                                hs).transpose(0, 2).contiguous()
        return output.reshape(bs, shard_seqlen, hc, hs)


def all_to_all_3d(input_tensor: torch.Tensor,
                  is_seq_to_head: bool,
                  group=None,
                  use_sync: bool = False) -> torch.tensor:
    seq_world_size = dist.get_world_size(group)
    if is_seq_to_head:
        shard_seqlen, hc, hs = input_tensor.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        input_t = (input_tensor.reshape(shard_seqlen, seq_world_size, shard_hc,
                                        hs).transpose(0, 1).contiguous())

        output = torch.empty_like(input_t)
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                platform = input_tensor.device.type
                sync_func = getattr(torch, f"{platform}.synchronize")
                sync_func()
        else:
            output = input_t
        output = output.reshape(seqlen, shard_hc, hs)
        return output
    else:
        # Transfer shape (seqlen, hc/sp, hs) to (seqlen/sp, hc, hs)
        seqlen, shard_hc, hs = input_tensor.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        input_t = (input_tensor.reshape(seq_world_size, shard_seqlen, shard_hc,
                                        hs).transpose(1, 2).contiguous())

        output = torch.empty_like(input_t)
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                platform = input_tensor.device.type
                sync_func = getattr(torch, f"{platform}.synchronize")
                sync_func()
        else:
            output = input_t

        output = output.reshape(hc, shard_seqlen,
                                hs).transpose(0, 1).contiguous()
        return output


def all_gather_2d(input_tensor: torch.tensor,
                  world_size: int,
                  group=None) -> torch.tensor:
    s, d = input_tensor.shape
    input_gather = torch.zeros(world_size * s,
                               d,
                               dtype=input_tensor.dtype,
                               device=input_tensor.device)
    dist.all_gather_into_tensor(input_gather, input_tensor, group=group)

    return input_gather


def get_rank_world():
    rank = get_tp_group().rank
    tp_size = get_tp_group().world_size
    tp_group = get_tp_group().device_group
    return rank, tp_size, tp_group


class AscendQwen2_5_VisionMLP(nn.Module):
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
        super(Qwen2_5_VisionMLP,self).__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )

        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )
        self.act_fn = act_fn


class AscendQwen2_5_VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super(Qwen2_5_VisionAttention,self).__init__()
        # Per attention head and per partition values.
        self.use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )

        if self.use_data_parallel:
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.tp_world_size = 1
            self.tp_group = None
        else :
            self.tp_rank, self.tp_world_size, self.tp_group = get_rank_world()

        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_world_size
        )
         

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=True,
        )
        
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=True,
        )
 
        
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            multimodal_config=multimodal_config,
        )

        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        true_seq: int,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        if not self.use_data_parallel:
            x = einops.rearrange(
                x,
                "s b (three head head_dim) -> (b three) s head head_dim",
                b=1,
                three=3,
                head=self.num_attention_heads_per_partition*self.tp_world_size,
            )

            x = all_to_all_4d(x, is_seq_to_head=True, group=self.tp_group)
            cur_seq = x.shape[1]
            x = x[:, :true_seq, :, :]
        
            qkv = einops.rearrange(
                x,
                '(b three) s head head_dim -> b s three head head_dim',
                b=1,
                three=3,
                head=self.num_attention_heads_per_partition,
            )
            used_len = true_seq
        else :
            qkv = einops.rearrange(
                x,
                "s b (three head head_dim) -> b s three head head_dim",
                three=3,
                head=self.num_attention_heads_per_partition,
            )
            used_len = seq_len
		
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                used_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)
       
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if not self.use_data_parallel:
            padding = (0, 0, 0, 0, 0, cur_seq - true_seq)
            context_layer = F.pad(context_layer, padding)

            context_layer = context_layer.reshape(
                seq_len * self.tp_world_size,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head
            )
            
            context_layer = all_to_all_3d(context_layer, is_seq_to_head=False, group=self.tp_group)
            context_layer = einops.rearrange(
                context_layer, "(b s) h d -> s b (h d)", b=batch_size
            ).contiguous()
        else :
            context_layer = einops.rearrange(
                context_layer, "b s h d -> s b (h d)", b=batch_size
            ).contiguous()
        output, _ = self.proj(context_layer)
        return output


class AscendQwen2_5_VisionBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        true_seq: int
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            true_seq=true_seq
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class AscendQwen2_5_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super(Qwen2_5_VisionPatchMerger,self).__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
                return_bias=False,
                disable_tp=True,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
                return_bias=False,
                disable_tp=True,
            ),
        )


Qwen2_5_VisionMLP.__init__ = AscendQwen2_5_VisionMLP.__init__
Qwen2_5_VisionAttention.__init__ = AscendQwen2_5_VisionAttention.__init__
Qwen2_5_VisionAttention.forward = AscendQwen2_5_VisionAttention.forward
Qwen2_5_VisionBlock.forward = AscendQwen2_5_VisionBlock.forward
Qwen2_5_VisionPatchMerger.__init__ = AscendQwen2_5_VisionPatchMerger.__init__
