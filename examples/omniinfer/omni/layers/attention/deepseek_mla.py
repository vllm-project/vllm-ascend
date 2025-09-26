# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import time
from typing import Any, Optional, Tuple, Dict
import torch
from torch import nn
import torch_npu
import torchair as tng
import torch.distributed as dist
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm.attention import Attention
from vllm.utils import supports_dynamo
from vllm.config import CacheConfig, QuantizationConfig, CompilationLevel, get_current_vllm_config
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear
)
from vllm.distributed import get_world_group
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group
)
from vllm.platforms import current_platform
from contextlib import nullcontext

from omni.models.common.config.model_config import model_extra_config
from omni.layers.rotary_embedding import get_rope
from omni.layers.linear import (
    MergedReplicatedLinear,
    RowParallelLinearWithReduceScatter,
    DP2TPRowParallelLinear,
    Tp2DpAndTpRowParallelLinear,
    RowParallelLinearCross
)
from omni.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.communication_op import (
    mla_tensor_model_parallel_all_gather, reduce_scatter_cross, all_gather_world)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_o_proj_tp_group,
    get_o_proj_dp_group,
    GroupCoordinator,
    get_npu_device_count,
    get_local_group_from_list
)
from omni.models.common.config.model_config import model_extra_config
KVCACHE_NZ_DIM = 16


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekMLA(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            hidden_size: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            rope_is_neox_style: Optional[bool] = False,
            max_position_embeddings: int = 8192,
            cache_config: Optional[CacheConfig] = None, # type: ignore
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        if num_heads % self.tp_size != 0:
            raise RuntimeError("num_heads % tp_size != 0")
        self.num_local_heads = num_heads // self.tp_size
        self.scale = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_scale = None
        # FA is fully quantized, KVCache is not quantized, and the function is not enabled.
        self.quant_symbol = quant_config is not None

        self.merge_qkv = model_extra_config.operator_opt_config.merge_qkv
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                self.qkv_a_proj = MergedReplicatedLinear(self.hidden_size,
                                                         [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                                                         bias=False,
                                                         quant_config=quant_config,
                                                         prefix=f"{prefix}.qkv_a_proj")
            else:
                self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                                 self.q_lora_rank,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_a_proj")
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)

            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                self.o_proj = Tp2DpAndTpRowParallelLinear(self.num_heads * self.v_head_dim,
                                                          hidden_size,
                                                          tp_size=get_o_proj_tp_group().world_size,
                                                          tp_rank= get_o_proj_tp_group().rank_in_group,
                                                          bias=False,
                                                          quant_config=quant_config,
                                                          prefix=f"{prefix}.o_proj")
            else:
                self.o_proj = ReplicatedLinear(self.num_heads * self.v_head_dim,
                                               hidden_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.o_proj")
        elif model_extra_config.parall_config.o_proj_tp_size > 1:
            self.o_proj = DP2TPRowParallelLinear(self.num_heads * self.v_head_dim,
                                                 hidden_size,
                                                 tp_size=get_o_proj_tp_group().world_size,
                                                 tp_rank=get_o_proj_tp_group().rank_in_group,
                                                 bias=False,
                                                 input_is_parallel=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.o_proj")
        elif model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
            self.o_proj = RowParallelLinearCross(self.num_heads * self.v_head_dim,
                                                self.hidden_size,
                                                bias=False,
                                                tp_size=get_tensor_model_parallel_world_size() // get_npu_device_count(),
                                                tp_rank=get_tensor_model_parallel_rank() // get_npu_device_count(),
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelLinearWithReduceScatter(self.num_heads * self.v_head_dim,
                                                             self.hidden_size,
                                                             bias=False,
                                                             quant_config=quant_config,
                                                             prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'

        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=rope_is_neox_style)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scale = self.scale * mscale * mscale

        self.is_mla_prolog_init = False
        # we found npu_flash_attention can only works on 128 divisible head_dim, we pad it to target size here
        # and slice the final result to guarantee its functionality.
        self.padding_head_dim = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim - 1) // 128 +
            1) * 128

        cur_vllm_config = get_current_vllm_config()
        self.enable_graph_mode = (cur_vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())

        self.attn_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=current_platform.device_type)
        )
        self.qk_rope_head_dim_nz = self.qk_rope_head_dim // 16

        self.fa_quant = model_extra_config.operator_opt_config.fa_quant
        self.kv_scale_reci_tile = None
        self.kv_scale = None
        kv_lora_rank_cache_size = self.kv_lora_rank
        if self.fa_quant:
            kv_lora_rank_cache_size = kv_lora_rank_cache_size // 2
            self.kv_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.vllm_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=kv_lora_rank_cache_size + self.qk_rope_head_dim,
            scale=self.scale,
            use_mla=True,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.is_init = True
        self.W_UK = None
        self.W_UV = None
        # decode use mla absorb
        if model_extra_config.parall_config.dp_size > 1:
            kv_b_proj_weight = self.kv_b_proj.weight.T

            expected_shape = (
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
            )
            if kv_b_proj_weight.shape != expected_shape:
                raise RuntimeError(f"{kv_b_proj_weight.shape} != {expected_shape}")

            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            self.W_UK = self.W_UK.permute(1, 2, 0)
            self.W_UV = self.W_UV.transpose(0, 1)
            self.is_init = False
            self.norm_res = {}
            self.actual_seq_lengths = {}
            for batch_size in model_extra_config.operator_opt_config.decode_gear_list:
                self.norm_res[batch_size] = torch.zeros([batch_size * self.tp_size, self.q_lora_rank], dtype=torch.bfloat16, device=current_platform.device_type)
                self.actual_seq_lengths[batch_size] = torch.tensor(list(range(1, batch_size * self.tp_size + 1)), dtype=torch.int64, device=current_platform.device_type)
                torch._dynamo.mark_static(self.norm_res[batch_size])
                torch._dynamo.mark_static(self.actual_seq_lengths[batch_size])
        if self.quant_symbol and model_extra_config.operator_opt_config.use_mlaprolog:
            self.q_a_proj.weight_scale.data = self.q_a_proj.weight_scale.data.to(torch.float)
            self.q_b_proj.weight_scale.data = self.q_b_proj.weight_scale.data.to(torch.float)
            if self.kv_a_proj_with_mqa is not None:
                self.kv_a_proj_with_mqa.weight_scale.data = self.kv_a_proj_with_mqa.weight_scale.data.to(torch.float)
        if model_extra_config.operator_opt_config.c8_calib_path is not None:
            os.makedirs(model_extra_config.operator_opt_config.c8_calib_path, exist_ok=True)


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        comm_group: Optional[GroupCoordinator] = None,
    ) -> torch.Tensor:
        if not self.is_init:
            self.W_UK = torch.nn.Parameter(self.W_UK.contiguous(), requires_grad=False)
            self.W_UV = torch.nn.Parameter(self.W_UV.contiguous(), requires_grad=False)
            self.is_init = True
        if self.kv_scale is not None and self.kv_scale_reci_tile is None:
            self.kv_scale_reci_tile = torch.nn.Parameter(
                torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(1, -1), requires_grad=False)
        if attn_metadata is None or attn_metadata.prefill is not None:
            if os.getenv("ASCEND_PLATFORM", "A3")=="A2" and model_extra_config.operator_opt_config.pd_seperate_prefill:
                output = self._forward_prefill_a2(positions, hidden_states, kv_cache, attn_metadata)
            else:
                output = self._forward_prefill(positions, hidden_states, kv_cache, attn_metadata, comm_group=comm_group)
        else:
            output = self._forward_decode(
                positions, hidden_states, kv_cache, attn_metadata,
                use_rmsnorm_rope_cache=model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache
            )
        if model_extra_config.operator_opt_config.use_mlaprolog and not self.is_mla_prolog_init:
            self.is_mla_prolog_init = True
            self.q_a_proj.weight = self._process_mla_prolog_weight(self.q_a_proj.weight)
            self.q_b_proj.weight = self._process_mla_prolog_weight(self.q_b_proj.weight)
            self.kv_a_proj_with_mqa.weight = self._process_mla_prolog_weight(self.kv_a_proj_with_mqa.weight)
        return output

    def _process_mla_prolog_weight(self, weight):
        if weight.dtype == torch.int8:
            return weight
        weight.data = torch_npu.npu_format_cast(weight.data, 2)
        weight = torch.nn.Parameter(weight.transpose(0, 1).contiguous(), requires_grad = False)
        weight.data = torch_npu.npu_format_cast(weight.data, 29)
        return weight

    def _forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        comm_group: Optional[GroupCoordinator] = None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = tensor_model_parallel_all_gather(qkv, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)

                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_proj(hidden_states)[0]
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                # q = tensor_model_parallel_all_gather(q, dim=0)
                latent_cache = mla_tensor_model_parallel_all_gather(latent_cache, dim=0, comm_group=comm_group)

                q = self.q_a_layernorm(q)
                if self.quant_symbol:
                    q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                    # Quantizing before all_gather can reduce communication overhead.
                    q_quant = mla_tensor_model_parallel_all_gather(q_quant, dim=0, comm_group=comm_group)
                    q_scale = mla_tensor_model_parallel_all_gather(q_scale, dim=0, comm_group=comm_group)
                    q = {'x_int8':q_quant, 'pertoken_scale':q_scale}
                else:
                    q = mla_tensor_model_parallel_all_gather(q, dim=0, comm_group=comm_group)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = tensor_model_parallel_all_gather(q, dim=0)
            latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],  dim=-1)
        # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd, kv cache:bsnd
        q_pe = q_pe.unsqueeze(2)
        if attn_metadata is None or model_extra_config.operator_opt_config.enable_prefill_micro_batch:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos, sin = attn_metadata.prefill.cos, attn_metadata.prefill.sin
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) #BSH
        q[..., self.qk_nope_head_dim:] = q_pe
        if isinstance(kv_cache, Dict):
            kv_cache = kv_cache.get("kv_cache")
        if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0:
            # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd,kv cache:bsnd
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576), # bnsd
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                k_rope_scale=None,
                c_kv_scale=self.kv_scale_reci_tile,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True) # adapter NZ

            if model_extra_config.operator_opt_config.c8_calib_path is not None and get_world_group().rank_in_group == 0:
                layer_idx = int(self.prefix.split(sep='.')[-2])
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}_{layer_idx}.pth"
                save_path = os.path.join(model_extra_config.operator_opt_config.c8_calib_path, filename)
                torch.save(kv_a.detach().to("cpu").contiguous(), save_path)
        else:
            latent_cache = latent_cache.view(-1, latent_cache.size(-1))
            # adapt end
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)

        is_attn_output_reshape = model_extra_config.operator_opt_config.prefill_enable_mla_alltoall and attn_metadata is None
        o_proj_tp_size = get_o_proj_dp_group().world_size \
            if model_extra_config.parall_config.o_proj_tp_size > 1 else get_tensor_model_parallel_world_size()
        attn_output = torch.empty(
            q.shape[0] // o_proj_tp_size if is_attn_output_reshape else q.shape[0],
            self.num_local_heads * o_proj_tp_size if is_attn_output_reshape else self.num_local_heads,
            self.v_head_dim,
            device=q_nope.device,
            dtype=q_nope.dtype)

        if attn_metadata is not None:
            prefill_metadata = attn_metadata.prefill
            computed_tokens = 0
            assert not (self.fa_quant and len(prefill_metadata.seq_qlen_group) > 1)
            for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                prefill_metadata.seq_qlen_group,
                prefill_metadata.seq_kvlen_group)
            ):
                if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and\
                        kv_cache[0].numel() > 0 and not self.fa_quant:
                    # adapt nz
                    block_num, block_size, head_size, _ = kv_cache[0].shape
                    kv_cache_a = (kv_cache[0]
                                .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_pe = (kv_cache[1]
                                .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_a = kv_cache_a.transpose(1, 3)
                    kv_cache_pe = kv_cache_pe.transpose(1, 3)
                    # adapt end
                    kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]

                if model_extra_config.parall_config.dp_size > 1:
                    self.kv_b_proj.weight = torch.nn.Parameter(torch.cat((self.W_UK.permute(2,0,1), self.W_UV.transpose(0,1)), dim=-1) \
                                                                    .view(self.kv_lora_rank,-1).T, requires_grad=False)
                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                    self.kv_b_proj.weight = None
                else:
                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]

                kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                if prefill_metadata.max_query_len > 1:
                    attn_mask = self.attn_mask
                    sparse_mode = 3
                else:
                    attn_mask = None
                    sparse_mode = 0  # must be 0 if attn_mask is None
                prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)
                attn_output[computed_tokens:computed_tokens+actual_seq_qlen[-1]] = \
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        k_nope,
                        v,
                        query_rope=q_pe[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        key_rope=prefill_k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=self.num_local_heads,
                        input_layout="TND",
                        atten_mask=attn_mask,
                        sparse_mode=sparse_mode,
                        actual_seq_lengths=actual_seq_qlen,
                        actual_seq_lengths_kv=actual_seq_kvlen,
                        scale=self.scale,
                        next_tokens=0)[0]
                computed_tokens += actual_seq_qlen[-1]
        else:
            attn_output.fill_(0)

        # if only set prefill_enable_mla_alltoall means prefill o_proj tp to dp
        # if also set o_proj_tp_size means prefill o_proj tp to dp + tp
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            if attn_metadata is not None:
                if model_extra_config.parall_config.o_proj_tp_size > 1:
                    attn_output = attn_output.view(get_o_proj_dp_group().world_size, -1, self.num_local_heads, self.v_head_dim)
                attn_output = attn_output.reshape(-1)
                all_to_all_attn_output = torch.empty(
                    [q.shape[0] * self.num_local_heads * self.v_head_dim],
                    dtype=attn_output.dtype,
                    device=current_platform.device_type
                )
                device_group = get_o_proj_dp_group().device_group \
                    if model_extra_config.parall_config.o_proj_tp_size > 1 else get_tp_group().device_group
                dist.all_to_all_single(all_to_all_attn_output, attn_output, group=device_group)
                if model_extra_config.parall_config.o_proj_tp_size > 1:
                    attn_output = all_to_all_attn_output.view(
                        get_tensor_model_parallel_world_size() // get_o_proj_tp_group().world_size,
                        q.shape[0] // get_tensor_model_parallel_world_size() * get_o_proj_tp_group().world_size,
                        self.num_local_heads * self.v_head_dim
                    ).transpose(0, 1).contiguous()
                else:
                    attn_output = all_to_all_attn_output.view(
                        get_tensor_model_parallel_world_size(),
                        q.shape[0] // get_tensor_model_parallel_world_size(),
                        self.num_local_heads * self.v_head_dim
                    ).transpose(0, 1).contiguous()
            output, _ = self.o_proj.forward(
                attn_output.reshape(-1, o_proj_tp_size * self.num_local_heads * self.v_head_dim))
        else:
            attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                output, _ = self.o_proj.forward(attn_output, q.shape[0], 1, self.num_local_heads, self.v_head_dim)
            else:
                output = self.o_proj.forward(attn_output, comm_group=comm_group)[0]
        return output

    def _forward_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        use_rmsnorm_rope_cache: bool = True
    ) -> torch.Tensor:
        if use_rmsnorm_rope_cache:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache

            q_len = 1
            dequant_scale_q_nope = None
            nz_block_size = 32 if self.fa_quant else 16
            if model_extra_config.operator_opt_config.use_mlaprolog:
                block_num, block_size, head_size, _ = key_cache.shape
                bsz, _ = hidden_states.view(-1, hidden_states.shape[-1]).shape
                if self.quant_symbol:
                    hidden_states_mla_prolog, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                else:
                    hidden_states_mla_prolog = hidden_states
                cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                cache_index = attn_metadata.slot_mapping.view(bsz, -1)

                q_nope, q_pe, k_nope, k_rope, dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x = hidden_states_mla_prolog.view(bsz, 1, -1),
                    weight_dq=self.q_a_proj.weight, weight_uq_qr=self.q_b_proj.weight,
                    weight_uk=self.W_UK, weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
                    rmsnorm_gamma_cq=self.q_a_layernorm.weight, rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
                    rope_sin=sin.squeeze(1), rope_cos=cos.squeeze(1), cache_index=cache_index,
                    kv_cache=key_cache.view(-1, 128, 1, 512), kr_cache=value_cache.view(-1, 128, 1, 64),
                    dequant_scale_x=pertoken_scale.view(-1, 1) if self.quant_symbol else None, # pertoken quant
                    dequant_scale_w_dq=self.q_a_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
                    dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
                    dequant_scale_w_dkv_kr=self.kv_a_proj_with_mqa.weight_scale.view(1, -1) if self.quant_symbol else None,
                    quant_scale_ckv=self.kv_scale_reci_tile,
                    quant_scale_ckr=None,
                    smooth_scales_cq=None,
                    rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
                    rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
                    cache_mode = "PA_NZ")

                k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size)
                k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim_nz, block_size, 16)
                q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                q_pe = q_pe.view(bsz, self.num_local_heads, -1)
            else:
                if self.q_lora_rank is not None:
                    q_lowrank = self.q_a_proj(hidden_states)[0]
                else:
                    q_lowrank = self.q_proj(hidden_states)[0]

                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('11'):
                        kv = self.kv_a_proj_with_mqa(hidden_states)[0]

                    tng.scope.npu_wait_tensor(q_lowrank, q_lowrank)
                else:
                    kv = self.kv_a_proj_with_mqa(hidden_states)[0]

                if self.q_lora_rank is not None:
                    q, _ = self.q_a_layernorm(q_lowrank, self.norm_res[q_lowrank.shape[0]])
                    q = self.q_b_proj(q)[0]
                else:
                    q = q_lowrank
                bsz, _ = q.shape
                q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d

                q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
                q_nope = (
                    torch.matmul(q_nope, self.W_UK)
                    .transpose(1, 0)
                    .view(bsz, q_len, self.num_local_heads, -1)
                )

                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    stream_context = tng.scope.npu_stream_switch('11')
                else:
                    stream_context = nullcontext()
                with stream_context:
                    kv = kv.unsqueeze(1).unsqueeze(1)
                    cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    tmp_slot_mapping = attn_metadata.slot_mapping
                    block_num, block_size, head_size, _ = key_cache.shape
                    k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                        kv, self.kv_a_layernorm.weight,
                        cos, sin, tmp_slot_mapping,
                        value_cache, key_cache,
                        c_kv_scale=self.kv_scale_reci_tile,
                        epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

                    # adapter nz
                    k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size)
                    k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

                    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                        tng.scope.npu_wait_tensor(q_pe, k_nope)
                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
                    if self.fa_quant:
                        q_nope, dequant_scale_q_nope = torch_npu.npu_dynamic_quant(q_nope.view(bsz * self.num_heads, self.kv_lora_rank))
                        dequant_scale_q_nope = dequant_scale_q_nope.view(bsz, self.num_heads)
                    q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                    q_pe = q_pe.view(bsz, self.num_local_heads, -1)

            bsz, _, q_dim = q_nope.size()
            input_layout = "TND_NTD"
            op_scope = tng.ops if self.enable_graph_mode else torch.ops.npu

            if self.fa_quant:
                assert dequant_scale_q_nope is not None
                dequant_scale_q_nope = dequant_scale_q_nope.squeeze(-1)
                attn_output, _ = op_scope.npu_fused_infer_attention_v2(
                    q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                    num_query_heads=self.num_heads, num_key_value_heads=1,
                    input_layout=input_layout, softmax_scale=self.scale,
                    dequant_scale_query=dequant_scale_q_nope, 
                    dequant_scale_key=self.kv_scale, dequant_scale_value=self.kv_scale,
                    query_quant_mode=3, inner_precise=0,
                    block_table=attn_metadata.decode.block_table,
                    block_size=128,
                    actual_seq_qlen=self.actual_seq_lengths[bsz],
                    actual_seq_kvlen=attn_metadata.decode.seq_lens
                )
            else:
                attn_output, _ = op_scope.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout=input_layout,
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths=self.actual_seq_lengths[bsz],
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )

            # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
            attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD

            attn_output = (
                torch.matmul(attn_output, self.W_UV)
                .transpose(1, 0)
                .reshape(bsz, q_len, -1)
            )
            attn_output = attn_output.view(
                -1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                output, _ = self.o_proj.forward(attn_output, bsz, q_len, self.num_local_heads, self.v_head_dim)
            else:
                output, _ = self.o_proj.forward(attn_output)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache

            if self.q_lora_rank is not None:
                q_lowrank = self.q_a_proj(hidden_states)[0]
            else:
                q_lowrank = self.q_proj(hidden_states)[0]

            kv = hidden_states
            kv = self.kv_a_proj_with_mqa(kv)[0]

            if self.q_lora_rank is not None:
                q = self.q_a_layernorm(q_lowrank)
                q = self.q_b_proj(q)[0]
            else:
                q = q_lowrank
            bsz, _ = q.shape
            q_len = 1
            q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d

            q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
            q_nope = (
                torch.matmul(q_nope, self.W_UK)
                .transpose(1, 0)
                .view(bsz, q_len, self.num_local_heads, -1)
            )

            kv = kv.unsqueeze(1).unsqueeze(1)
            cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            tmp_slot_mapping = attn_metadata.slot_mapping
            block_num, block_size, head_size, _ = key_cache.shape
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv, self.kv_a_layernorm.weight,
                cos, sin, tmp_slot_mapping,
                value_cache, key_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

            # adapter nz
            k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
            k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
            q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)

            bsz, q_len, _, q_dim = q_nope.size()
            if self.enable_graph_mode:
                attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout="BSND",
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout="BSND",
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )

            # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
            attn_output = attn_output.squeeze(1).transpose(0, 1)
            # attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
            # attn_output = pp_matmul(attn_output, self.W_UV, mm_type=4)
            attn_output = (
                torch.matmul(attn_output, self.W_UV)
                .transpose(1, 0)
                .reshape(bsz, q_len, -1)
            )
            attn_output = attn_output.view(
                -1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                output, _ = self.o_proj.forward(attn_output, bsz, q_len, self.num_local_heads, self.v_head_dim)
            else:
                output, _ = self.o_proj.forward(attn_output)
        return output
    
    def _forward_prefill_a2(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata is None:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos, sin = attn_metadata.prefill.cos, attn_metadata.prefill.sin

        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = all_gather_world(qkv, idx=0, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
                q = self.q_a_layernorm(q)
            else:
                if not isinstance(hidden_states, Dict):
                    h_quant, h_scale = torch_npu.npu_dynamic_quant(hidden_states)
                    hidden_states = {'x_int8': h_quant,
                                     'pertoken_scale':h_scale}
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

                latent_cache = all_gather_world(latent_cache, idx=0, dim=0)

                q = self.q_a_proj(hidden_states)[0]
                q = self.q_a_layernorm(q)
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                q_scale = all_gather_world(q_scale, idx=1, dim=0)
                q_quant = all_gather_world(q_quant, idx=1, dim=0)
                q = {'x_int8': q_quant,
                     'pertoken_scale': q_scale}

            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = all_gather_world(q, idx=0, dim=0)
            latent_cache = all_gather_world(latent_cache, idx=0, dim=0)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_nope, q_pe], dim=-1)

        if attn_metadata is not None:
            if isinstance(kv_cache, Dict):
                kv_cache = kv_cache.get("kv_cache")
            if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0: 
                _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                    latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                    self.kv_a_layernorm.weight,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    attn_metadata.slot_mapping,
                    kv_cache[1],
                    kv_cache[0],
                    epsilon=self.kv_a_layernorm.variance_epsilon,
                    cache_mode="PA_NZ",
                    is_output_kv=True)
            else:
                latent_cache = latent_cache.view(-1, latent_cache.size(-1))
                kv_a, k_pe = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_a = self.kv_a_layernorm(kv_a)
                k_pe = k_pe.unsqueeze(1)
                k_pe = k_pe.unsqueeze(2)
                k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
                k_pe = k_pe.squeeze(2)
            
            prefill_metadata = attn_metadata.prefill
            if len(prefill_metadata.seq_qlen_group) == 1:
                # normally execute
                actual_seq_qlen = prefill_metadata.seq_qlen_group[0] if prefill_metadata is not None else [q.shape[0]]
                actual_seq_kvlen = prefill_metadata.seq_kvlen_group[0] if prefill_metadata is not None else [q.shape[0]]

                kv = self.kv_b_proj.forward(kv_a)[0]
                kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)], dim=-1)

                if prefill_metadata.max_query_len > 1:
                    attn_mask = self.attn_mask
                else:
                    attn_mask = None
        
                if q.shape[0] != actual_seq_qlen[-1]:
                    actual_seq_qlen.append(q.shape[0])
                if k.shape[0] != actual_seq_kvlen[-1]:
                    actual_seq_kvlen.append(k.shape[0])

                attn_output = torch_npu.npu_fused_infer_attention_score(
                    q, k, v,
                    num_heads=self.num_local_heads,
                    input_layout="TND",
                    scale=self.scale,
                    sparse_mode=3,
                    atten_mask=attn_mask,
                    actual_seq_lengths=actual_seq_qlen,
                    actual_seq_lengths_kv=actual_seq_kvlen)[0].view(-1, self.num_local_heads, self.v_head_dim)

                q, k, v = None, None, None
                kv, k_nope = None, None
            else:
                attn_output = torch.empty(q.shape[0],
                                        self.num_local_heads,
                                        self.v_head_dim,
                                        device=q_nope.device,
                                        dtype=q_nope.dtype)
                computed_tokens = 0
                for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                        prefill_metadata.seq_qlen_group,
                        prefill_metadata.seq_kvlen_group)
                ):
                    prefill_q = q[computed_tokens:computed_tokens + actual_seq_qlen[-1]]
                    if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and \
                            kv_cache[0].numel() > 0:

                        block_num, block_size, head_size, _ = kv_cache[0].shape
                        kv_cache_a = (kv_cache[0]
                                    .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                        kv_cache_pe = (kv_cache[1]
                                    .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size,
                                            KVCACHE_NZ_DIM))
                        kv_cache_a = kv_cache_a.transpose(1, 3)
                        kv_cache_pe = kv_cache_pe.transpose(1, 3)

                        kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                            .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                        k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                            .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                    prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]

                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                    prefill_k = torch.cat(
                        [k_nope, prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)],
                        dim=-1)

                    if prefill_metadata.max_query_len > 1:
                        attn_mask = self.attn_mask
                    else:
                        attn_mask = None

                    prefill_v = v
                    attn_output[computed_tokens:computed_tokens + actual_seq_qlen[-1]] = \
                        torch_npu.npu_fused_infer_attention_score(
                            prefill_q,
                            prefill_k,
                            prefill_v,
                            num_heads=self.num_local_heads,
                            input_layout="TND",
                            scale=self.scale,
                            sparse_mode=3,
                            atten_mask=attn_mask,
                            actual_seq_lengths=actual_seq_qlen,
                            actual_seq_lengths_kv=actual_seq_kvlen)[0].view(-1, self.num_local_heads, self.v_head_dim)

                    computed_tokens += actual_seq_qlen[-1]
                    prefill_q, prefill_k, prefill_v = None, None, None
                    kv, k_nope = None, None,
                    q_nope, q_pe = None, None

            if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
                attn_output = attn_output.reshape(attn_output.shape[0], -1)
                attn_output = attn_output.reshape(self.tp_size // get_npu_device_count(), get_npu_device_count(),
                                                attn_output.shape[0] // self.tp_size, -1) \
                                        .transpose(0, 1).reshape(attn_output.shape[0], -1)
                attn_output = get_local_group_from_list(0).all_to_all(attn_output)
                output, _ = self.o_proj.forward(attn_output)
                output = reduce_scatter_cross(output, idx=0)
            else:
                attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
                output = self.o_proj.forward(attn_output)[0]
        else:
            attn_output = torch.zeros(q.shape[0],
                                      self.num_local_heads,
                                      self.v_head_dim,
                                      device=q_nope.device,
                                      dtype=q_nope.dtype)
            if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
                attn_output = attn_output.reshape(attn_output.shape[0], -1)
                attn_output = attn_output.reshape(self.tp_size // get_npu_device_count(), get_npu_device_count(),
                                                attn_output.shape[0] // self.tp_size, -1) \
                                        .transpose(0, 1).reshape(attn_output.shape[0], -1)
                attn_output = get_local_group_from_list(0).all_to_all(attn_output)
                output, _ = self.o_proj.forward(attn_output)
                output = reduce_scatter_cross(output, idx=0)
            else:
                attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
                output = self.o_proj.forward(attn_output)[0]

        attn_output = None
        return output