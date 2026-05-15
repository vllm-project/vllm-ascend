# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only DeepseekV2/DeepseekV3 model."""

import math
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.interfaces import (MixtureOfExperts,
                                                   SupportsEagle, SupportsLoRA,
                                                   SupportsPP)
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix,
    sequence_parallel_chunk)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

logger = init_logger(__name__)


def hadamard_transform_ref(x: torch.Tensor, scale=1.0):
    from scipy.linalg import hadamard
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(
        x,
        torch.tensor(hadamard(dim_padded, dtype=float),
                     dtype=x.dtype,
                     device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size**-0.5)


def precompute_freqs_cis_cpu(dim, seqlen, original_seq_len, base, factor,
                             beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(
            max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max -
                                                                        min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                          original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor,
                     freqs_cis: torch.Tensor,
                     inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis.to(x.device)).flatten(-2)
    y.copy_(x)
    return y


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config
                                        | DeepseekV3Config,
                                        weight_name: str) -> int | None:
    if weight_name.startswith("mtp."):
        return 0
    return None


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           disable_tp=is_sequence_parallel,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):

    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor",
                                             1.5)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled())
        self.is_fusion_moe_shared_experts_enabled = getattr(
            get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.hash = layer_idx < config.n_hash_layers and not is_draft_layer
        if self.hash:
            self.gate.tid2eid = nn.Parameter(torch.empty(
                config.vocab_size,
                config.n_activated_experts,
                dtype=torch.int32),
                                             requires_grad=False)
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32))

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_inter_dim,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "score_func", "softmax"),
            # we do scaling outside, set factor to 1.0 to avoid double mul
            # aiter applies routed_scaling_factor internally
            # routed_scaling_factor=1.5,
            routed_scaling_factor=1.0 if not self.is_rocm_aiter_moe_enabled
            else self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts
            if self.is_fusion_moe_shared_experts_enabled else 0,
            hash=layer_idx < config.n_hash_layers and not is_draft_layer,
            tid2eid=self.gate.tid2eid)

    def forward(self,
                hidden_states: torch.Tensor,
                input_ids=None) -> torch.Tensor:

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            fused_moe_out = self.experts(hidden_states=hidden_states,
                                         router_logits=hidden_states)
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits = F.linear(hidden_states.float(), self.gate.weight)
            fused_moe_out = self.experts(hidden_states=hidden_states,
                                         router_logits=router_logits)

        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        # Fix FP16 overflow
        # See DeepseekV2DecoderLayer for more details.
        if hidden_states.dtype != torch.float16:
            if not self.is_rocm_aiter_moe_enabled:
                if self.shared_experts is not None:
                    assert shared_output is not None
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output,
                        self.routed_scaling_factor)
                else:
                    final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states = muls_add_triton(
                shared_output, final_hidden_states,
                1.0 / self.routed_scaling_factor)

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(original_max_position_embeddings: int,
                         scaling_beta: float,
                         positions: torch.Tensor) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


class Indexer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        self.compressor = Compressor(
            vllm_config,
            config,
            compress_ratio,
            self.head_dim,
            True,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.compressor")  #Compressor(4, 128)

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor, positions,
                rotary_emb) -> torch.Tensor:
        return


class Compressor(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = head_dim - config.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = config.norm_eps
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio,
                        coff * self.head_dim,
                        dtype=torch.float32))
        self.wkv = ReplicatedLinear(
            self.dim,
            coff * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.wgate = ReplicatedLinear(
            self.dim,
            coff * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wgate",
            return_bias=False,
        )
        self.norm = RMSNorm(self.head_dim, config.norm_eps)

        self.kv_cache = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_state=None,
    ) -> torch.Tensor:
        pass

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        dtype = x.dtype
        if inverse:
            sin = sin * -1
        tnd_layout = 1
        if len(x.shape) == 3:
            num_tokens, num_heads, rotary_dim = x.shape
        else:
            tnd_layout = 0
            _, num_tokens, num_heads, rotary_dim = x.shape
        x_rot = torch_npu.npu_rotary_mul(x.reshape(
            num_tokens, num_heads, 1, rotary_dim).to(torch.float32),
                                         cos,
                                         sin,
                                         rotary_mode="interleave")
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1, num_tokens, -1, rotary_dim)
        return x.to(dtype)


class DeepseekV4Attention(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        max_position_embeddings: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        tp_size = get_tensor_model_parallel_world_size()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.head_dim - config.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.window_size
        self.eps = config.norm_eps
        self.norm_eps = config.norm_eps
        self.scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
            return_bias=False,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_a",
            return_bias=False,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * config.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
            return_bias=False,
        )
        self.compress_ratio = config.compress_ratios[layer_idx]

        if self.compress_ratio > 1:
            config.rope_parameters['rope_theta'] = 160000
            rope_groups = ['default', f'c{self.compress_ratio}']
        else:
            config.rope_parameters['rope_theta'] = 10000
            rope_groups = ['default']
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f'{prefix}.attn',
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=config.
            rope_parameters['original_max_position_embeddings']
            if self.compress_ratio > 0 else 0,
            is_neox_style=False,
            scaling_factor=config.rope_parameters['factor'],
            base=config.rope_parameters['rope_theta'],
            beta_fast=config.rope_parameters['beta_fast'],
            beta_slow=config.rope_parameters['beta_slow'],
            rope_groups=rope_groups)

        self.compressor: Optional[Compressor] = None
        self.indexer: Optional[Indexer] = None

        if self.compress_ratio > 1:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                self.head_dim,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(
                    vllm_config,
                    config,
                    self.compress_ratio,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    prefix=f"{prefix}.indexer",
                )

        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            # prefix=f'{prefix}.attn',
            prefix=f'{prefix}',
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dsa_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps

        attn_cls = DeepseekV4Attention

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mlp = DeepseekV4MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor",
                                             1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor,
               hc_scale: torch.Tensor, hc_base: torch.Tensor):
        y = torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base,
                                           self.hc_mult,
                                           self.hc_sinkhorn_iters,
                                           self.norm_eps, self.hc_eps)
        return y

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor):
        y = torch.ops._C_ascend.npu_hc_post(x.unsqueeze(dim=0),
                                            residual.unsqueeze(dim=0),
                                            post.unsqueeze(dim=0),
                                            comb.unsqueeze(dim=0))
        return y.squeeze(dim=0)

    def forward(self,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor | None,
                llama_4_scaling: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn,
                                                self.hc_attn_scale,
                                                self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
            "llama_4_scaling": llama_4_scaling
        }
        hidden_states = self.self_attn(**attn_kwargs)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn,
                                                self.hc_ffn_scale,
                                                self.hc_ffn_base)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        return hidden_states, residual


@support_torch_compile
class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

        self.norm_eps = config.norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(
            torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor,
                hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = None
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult,
                                                          1)  #(b,s, c, h)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            llama_4_scaling)
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn,
                                     self.hc_head_scale, self.hc_head_base)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV2MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV4MoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekV4MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning(
                "DeepSeekV2: No DeepseekV4MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP,
                                  DeepseekV2MixtureOfExperts, SupportsLoRA,
                                  SupportsEagle):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        # Set MoE hyperparameters
        self.num_moe_layers = (self.config.num_hidden_layers)
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts +
            (self.config.n_shared_experts
             if getattr(get_ascend_config(), "mix_placement", False) else 0),
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled())
        rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(),
                                                       "mix_placement", False)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts +
            (self.config.n_shared_experts
             if rocm_aiter_moe_shared_expert_enabled else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # TODO:
            if not name.startswith('model'):
                name = f'model.{name}'

            if '.w1.' in name:
                name = name.replace('.w1.', '.gate_proj.')
            if '.w2.' in name:
                name = name.replace('.w2.', '.down_proj.')
            if '.w3.' in name:
                name = name.replace('.w3.', '.up_proj.')

            if 'model.head.' in name and 'model.lm_head.' not in name:
                name = name.replace('model.head.', 'lm_head.')
            if 'model.lm_head.' in name:
                name = name.replace('model.lm_head.', 'lm_head.')
            if 'embed.' in name and 'embed_token.' not in name:
                name = name.replace('embed.', 'embed_tokens.')
            if 'attn' in name and 'self_attn' not in name:
                name = name.replace('.attn.', '.self_attn.')
            if '.ffn.' in name:
                name = name.replace('.ffn.', '.mlp.')
            if '.ffn_norm.' in name:
                name = name.replace('.ffn_norm.', '.post_attention_layernorm.')
            if '.attn_norm.' in name:
                name = name.replace('.attn_norm.', '.input_layernorm.')

            if "rotary_emb.inv_freq" in name:
                continue
            if ".gate.bias" in name:
                name = name.replace('.gate.bias',
                                    '.gate.e_score_correction_bias')

            if "sink" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = loaded_weight.narrow(0, head_start,
                                                     heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled
                and ("mlp.shared_experts" in name))

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (param_name == "fused_qkv_a_proj"
                    ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts",
                                         1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}")
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j *
                                                           chunk_size:(j + 1) *
                                                           chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j *
                                                           chunk_size:(j + 1) *
                                                           chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(
                            weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(Callable[..., bool],
                                                    param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params
