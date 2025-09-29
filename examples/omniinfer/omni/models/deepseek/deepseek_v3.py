# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

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
"""Inference-only DeepseekV3 model."""
import copy
import itertools
from typing import Iterable, List, Optional, Set, Tuple, Union
import torch
from torch import nn
from transformers import PretrainedConfig
import torch.distributed as dist
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    make_empty_intermediate_tensors_factory,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from omni.models.common.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear,
)
from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding
)
from omni.models.common.layers.activation import SiluAndMul
from omni.models.common.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.parallel_state import (
    get_stream1_attn_group,
    get_stream1_mlp_group,
    get_stream1_moe_group,
    get_mlp_tp_group,
    GroupCoordinator
)
import torch.nn.functional as F

from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.layers.moe.deepseek_moe import DeepseekMoE 
from omni.models.common.layers.attention.deepseek_mla import DeepseekMLA 
from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.attention.backend.mla import group_request_list

if model_extra_config.operator_opt_config.unquant_bmm_nz:
    # if use weight nz, this config must be True
    torch.npu.config.allow_internal_format = True

"""MLP module activation split length, split by 64G VRAM, need to confirm the optimal split length based on sequence length and performance"""
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64

def _get_pad_size(num_seqs, split_size):
    """Calculate padding size needed to make sequence count divisible by tp size."""
    return (split_size - num_seqs % split_size) % split_size

def pad_inputs(input, query_lens, sp_size):
    padded_lengths = _get_pad_size(query_lens, sp_size)
    segments = []
    start_idx = 0
    for length, pad_length in zip(query_lens, padded_lengths):
        segment = input[start_idx : start_idx + length]
        padded_segment = F.pad(segment, (0, 0, 0, pad_length), "constant", 0)
        segments.append(padded_segment)
        start_idx += length

    return torch.cat(segments, dim=0)

# todo query_lens是tensor，需要适配
def generate_sp_inputs(hidden_states, attn_metadata):
    sp_size = model_extra_config.parall_config.attn_sp_size
    if attn_metadata is not None:
        hidden_states = pad_inputs(hidden_states, attn_metadata.prefill.actual_query_lens, sp_size * 2)
        # split input for sp attention
        hidden_states_list = torch.split(hidden_states, attn_metadata.prefill.sp_split_list, dim=0)
        hidden_states = torch.cat([hidden_states_list[i] for i in attn_metadata.prefill.sp_zigzag_index], dim=0)
    else:
        hidden_states = torch.split(hidden_states, hidden_states.size(0) // sp_size, dim=0)[get_tensor_model_parallel_rank()]
    return hidden_states

class ParallelDeepseekMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.gate_up_proj = AscendMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            tp_size=get_mlp_tp_group().world_size,
            tp_rank=get_mlp_tp_group().rank_in_group,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           tp_size=get_mlp_tp_group().world_size,
                                           tp_rank=get_mlp_tp_group().rank_in_group,
                                           quant_config=quant_config,
                                           reduce_results=False,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)


    def forward(self, x, residual, attn_metadata, layerid=None):
        x = get_mlp_tp_group().all_gather(x, dim=0)

        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        # P and D are both cut, and are concave at the node (16)
        x = get_mlp_tp_group().reduce_scatter(x)
        return x, residual


class DeepseekDecoderLayer(nn.Module):

    is_split_hidden_states = False

    def __init__(
            self,
            config: PretrainedConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        self.quant_symbol = quant_config is not None
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.self_attn = DeepseekMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = ParallelDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None,
            next_attention_weights: Optional[dict] = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            quant_symbol = (self.quant_symbol and not model_extra_config.operator_opt_config.use_mlaprolog and not model_extra_config.operator_opt_config.enable_dsa)
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=quant_symbol)
            # Adapt end.
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        is_prefill = attn_metadata is None or attn_metadata.prefill is not None

        if self.is_moe == True and not is_prefill and model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.mlp.prefix, 'stream-fusion=1'):
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # hidden : tokens * 7168

        # Perform full hidden splitting to avoid OOM
        if (model_extra_config.parall_config.dp_size > 1 or DeepseekDecoderLayer.is_split_hidden_states) and is_prefill:
            # During prefill, chunk is only triggered when an extremely large number of identical tokens is detected — to prevent GMM from OOM. 
            # Prefill performance may degrade slightly as a trade-off. 
            # For longer sequences (e.g., >256K or 512K tokens), consider adjusting SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER to optimize memory usage or avoid OOM.
            reduce_length = torch.tensor(hidden_states.shape[0], dtype=torch.int64, device=current_platform.device_type)
            local_length = hidden_states.shape[0]
            # global_max_length = torch.tensor(0, dtype=torch.int64)
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - hidden_states.shape[0]
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )
            residual = torch.nn.functional.pad(
                residual, (0, 0, 0, pad_size)
            )
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            residual_list = residual.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            hidden_state_out = []
            residual_out = []
            for i in range(len(hidden_states_list)):
                hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata, layer_id)
                hidden_state_out.append(hidden_states)
                residual_out.append(residual)
            hidden_states = torch.cat(hidden_state_out)[:local_length]
            residual = torch.cat(residual_out)[:local_length]
        else:
            if self.is_moe == True:
                # omni placement do not support super kernel
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
                if isinstance(hidden_states, (tuple, list)):
                    assert len(hidden_states) == 2
                    hidden_states = hidden_states[0] + hidden_states[1]
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

        return hidden_states, residual

    def forward_attn(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            comm_group: Optional[GroupCoordinator] = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=(not model_extra_config.operator_opt_config.use_mlaprolog))
            # Adapt end.
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            comm_group=comm_group
        )

        return hidden_states, residual

    def forward_mlp(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None,
            next_attention_weights: Optional[dict] = None,
            comm_group: Optional[GroupCoordinator] = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        is_prefill = attn_metadata is None or attn_metadata.prefill is not None

        if self.is_moe == True and not is_prefill and model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.mlp.prefix, 'stream-fusion=1'):
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_moe == True:
            # omni placement do not support super kernel
            hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id, next_attention_weights, comm_group=comm_group)
            if isinstance(hidden_states, (tuple, list)):
                assert len(hidden_states) == 2
                # 0 is the shared expert hidden_states, 1 is the routing expert hidden_states, add operation cannot be placed in the super kernel
                hidden_states = hidden_states[0] + hidden_states[1]
        else:
            hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, comm_group=comm_group)

        return hidden_states, residual


class DeepseekV3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.prefix = f"{prefix}.layers"
        self.postfix = ".self_attn.attn"
        self.tp_size = get_tensor_model_parallel_world_size()
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekDecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if model_extra_config.operator_opt_config.enable_prefill_micro_batch:
            self.stream1 = torch.npu.Stream()
            self.stream1_attn_group = get_stream1_attn_group()
            self.stream1_mlp_group = get_stream1_mlp_group()
            self.stream1_moe_group = get_stream1_moe_group()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=0 if model_extra_config.parall_config.attn_sp_size > 1 else 1)

    def should_split_hidden_states(self, input_ids: torch.Tensor, ratio_threshold: float, count_threshold: int) -> bool:
        is_split_hidden_states = False
        if ratio_threshold == 0.0 or count_threshold == 0:
            return is_split_hidden_states
        flattened = input_ids.view(-1)
        min_val = flattened.min()
        if min_val.item() < 0:
            flattened = flattened - min_val # Ensure tensor is non-negative
        counts = torch.bincount(flattened)
        max_count = counts.max().item()
        total = flattened.numel() 
        if total == 0:
            return is_split_hidden_states
        max_token_ratio = max_count / total
        # Split hidden_states if token count or ratio exceeds threshold, to prevent GMM OOM in MoE.
        is_split_hidden_states = max_token_ratio >= ratio_threshold or max_count >= count_threshold
        return is_split_hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            max_num_tokens=None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        attn_metadata_first = self.get_layer_attn_metadata(attn_metadata, 0)

        if model_extra_config.operator_opt_config.use_omni_cache:
            if attn_metadata_first is not None and attn_metadata_first.prefill is not None:
                attn_metadata_first.omni_cache.synchronize_h2d(
                    prefix_meta=attn_metadata_first.prefill.prefix_meta,
                    layer_idx=0,
                )

        if model_extra_config.operator_opt_config.enable_prefill_micro_batch and \
            attn_metadata is not None and attn_metadata_first is not None \
            and attn_metadata_first.prefill is not None and \
            len(attn_metadata_first.prefill.seq_lens) > 1:
            return self.forward_micro_batch(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, max_num_tokens)
        else:
            return self.forward_normal(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)

    def forward_normal(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        attn_metadata_first = self.get_layer_attn_metadata(attn_metadata, 0)
        is_prefill = attn_metadata is None or (attn_metadata_first is not None and attn_metadata_first.prefill is not None)
        if is_prefill:
            DeepseekDecoderLayer.is_split_hidden_states = False
        if get_pp_group().is_first_rank:
            if input_ids.numel() >= model_extra_config.operator_opt_config.max_split_token_count_threshold and \
                    is_prefill and \
                    kv_caches is not None:
                DeepseekDecoderLayer.is_split_hidden_states = self.should_split_hidden_states(
                    input_ids,
                    model_extra_config.operator_opt_config.max_split_token_ratio_threshold,
                    model_extra_config.operator_opt_config.max_split_token_count_threshold
                )

            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        attn_metadata = self.get_layer_attn_metadata(attn_metadata, 0)
        if is_prefill and model_extra_config.parall_config.attn_sp_size > 1:
            # split input for sp attention
            hidden_states = generate_sp_inputs(hidden_states, attn_metadata)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - 3

            if i >= self.first_k_dense_replace and i < self.end_layer - 1:
                next_attention_weights = {
                    'q_a_proj_weight': self.layers[i + 1].self_attn.q_a_proj.weight,
                    'kv_a_proj_with_mqa_weight': self.layers[i + 1].self_attn.kv_a_proj_with_mqa.weight,
                    'q_b_proj_weight': self.layers[i + 1].self_attn.q_b_proj.weight,
                    'W_UK': self.layers[i + 1].self_attn.W_UK
                }
            else:
                next_attention_weights = {
                    'q_a_proj_weight': None,
                    'kv_a_proj_with_mqa_weight': None,
                    'q_b_proj_weight': None,
                    'W_UK': None
                }
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata,
                                            residual,
                                            layer_id,
                                            next_attention_weights)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        if model_extra_config.parall_config.attn_sp_size > 1 and is_prefill:
            # reverse sp split
            if attn_metadata is not None:
                prefill_meta = attn_metadata.prefill
                outputs_list = torch.split(hidden_states, prefill_meta.sp_reverse_split_list, dim=0)
                hidden_states = torch.cat([outputs_list[i] for i in prefill_meta.sp_reverse_index], dim=0)

        return hidden_states

    def forward_micro_batch(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            max_num_tokens=None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Split requests into two micro-batches (batch0, batch1) with balanced tokens
        seq_lens_temp = self.get_layer_attn_metadata(attn_metadata, 0).prefill.seq_lens
        seq_len_left, seq_len_right, split_idx = self.partition_list(seq_lens_temp, sum(seq_lens_temp))
        input_ids_mb0, input_ids_mb1 = self.index_batch(input_ids, 0, sum(seq_len_left))
        positions_mb0, positions_mb1 = self.index_batch(positions, 0, sum(seq_len_left))
        residual_mb0 = None
        residual_mb1 = None
        curr_stream = torch.npu.current_stream()

        if get_pp_group().is_first_rank:
            # Perform embedding ops on separate streams while maintaining execution order within each stream
            # Optimized execution order:
            # 1. stream0: attn (current layer)
            # 2. stream1: attn (current layer)
            # 3. stream0: mlp (current layer) + attn (next layer)
            # 4. stream1: mlp (current layer) + attn (next layer)
            with torch.npu.stream(curr_stream):
                pad_size_mb0 = _get_pad_size(positions_mb0.shape[0], self.tp_size)
                positions_mb0 = self.pad_tensor(positions_mb0, pad_size_mb0, 0)
                padding = torch.randint(1, self.vocab_size, (pad_size_mb0,),
                                        dtype=input_ids.dtype,
                                        device=input_ids.device)
                input_ids_mb0 = torch.cat([input_ids_mb0, padding])
                metadata0 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, 0), True, sum(seq_len_left), pad_size_mb0, max_num_tokens)
                hidden_states_mb0 = self.get_input_embeddings(input_ids_mb0)
                hidden_states_mb0, residual_mb0 = self.layers[0].forward_attn(positions_mb0,
                                                                              hidden_states_mb0,
                                                                              kv_caches[0] if kv_caches is not None else None,
                                                                              metadata0,
                                                                              residual_mb0)
            with torch.npu.stream(self.stream1):
                pad_size_mb1 = _get_pad_size(positions_mb1.shape[0], self.tp_size)
                positions_mb1 = self.pad_tensor(positions_mb1, pad_size_mb1, 0)
                padding = torch.randint(1, self.vocab_size, (pad_size_mb1,),
                                        dtype=input_ids.dtype,
                                        device=input_ids.device)
                input_ids1 = torch.cat([input_ids_mb1, padding])
                metadata1 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, 0), False, sum(seq_len_left), pad_size_mb1, max_num_tokens)
                hidden_states_mb1 = self.get_input_embeddings(input_ids1)
                hidden_states_mb1, residual_mb1 = self.layers[0].forward_attn(positions_mb1,
                                                                              hidden_states_mb1,
                                                                              kv_caches[0] if kv_caches is not None else None,
                                                                              metadata1,
                                                                              residual_mb1,
                                                                              comm_group=self.stream1_attn_group)
        else:
            assert intermediate_tensors is not None, "intermediate_tensors is None"
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - 3
            if i >= self.first_k_dense_replace and i < self.end_layer - 1:
                next_attention_weights = {
                    'q_a_proj_weight': self.layers[i + 1].self_attn.q_a_proj.weight,
                    'kv_a_proj_with_mqa_weight': self.layers[i + 1].self_attn.kv_a_proj_with_mqa.weight,
                    'q_b_proj_weight': self.layers[i + 1].self_attn.q_b_proj.weight,
                    'W_UK': self.layers[i + 1].self_attn.W_UK
                }
            else:
                next_attention_weights = {
                    'q_a_proj_weight': None,
                    'kv_a_proj_with_mqa_weight': None,
                    'q_b_proj_weight': None,
                    'W_UK': None
                }
            if i < self.first_k_dense_replace:
                stream1_mlp_comm_group = self.stream1_mlp_group
            else:
                stream1_mlp_comm_group = self.stream1_moe_group

            with torch.npu.stream(curr_stream):
                metadata0 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, i), True, sum(seq_len_left), pad_size_mb0, max_num_tokens)
                hidden_states_mb0, residual_mb0 = layer.forward_mlp(hidden_states_mb0,
                                                                    metadata0,
                                                                    residual_mb0,
                                                                    layer_id,
                                                                    next_attention_weights)
            if (i + 1) in range(self.start_layer, self.end_layer):
                with torch.npu.stream(curr_stream):
                    metadata0 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, i+1), True,
                                                               sum(seq_len_left), pad_size_mb0, max_num_tokens)
                    hidden_states_mb0, residual_mb0 = self.layers[i+1].forward_attn(positions_mb0,
                                                                                  hidden_states_mb0,
                                                                                  kv_caches[i + 1 - self.start_layer] if kv_caches is not None else None,
                                                                                  metadata0,
                                                                                  residual_mb0)
            with torch.npu.stream(self.stream1):
                metadata1 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, i), False, sum(seq_len_left), pad_size_mb1, max_num_tokens)
                hidden_states_mb1, residual_mb1 = layer.forward_mlp(hidden_states_mb1,
                                                                    metadata1,
                                                                    residual_mb1,
                                                                    layer_id,
                                                                    next_attention_weights,
                                                                    comm_group=stream1_mlp_comm_group)
            if (i + 1) in range(self.start_layer, self.end_layer):
                with torch.npu.stream(self.stream1):
                    metadata1 = self.split_attn_metadata_index(self.get_layer_attn_metadata(attn_metadata, i+1), False,
                                                               sum(seq_len_left), pad_size_mb1, max_num_tokens)
                    hidden_states_mb1, residual_mb1 = self.layers[i+1].forward_attn(positions_mb1,
                                                                                  hidden_states_mb1,
                                                                                  kv_caches[i + 1 - self.start_layer] if kv_caches is not None else None,
                                                                                  metadata1,
                                                                                  residual_mb1,
                                                                                  comm_group=self.stream1_attn_group)

        curr_stream.wait_stream(self.stream1)
        self.stream1.wait_stream(curr_stream)
        hidden_states_mb0, _ = self.norm(hidden_states_mb0, residual_mb0)
        hidden_states_mb1, _ = self.norm(hidden_states_mb1, residual_mb1)
        hidden_states_mb0 = tensor_model_parallel_all_gather(hidden_states_mb0, dim=0)
        hidden_states_mb1 = tensor_model_parallel_all_gather(hidden_states_mb1, dim=0)
        # Calculate original sequence lengths by removing padding
        original_size_0 = hidden_states_mb0.shape[0] - pad_size_mb0
        original_size_1 = hidden_states_mb1.shape[0] - pad_size_mb1
        # Remove padding from each micro batch
        hidden_states_mb0 = hidden_states_mb0[:original_size_0]
        hidden_states_mb1 = hidden_states_mb1[:original_size_1]
        hidden_states = torch.cat([hidden_states_mb0, hidden_states_mb1], dim=0)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        return hidden_states

    def get_layer_attn_metadata(self, attn_metadata, layer_idx):
        if attn_metadata is None:
            return None
        if isinstance(attn_metadata, dict):
            key_idx = self.prefix + "." + str(layer_idx) + self.postfix
            return attn_metadata[key_idx]

    def index_batch(self, ori_tensor, split_dim, split_idx):
        """Split tensor into two parts along specified dimension at split index."""
        return torch.split(ori_tensor, [split_idx, ori_tensor.size(split_dim) - split_idx], dim=split_dim)

    def partition_list(self, lst, pos):
        """Partition list into two parts with balanced sum around target position."""
        target = pos // 2
        left = []
        right = []
        cur_sum = 0
        split_index = 0

        for i, num in enumerate(lst):
            if cur_sum < target:
                left.append(num)
                cur_sum += num
                split_index = i + 1
            else:
                right.append(num)

        if not right:
            right.append(left.pop())
            split_index -= 1
        # try to make tokens processed by two streams as evently as possible
        gap1 = abs(sum(left) - sum(right))
        gap2 = abs(sum(left) - sum(right) - 2 * left[-1])
        if gap1 < gap2:
            return left, right, split_index
        else:
            right.insert(0, left.pop())
            return left, right, split_index - 1

    def pad_tensor(self, tensor, pad_size, pad_value=0):
        """Pad tensor with specified value along first dimension."""
        padded_shape = (pad_size, tensor.shape[-1]) if tensor.dim() > 1 else (pad_size,)
        padding = torch.full(
            padded_shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device
        )
        return torch.cat([tensor, padding])

    def split_attn_metadata_index(self, metadata, is_local_stream, split_idx, pad_size, max_num_tokens):
        """Split attention metadata for parallel processing across streams.

        Args:
            metadata: Original attention metadata
            is_local_stream: Flag indicating local stream processing
            split_idx: Index to split metadata
            pad_size: Padding size for alignment
            max_num_tokens: Maximum number of tokens

        Returns:
            Modified metadata split for specified stream
        """
        slot_mapping = metadata.slot_mapping
        seq_lens = metadata.prefill.seq_lens
        query_lens = metadata.prefill.query_lens
        block_table = metadata.prefill.block_table

        slot_mapping1, slot_mapping2 = self.index_batch(slot_mapping, 0, split_idx)
        seq_lens1, seq_lens2, _ = self.partition_list(seq_lens, sum(seq_lens))
        query_lens1, query_lens2, _ = self.partition_list(query_lens, sum(query_lens))
        if is_local_stream:
            metadata_out = self.refresh_metadata(slot_mapping1, pad_size, seq_lens1, query_lens1, block_table, max_num_tokens, metadata)
        else:
            metadata_out = self.refresh_metadata(slot_mapping2, pad_size, seq_lens2, query_lens2, block_table, max_num_tokens, metadata)
        return metadata_out

    def refresh_metadata(self, slot_mapping, pad_size, seq_lens, query_lens, block_table, max_num_tokens, metadata):
        metadata_out = copy.deepcopy(metadata)
        slot_mapping = self.pad_tensor(slot_mapping, pad_size, pad_value=-1)
        seq_kvlen_group, seq_qlen_group, _ = group_request_list(
            seq_lens,
            query_lens,
            block_table,
            max_num_tokens)
        seq_qlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_qlen_group]
        seq_kvlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_kvlen_group]
        metadata_out.slot_mapping = slot_mapping
        metadata_out.prefill.seq_lens = seq_lens
        metadata_out.prefill.query_lens = query_lens
        metadata_out.prefill.seq_qlen_group = seq_qlen_group
        metadata_out.prefill.seq_kvlen_group = seq_kvlen_group
        return metadata_out


@support_torch_compile
class DeepseekV3ForCausalLM(nn.Module):

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV3Model(vllm_config=vllm_config, prefix="model")

        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
									  parallel_lmhead=(model_extra_config.parall_config.dp_size > 1))
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.return_hidden_states = True
        self.max_num_token = vllm_config.scheduler_config.max_num_batched_tokens

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: Union[AttentionMetadata, dict] = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors, self.max_num_token)
        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        if self.return_hidden_states:
            return hidden_states, logits
        else:
            return logits

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        # Get the logits for the next tokens.
        logits = self.lm_head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        if model_extra_config.operator_opt_config.merge_qkv:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("qkv_a_proj", "q_a_proj", 0),
                ("qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ]
        else:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'DeepseekV3ForCausalLM' and self.config.num_nextn_predict_layers > 0:
                mtp_prefix = [f"model.layers.{self.config.num_hidden_layers + layer_idx}" for layer_idx in range(self.config.num_nextn_predict_layers)]
                if name.startswith(tuple(mtp_prefix)):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        if attn_metadata.prefill:
            return True

        return False
