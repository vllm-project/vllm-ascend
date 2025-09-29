# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only PanguUltraMoE model."""
from typing import Iterable, List, Optional, Set, Tuple, Union, Literal
import os
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torch.distributed as dist
import torchair._contrib.custom_torch_ops

from vllm.platforms import current_platform
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import AttentionMetadata
from vllm.distributed import (get_pp_group,
                              get_ep_group,
                              tensor_model_parallel_all_gather,
                              get_world_group)

from vllm.config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter, make_layers, make_empty_intermediate_tensors_factory)

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from omni.models.common.layers.activation import SiluAndMul
from omni.models.common.layers.layernorm import RMSNorm

from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)

from omni.models.common.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear
)
from omni.adaptors.vllm.distributed.communication_op import (
    reduce_scatter_two_stage, all_gather_two_stage, all_gather_local, reduce_scatter_local,
    reduce_scatter_pipeline, all_gather_pipeline,
    reduce_scatter_round_pipeline, all_gather_round_pipeline)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_npu_device_count,
    get_local_group_size, 
    get_local_group_rank
)

from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.layers.attention.deepseek_mla import DeepseekMLA
from omni.models.common.layers.moe.deepseek_moe import DeepseekMoE
from omni.models.common.config.model_config import model_extra_config

"""MLP 模块激活拆分长度，按64G显存拆分，需要根据序列长度以及性能确认最佳拆分长度"""
SEQ_SPLIT_LENGTH = 4096
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64 if model_extra_config.operator_opt_config.prefill_moe_all_to_all else 256


class ParallelPanguUltraMoEMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            tp_parallel: Literal["global", "local", "no_tp"] = "no_tp",
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_parallel = tp_parallel

        if tp_parallel == "local":
            self.tp_size = get_local_group_size()
            self.tp_rank = get_local_group_rank()
        elif tp_parallel == "global":
            self.tp_size = get_ep_group().world_size
            self.tp_rank = get_ep_group().rank_in_group
        elif tp_parallel == "no_tp":
            self.tp_size = 1
            self.tp_rank = 0

        self.gate_up_proj = AscendMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        if os.environ["ROLE"] == "decode":
            self.gate_up_proj.throw_dequant = True
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                                 hidden_size,
                                                 tp_size=self.tp_size,
                                                 tp_rank=self.tp_rank,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 reduce_results=False,
                                                 prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False
        self.device_count = get_npu_device_count()
        self.node_rank = get_world_group().rank_in_group // get_npu_device_count()
        self.which_half = get_world_group().rank_in_group // (get_world_group().world_size // 2)

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(self, x, residual, attn_metadata, pertoken_scale=None, no_communication=False):
        if self.tp_parallel == "no_tp" or no_communication:
            return self.forward_no_tp(x, residual, attn_metadata, pertoken_scale)

        if self.tp_parallel == "local":
            return self.forward_local_tp(x, residual, attn_metadata)

        if self.tp_parallel == "global":
            return self.forward_global_tp(x, residual, attn_metadata)

    def forward_no_tp(self, x, residual, attn_metadata, pertoken_scale=None):
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        x = {'x_int8': x,
             'pertoken_scale': pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        return x, residual

    def forward_local_tp(self, x, residual, attn_metadata):
        pad_size = 0
        is_prefill = (attn_metadata is None or attn_metadata.prefill)
        if is_prefill and model_extra_config.parall_config.dp_size > 1:
            local_length = x.shape[0]
            reduce_length = torch.tensor(x.shape[0], dtype=torch.int64, device=current_platform.device_type)
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - x.shape[0]

            x = torch.nn.functional.pad(
                x, (0, 0, 0, pad_size)
            )

        x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        x = all_gather_local(x, idx=0, dim=0)
        pertoken_scale = all_gather_local(pertoken_scale, idx=1, dim=0)

        x = {'x_int8': x,
             'pertoken_scale': pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        x = reduce_scatter_local(x, idx=0)

        if is_prefill and pad_size > 0:
            x = x[:local_length, :]
        return x, residual

    def forward_global_tp(self, x, residual, attn_metadata):
        is_prefill = (attn_metadata is None or attn_metadata.prefill)
        if not is_prefill:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            global_pertoken_scale = all_gather_two_stage(pertoken_scale, idx=1, dim=0, reverse=True)
            if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                x = all_gather_round_pipeline(x, idx=0, node_rank=self.node_rank, dim=0)
            elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                x = all_gather_pipeline(x, idx=0, which_half=self.which_half, dim=0)
                global_pertoken_scale = global_pertoken_scale.view(2, -1, self.device_count, pertoken_scale.shape[0]) \
                    .transpose(1, 2).reshape(-1)
            else:
                x = all_gather_two_stage(x, idx=0, dim=0)
                global_pertoken_scale = global_pertoken_scale.view(-1, self.device_count, pertoken_scale.shape[0]) \
                    .transpose(0, 1).reshape(-1)
        else:
            pad_size = 0
            if model_extra_config.parall_config.dp_size > 1:
                local_length = x.shape[0]
                reduce_length = torch.tensor(x.shape[0], dtype=torch.int64, device=current_platform.device_type)
                dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
                global_max_length = reduce_length.item()
                pad_size = global_max_length - x.shape[0]

                x = torch.nn.functional.pad(
                    x, (0, 0, 0, pad_size)
                )

            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            x = all_gather_two_stage(x, idx=0, dim=0)
            global_pertoken_scale = all_gather_two_stage(pertoken_scale, idx=1, dim=0)

        x = {'x_int8': x,
             'pertoken_scale': global_pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        if not is_prefill:
            if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                x = reduce_scatter_round_pipeline(x, idx=0, node_rank=self.node_rank)
            elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                x = reduce_scatter_pipeline(x, idx=0, which_half=self.which_half)
            else:
                x = reduce_scatter_two_stage(x, idx=0)
        else:
            x = reduce_scatter_two_stage(x, idx=0)

        if is_prefill and pad_size > 0:
            x = x[:local_length, :]
        return x, residual


class PanguUltraMoEDecoderLayer(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        num_dense_layers = getattr(config, "num_dense_layers", 3)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.self_attn = DeepseekMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.attention_qk_dim,
            qk_rope_head_dim=config.attention_qk_rope_dim,
            v_head_dim=config.attention_v_dim,
            q_lora_rank=config.attention_q_lora_dim if hasattr(config, "attention_q_lora_dim") else None,
            kv_lora_rank=config.attention_kv_lora_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=True,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (config.num_routed_experts is not None
            and layer_idx >= num_dense_layers
            and layer_idx % moe_layer_freq == 0):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            if model_extra_config.parall_config.dense_mlp_tp_size == 1:
                dense_tp_parallel = "no_tp"
            elif model_extra_config.parall_config.dense_mlp_tp_size <= 8:
                dense_tp_parallel = "local"
            else:
                dense_tp_parallel = "global"

            self.mlp = ParallelPanguUltraMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                tp_parallel=dense_tp_parallel,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        
        if getattr(config, 'sandwich_norm', False):
            self.sandwich_norm = True
            self.pre_mlp_layernorm = RMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
            self.post_mlp_layernorm = RMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
        else:
            self.sandwich_norm = False

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None,
            kv_prefetch: torch.Tensor = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=True)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )

        is_prefill = attn_metadata is None or attn_metadata.prefill is not None

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(
                hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(
                hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Perform full hidden splitting to avoid OOM
        if (model_extra_config.operator_opt_config.prefill_moe_all_to_all or model_extra_config.parall_config.dp_size > 1) and is_prefill:
            reduce_length = torch.tensor(hidden_states.shape[0], dtype=torch.int64, device=current_platform.device_type)
            local_length = hidden_states.shape[0]
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
                if self.is_moe == True:
                    hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata, layer_id)
                else:
                    hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata)
                hidden_state_out.append(hidden_states)
                residual_out.append(residual)
            hidden_states = torch.cat(hidden_state_out)[:local_length]
            residual = torch.cat(residual_out)[:local_length]
        else:
            if self.is_moe == True:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id, kv_prefetch)
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual


class PanguUltraMoEModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                parallel_lmhead=(model_extra_config.parall_config.dp_size > 1),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PanguUltraMoEDecoderLayer(
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

        self.is_init = False
        self.num_dense_layers = config.num_dense_layers
        self.num_hidden_layers = config.num_hidden_layers

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if model_extra_config.operator_opt_config.use_prefetch and not self.is_init:
            prefetch_start_layer = self.start_layer if self.start_layer > self.num_dense_layers else self.num_dense_layers
            prefetch_end_layer = self.end_layer if self.end_layer < self.num_hidden_layers - 1 else self.num_hidden_layers - 1
            for layer_id in range(prefetch_start_layer, prefetch_end_layer):
                self.layers[layer_id].mlp.attn_prefetch = self.layers[layer_id + 1].self_attn
            self.is_init = True

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - self.num_dense_layers
            if model_extra_config.operator_opt_config.use_prefetch and i < self.end_layer - 1 and kv_caches is not None:
                kv_prefetch = kv_caches[i + 1 - self.start_layer]
            else:
                kv_prefetch = None
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata, residual, layer_id,
                                            kv_prefetch)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        
        return hidden_states


@support_torch_compile
class PanguUltraMoEForCausalLM(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = PanguUltraMoEModel(vllm_config=vllm_config, prefix="model")
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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: AttentionMetadata = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        
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
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("qkv_a_proj", "q_a_proj", 0),
                ("qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ]
        else:
            stacked_params_mapping = [
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'PanguUltraMoEForCausalLM' and self.config.num_mtp_layers > 0:
                mtp_prefix = [f"model.layers.{self.config.num_hidden_layers + layer_idx}" for layer_idx in range(self.config.num_mtp_layers)]
                if name.startswith(tuple(mtp_prefix)):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
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
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
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
        attn_metadata = kwargs.get('attn_metadata', None)
        
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        if attn_metadata is None:
            return True

        if attn_metadata.prefill:
            return True

        return False
