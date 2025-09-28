#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn
from torch.nn import Parameter
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (divide, get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group, get_world_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    extract_layer_index, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p

_ROUTER_SCALE = None


def use_h2p():
    # only use H2P when dp_size > 1.
    if get_dp_group().world_size > 1:
        return True
    return False


# This class is adapted from vllm.model_executor.layers.linear.MergedColumnParallelLinear.
# It is used to customize parallelism of certain linear(e.g., shared experts with all-rank tp).
class CustomMergedColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        # Divide the weight matrix along the last dimension.
        output_size = sum(output_sizes)
        self.output_sizes = output_sizes
        self.tp_size = get_tp_group().world_size
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        assert loaded_shard_id < len(self.output_sizes)

        tp_rank = get_tp_group().rank_in_group
        tp_size = get_tp_group().world_size
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


# This class is adapted from vllm.model_executor.layers.linear.RowParallelLinear.
# It is used to customize parallelism of certain linear(e.g., shared experts with all-rank tp)
# and detach communication to enable customized communication algorithms(e.g., H2P).
class CustomRowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        group=None,
    ):
        # Divide the weight matrix along the first dimension.
        self.group = group if group is not None else get_tp_group()
        self.tp_rank = self.group.rank_in_group
        self.tp_size = self.group.world_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = self.group.rank_in_group
        input_dim = getattr(param, "input_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        is_sharded_weight = is_sharded_weight

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        input_parallel = input_

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output = self.quant_method.apply(self, input_parallel, bias=bias_)

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class PanguProMoEMLP(nn.Module):

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
        if not use_h2p():
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.down_proj",
            )
        else:
            self.gate_up_proj = CustomMergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = CustomRowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.down_proj",
            )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def topk_wrapper(num_voted_experts):

    def pangu_group8_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool = False,
        num_expert_group: int = 0,
        topk_group: int = 0,
        global_num_experts: int = 0,
    ):
        scores = F.softmax(gating_output, dim=1)
        num_tokens = scores.shape[0]
        router_scale = _ROUTER_SCALE.squeeze(  # type: ignore
        )
        # TODO: support disable expert parallel
        ep_size = get_ep_group().world_size
        local_num_experts = global_num_experts // ep_size
        local_num_group = topk // ep_size
        experts_per_group = global_num_experts // topk
        local_group_start = get_ep_group().rank_in_group * local_num_experts
        local_group_end = (get_ep_group().rank_in_group +
                           1) * local_num_experts
        scores = F.softmax(gating_output, dim=1)
        scores = scores[..., local_group_start:local_group_end]

        router_weights = router_scale[local_group_start:local_group_end]

        if num_voted_experts == 8:
            # use original topk
            topk_weights, topk_ids = torch.max(scores.view(
                scores.shape[0], local_num_group, -1),
                                               dim=-1)
            bias = torch.arange(0,
                                local_num_experts,
                                experts_per_group,
                                device=scores.device,
                                dtype=torch.int32).unsqueeze(0)
            topk_ids = topk_ids.to(torch.int32) + bias

        else:
            group_expert_indices = torch.arange(experts_per_group,
                                                dtype=torch.int32,
                                                device=scores.device).view(
                                                    1, 1, -1)
            group_expert_offset = (torch.arange(
                local_num_group, dtype=torch.int32, device=scores.device) *
                                   experts_per_group).unsqueeze(0)
            expert_index_range = torch.arange(experts_per_group,
                                              dtype=torch.int32,
                                              device=scores.device)

            scores_grouped = scores.view(num_tokens, local_num_group,
                                         experts_per_group)
            best_expert_idx = torch.argmax(scores_grouped,
                                           dim=2)  # (num_tokens, num_groups)
            vote_mask = (best_expert_idx.unsqueeze(-1).to(
                torch.int32) == group_expert_indices)

            expert_vote_freq = vote_mask.sum(dim=0)

            sorted_indices = torch.argsort(expert_vote_freq,
                                           dim=1,
                                           descending=True).to(torch.int32)
            topk_experts = sorted_indices[:, :num_voted_experts]
            keep_mask = ((
                topk_experts.unsqueeze(-1) == expert_index_range).any(
                    dim=1)).unsqueeze(0)

            masked_scores = torch.where(keep_mask, scores_grouped, 0)

            topk_weights, best_pos_in_group = masked_scores.max(dim=2)
            best_pos_in_group = best_pos_in_group.to(torch.int32)
            topk_ids = (best_pos_in_group + group_expert_offset).to(
                torch.int32)

        flatten_topk_ids = topk_ids.view(-1)
        router_weights = router_weights.index_select(0, flatten_topk_ids).view(
            topk_ids.shape)
        topk_weights *= router_weights

        return topk_weights, topk_ids

    return pangu_group8_topk


class PanguProMoESparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_scale = torch.nn.Parameter(
            torch.ones((1, self.num_experts)))

        # on 300I Duo platform, we find that num_voted_experts set to 5 achieves
        # good performance without sacrifice too much accuracy. for other platform,
        # this is set to 8 to use original pangu grouped topk.
        num_voted_experts = 5 if is_310p() else 8

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            custom_routing_function=topk_wrapper(num_voted_experts),
            prefix=f"{prefix}.experts",
        )
        self.use_ep = self.experts.use_ep

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = PanguProMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None  # type: ignore

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        global _ROUTER_SCALE
        _ROUTER_SCALE = self.router_scale

        # TODO(angazenn): Does not support MC2 currently
        get_forward_context().moe_comm_method_name = "allgathercommimpl"

        if not use_h2p():
            final_hidden_states = self.experts.forward_impl(
                hidden_states=hidden_states, router_logits=router_logits)
        else:
            # TODO: when using h2p, we have to skip communication in vLLM
            # native FusedMoE. here we need to design a better FusedMoE
            # (maybe using AscendFusedMoE) to enable these different
            # communication schema.
            final_hidden_states = self.experts.quant_method.apply(
                layer=self.experts,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.experts.top_k,
                renormalize=False,
                use_grouped_topk=False,
                global_num_experts=self.experts.global_num_experts,
                expert_map=self.experts.expert_map,
                custom_routing_function=self.experts.custom_routing_function,
                apply_router_weight_on_input=self.experts.
                apply_router_weight_on_input)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if not use_h2p():
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class PanguProMoEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        if use_h2p():
            self.o_proj = CustomRowParallelLinear(self.total_num_heads *
                                                  self.head_dim,
                                                  hidden_size,
                                                  bias=True,
                                                  quant_config=quant_config,
                                                  prefix=f"{prefix}.o_proj",
                                                  group=get_tp_group())
        else:
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.torchair_graph_enabled:
            forward_kwargs = {'trace_flag': False}
            output_shape = q.shape
            attn_output = torch.empty(output_shape,
                                      dtype=q.dtype,
                                      device=q.device)
            forward_kwargs['output'] = attn_output
            attn_output = self.attn.impl.forward(self.attn, q, k, v, kv_cache,
                                                 attn_metadata,
                                                 **forward_kwargs)
        else:
            attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


class PanguProMoEDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        self.self_attn = PanguProMoEAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (config.num_experts > 0):
            self.mlp = PanguProMoESparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = PanguProMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        h2p_unpad_idx: Optional[torch.Tensor] = None,
        h2p_pad_idx: Optional[torch.Tensor] = None,
        is_start_layer: Optional[bool] = False,
    ) -> torch.Tensor:
        need_h2p_pad = h2p_unpad_idx is not None and h2p_pad_idx is not None \
            and h2p_unpad_idx.shape[0] < h2p_pad_idx.shape[0]
        tp_size = get_tp_group().world_size

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        if use_h2p():
            if is_start_layer:
                if need_h2p_pad:
                    residual = residual.index_select(dim=0, index=h2p_pad_idx)
                residual = torch.tensor_split(
                    residual, tp_size)[get_tp_group().rank_in_group]
            else:
                if tp_size > 1:
                    hidden_states = get_tp_group().all_gather(hidden_states, 0)
                if need_h2p_pad:
                    hidden_states = hidden_states.index_select(
                        dim=0, index=h2p_unpad_idx)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if use_h2p():
            if need_h2p_pad:
                hidden_states = hidden_states.index_select(dim=0,
                                                           index=h2p_pad_idx)
            if tp_size > 1:
                hidden_states = dist._functional_collectives.reduce_scatter_tensor(
                    hidden_states,
                    "sum",
                    scatter_dim=0,
                    group=get_tp_group().device_group)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        if use_h2p():
            all_rank_group = get_world_group().device_group
            output_size = (hidden_states.shape[0] *
                           get_world_group().world_size,
                           hidden_states.shape[1])
            # Allocate output tensor.
            output_tensor = torch.empty(output_size,
                                        dtype=hidden_states.dtype,
                                        device=hidden_states.device)
            # All-gather.
            dist.all_gather_into_tensor(output_tensor,
                                        hidden_states,
                                        group=all_rank_group)
            hidden_states = output_tensor

        hidden_states = self.mlp(hidden_states, attn_metadata=attn_metadata)

        if use_h2p():
            hidden_states = dist._functional_collectives.reduce_scatter_tensor(
                hidden_states,
                "sum",
                scatter_dim=0,
                group=get_world_group().device_group)

        return hidden_states, residual


@support_torch_compile
class PanguProMoEModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PanguProMoEDecoderLayer(config=config,
                                                   cache_config=cache_config,
                                                   quant_config=quant_config,
                                                   prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if use_h2p():
            # calculate necessary padding/unpadding idx before model forward.

            # the attn_metadata will be passed directly when use torchair.
            # if attn_meatadata is not passed, we try to get it from forward_context.
            if attn_metadata is None:
                attn_metadata = get_forward_context().attn_metadata

            max_tokens_across_dp = get_forward_context().max_tokens_across_dp

            tp_size = get_tp_group().world_size
            # reduce scatter will split the input tensor into equal sizes and then scatter them on all ranks.
            # we need pad it before if the shape can't be divided by group size.
            # for h2p, we need pad it so that it can be divided by tp_size.
            h2p_padded_len = (
                tp_size - (max_tokens_across_dp % tp_size)
            ) % tp_size + max_tokens_across_dp - hidden_states.shape[0]
            h2p_unpad_idx = torch.arange(hidden_states.shape[0],
                                         device=hidden_states.device,
                                         dtype=torch.int32)
            h2p_pad_idx = torch.cat([
                h2p_unpad_idx,
                torch.zeros(h2p_padded_len,
                            dtype=torch.int32,
                            device=hidden_states.device)
            ])
        else:
            h2p_unpad_idx = None
            h2p_pad_idx = None

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata, h2p_unpad_idx, h2p_pad_idx,
                i == self.start_layer)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        if use_h2p():
            if get_tp_group().world_size > 1:
                hidden_states = get_tp_group().all_gather(hidden_states, 0)
            if h2p_unpad_idx.shape[0] < h2p_pad_idx.shape[0]:
                hidden_states = hidden_states.index_select(dim=0,
                                                           index=h2p_unpad_idx)
        return hidden_states


class PanguProMoEForCausalLM(nn.Module, SupportsPP):

    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = PanguProMoEModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())  # from model
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            # =======================================================
            # BF: add this to load with less layers
            if 'layers' in name:
                layer_idx = int(name.split('layers.')[-1].split('.')[0])
                if layer_idx >= self.model.end_layer:
                    continue

            if "rotary_emb.inv_freq" in name:
                continue

            if "module" in name:
                continue

            if name.endswith('kv_cache_offset'):
                continue

            if name.endswith("k_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace(
                    "k_proj.kv_cache_scale", "attn.key_antiquant_scale")
                if remapped_kv_scale_name not in params_dict:
                    logger.warning_once(
                        "Found kv scale in the checkpoint "
                        f"(e.g. {name}), but not found the expected "
                        f"name in the model "
                        f"(e.g. {remapped_kv_scale_name}). "
                        "kv-scale is not loaded.")
                    continue
                else:
                    name = remapped_kv_scale_name
                    param = params_dict[name]
                    loaded_weight = torch.tensor_split(loaded_weight,
                                                       tp_size,
                                                       dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

            if name.endswith("v_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace(
                    "v_proj.kv_cache_scale", "attn.value_antiquant_scale")
                if remapped_kv_scale_name not in params_dict:
                    logger.warning_once(
                        "Found kv scale in the checkpoint "
                        f"(e.g. {name}), but not found the expected "
                        f"name in the model "
                        f"(e.g. {remapped_kv_scale_name}). "
                        "kv-scale is not loaded.")
                    continue
                else:
                    name = remapped_kv_scale_name
                    param = params_dict[name]
                    loaded_weight = torch.tensor_split(loaded_weight,
                                                       tp_size,
                                                       dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

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
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue

                # Skip layers on other devices.
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
                    # breakpoint()
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    # breakpoint()
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
            if is_310p() and "head" in name:
                # on 300I Duo platform, ACL_FORMAT_FRACTAL_NZ is much more preferred than
                # ACL_FORMAT_FRACTAL_ND by matmul operation. Since lmhead is also implemented
                # by linear, we manually cast the format here.
                param.data = torch_npu.npu_format_cast(param.data,
                                                       ACL_FORMAT_FRACTAL_NZ)
        return loaded_params
