# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import QuantizationConfig, VllmConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

if os.getenv("ASCEND_PLATFORM", "A3")=="A2":
    from .pangu_ultra_moe_a2 import PanguUltraMoEDecoderLayer
else:
    from .pangu_ultra_moe import PanguUltraMoEDecoderLayer

from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.layernorm import RMSNorm  # zxp: not use
from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig,
                                        weight_name: str) -> Optional[int]:
    if (hasattr(config, "num_mtp_layers")
            and config.num_mtp_layers > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None

class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        ignore_share_weight: bool = True,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = None if ignore_share_weight else \
            ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

@support_torch_compile
class PanguUltraMoEMultiTokenPredictorLayer(PanguUltraMoEDecoderLayer):
    def __init__(self, *,
                 vllm_config,
                 prefix: str,
    ):
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

        super().__init__(self.config, prefix,
                         cache_config=self.cache_config,
                         quant_config=self.quant_config,
                        )

        self.ignore_share_weight = True # TODO get from config
        self.embed_tokens = None if self.ignore_share_weight else \
            VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=prefix,
            )
        self.shared_head = SharedHead(self.config, self.quant_config, self.ignore_share_weight)
        self.enorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.hnorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size, bias=False)
        self.logits_processor = LogitsProcessor(self.config.vocab_size, logits_as_input=True)
        self.layer_idx = int(prefix.split('.')[-1])

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        tok_embeds = self.enorm(self.get_input_embeddings(input_ids))
        if len(tok_embeds.shape) > 2:
            tok_embeds = tok_embeds.view(-1, self.config.hidden_size)

        tp_size = get_tensor_model_parallel_world_size()  # cloud: get_tp_group().world_size
        rank_in_group = get_tensor_model_parallel_rank()

        if tp_size > 1:
            token_num = previous_hidden_states.shape[0]
            start_range = rank_in_group * (token_num // tp_size)
            end_range = (1 + rank_in_group) * (token_num // tp_size)
            previous_hidden_states = previous_hidden_states[start_range: end_range, :]

        previous = self.hnorm(previous_hidden_states)
        cat_hidden_states = torch.cat([tok_embeds, previous], dim=-1)
        hidden_states = self.eh_proj.forward(cat_hidden_states)

        encoded_states, residual = PanguUltraMoEDecoderLayer.forward(
            self,
            positions=positions,
            kv_cache=kv_caches[self.layer_idx] if kv_caches is not None else None,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
        )

        hidden_states, _ = self.shared_head.norm(encoded_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        return logits, hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if model_extra_config.parall_config.dp_size <= 1 and selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)
        # Get the logits for the next tokens.
        logits = self.shared_head.head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(self.shared_head["head"], hidden_states, sampling_metadata)
        return logits

    def should_use_eager_mode(self, *args, **kwargs):
        if len(kwargs) == 0:
           return True

        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        if attn_metadata.prefill:
            return True

        return False

class PanguUltraMoEMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.mtp_start_layer_idx = self.config.num_hidden_layers
        self.num_mtp_layers = self.config.num_mtp_layers

        self.ignore_share_weight = True # TODO get from config
        if self.num_mtp_layers > 1:
            raise ValueError(f"Only support 1 mtp layers at most, while get num_nextn_predict_layers = {self.num_mtp_layers}.")
        mtp_class_list = [PanguUltraMoEMultiTokenPredictorLayer]
        self.layers = nn.ModuleDict({
            str(i + self.mtp_start_layer_idx):
            mtp_class_list[i](vllm_config=vllm_config,
                              prefix=f"{prefix}.layers.{i + self.mtp_start_layer_idx}")
            for i in range(min(self.num_mtp_layers, vllm_config.speculative_config.num_speculative_tokens))
        })
        self.logits_processor = LogitsProcessor(self.config.vocab_size, logits_as_input=True)
        self.greedy_sampler = Sampler()
    
    def set_share_weight(self, target_model):
        if self.ignore_share_weight:
            for _, layer in self.layers.items():
                layer.embed_tokens = target_model.model.embed_tokens
                layer.shared_head.head = target_model.lm_head

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            mtp_layer_idx = 0,
    ) -> torch.Tensor:
        return self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices,
        )


class PanguUltraMoEMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.model = PanguUltraMoEMultiTokenPredictor(vllm_config=vllm_config, prefix=f"model")
    
    def set_share_weight(self, target_model):
        self.model.set_share_weight(target_model)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            mtp_layer_idx = 0,
            **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices,
            mtp_layer_idx=mtp_layer_idx,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # 字段说明: (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # 字段说明: (param_name, weight_name, expert_id, shard_id)
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
            if self.model.ignore_share_weight and any(
                    substring in name for substring in ["embed_tokens.weight", "shared_head.head"]):
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
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
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
