#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen3_vl_moe.py
# Copyright 2023 The vLLM team.
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

import typing
from typing import Callable

import torch
import torch.nn as nn
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_ep_group
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (SupportsLoRA, SupportsMultiModal, SupportsPP, MixtureOfExperts)
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeDecoderLayer,
                                                  Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel, Qwen3MoeLLMForCausalLM, Qwen3VLDummyInputsBuilder,
    Qwen3VLMoeForConditionalGeneration, Qwen3VLMultiModalProcessor, Qwen3VLMoeProcessingInfo)
from vllm_ascend.models.qwen3_vl import AscendQwen3_VisionTransformer
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, make_layers, PPMissingLayer, maybe_prefix, WeightsMapper)
from vllm.multimodal import MULTIMODAL_REGISTRY


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0
    })
class AscendQwen3MoeLLMModel(Qwen3MoeLLMModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        parallel_config = vllm_config.parallel_config
        enable_eplb = parallel_config.enable_eplb
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix,
                                                enable_eplb=enable_eplb),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.
                deepstack_visual_indexes), (
                    "start_layer should be greater than or equal to "
                    "len(deepstack_visual_indexes)")

    def load_fused_expert_weights(self, name: str, params_dict: dict,
                                  loaded_weight: torch.Tensor, shard_id: str,
                                  num_experts: int):
        param = params_dict[name]
        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()
        ep_size = ep_group.size()
        local_experts_num = (num_experts // ep_size)
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        for expert_id in range(ep_rank * local_experts_num, (ep_rank + 1) * local_experts_num):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(param,
                                    curr_expert_weight,
                                    name,
                                    shard_id,
                                    expert_id,
                                    return_success=True)
            if not success:
                return False
        return True


class AscendQwen3MoeLLMForCausalLM(Qwen3MoeLLMForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        SupportsPP.__init__(self)
        SupportsLoRA.__init__(self)
        MixtureOfExperts.__init__(self)
        self.config = vllm_config.model_config.hf_config.text_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendQwen3MoeLLMModel(vllm_config=vllm_config,
                                            prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.expert_weights: list[torch.Tensor] = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
                                        info=Qwen3VLMoeProcessingInfo,
                                        dummy_inputs=Qwen3VLDummyInputsBuilder)
class AscendQwen3VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        SupportsMultiModal.__init__(self)
        SupportsLoRA.__init__(self)
        SupportsPP.__init__(self)
        config: Qwen3VLMoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        self.visual = AscendQwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel,
        )

        self.language_model = AscendQwen3MoeLLMForCausalLM(vllm_config=vllm_config,
                                                           prefix=maybe_prefix(prefix, "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.use_deepstack = hasattr(config.vision_config,
                                     'deepstack_visual_indexes')
        self.deepstack_num_level = len(
            config.vision_config.deepstack_visual_indexes
        ) if self.use_deepstack else 0
        # register buffer for deepstack
        self.deepstack_input_embeds = [
            torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size)
            for _ in range(self.deepstack_num_level)
        ] if self.use_deepstack else None