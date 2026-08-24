# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from itertools import islice
from types import SimpleNamespace
from typing import Any

import torch
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekForCausalLM,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
    _get_llama_4_scaling,
)
from vllm.model_executor.models.mistral_large_3 import MistralLarge3ForCausalLM
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors


def _get_skip_topk(config: DeepseekV2Config | DeepseekV3Config, layer_id: int) -> bool:
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None and 0 <= layer_id < len(pattern):
        return pattern[layer_id] == "S"

    frequency = getattr(config, "index_topk_freq", 1)
    offset = getattr(config, "index_skip_topk_offset", 2)
    return max(layer_id - offset + 1, 0) % frequency != 0


def should_skip_indexer_init(
    config: DeepseekV2Config | DeepseekV3Config | SimpleNamespace,
    prefix: str,
    skip_topk: bool,
) -> bool:
    """Separate runtime top-k reuse from the checkpoint's module layout."""
    if not skip_topk:
        return False

    layer_id = extract_layer_index(prefix)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False

    indexer_types = getattr(config, "indexer_types", None)
    indexer_type = indexer_types[layer_id] if indexer_types is not None and layer_id < len(indexer_types) else None
    return isinstance(indexer_type, str) and indexer_type.lower() == "shared"


def get_indexer_init_pattern(
    config: DeepseekV2Config | DeepseekV3Config | SimpleNamespace,
) -> tuple[list[bool], list[str]]:
    """Return runtime skip flags and the temporary upstream init pattern.

    Upstream currently uses one flag for both top-k reuse and Indexer module
    construction. GLM-5.1 needs per-layer Indexers even when top-k is reused,
    while GLM-5.2 omits modules explicitly marked as shared. During model
    construction we therefore expose only the checkpoint-layout decision to
    upstream, then restore the runtime skip flags on the created wrappers.
    """
    num_hidden_layers = int(config.num_hidden_layers)
    runtime_skip = [_get_skip_topk(config, layer_id) for layer_id in range(num_hidden_layers)]
    init_pattern = [
        "S"
        if should_skip_indexer_init(
            config,
            f"model.layers.{layer_id}.self_attn",
            runtime_skip[layer_id],
        )
        else "F"
        for layer_id in range(num_hidden_layers)
    ]
    return runtime_skip, init_pattern


class AscendDeepseekV2Model(DeepseekV2Model):
    """DeepSeek model with Ascend's runtime sequence-sharded layout contract."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            raise TypeError("vllm_config is required")

        config = vllm_config.model_config.hf_config
        runtime_skip: list[bool] | None = None
        original_pattern = getattr(config, "index_topk_pattern", None)
        had_pattern = hasattr(config, "index_topk_pattern")
        if hasattr(config, "index_topk"):
            runtime_skip, init_pattern = get_indexer_init_pattern(config)
            config.index_topk_pattern = init_pattern

        try:
            super().__init__(*args, **kwargs)
        finally:
            if runtime_skip is not None:
                if had_pattern:
                    config.index_topk_pattern = original_pattern
                else:
                    delattr(config, "index_topk_pattern")

        if runtime_skip is not None:
            for layer_id in range(self.start_layer, self.end_layer):
                layer = self.layers[layer_id]
                mla_wrapper = getattr(
                    getattr(layer, "self_attn", None),
                    "mla_attn",
                    None,
                )
                if mla_wrapper is not None:
                    mla_wrapper.skip_topk = runtime_skip[layer_id]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided to AscendDeepseekV2Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state,
                        0,
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if hidden_states.shape[0] != positions.shape[0]:
            combined_states = torch.cat([hidden_states, residual], dim=-1)
            combined_states = tensor_model_parallel_all_gather(combined_states, 0)
            combined_states = combined_states[: positions.shape[0]]
            hidden_states, residual = combined_states.split(
                [self.hidden_size, self.hidden_size],
                dim=-1,
            )
            residual = residual.contiguous()

        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = AscendDeepseekV2Model


class AscendDeepseekForCausalLM(DeepseekForCausalLM):
    model_cls = AscendDeepseekV2Model


class AscendDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    model_cls = AscendDeepseekV2Model


class AscendMistralLarge3ForCausalLM(MistralLarge3ForCausalLM):
    model_cls = AscendDeepseekV2Model
