#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""Xlite integration module for vLLM-Ascend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size, get_world_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import AttnMeta, AttnMHA, Model, ModelConfig, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.xlite.utils import _get_nested_attr

XliteInitResult: TypeAlias = tuple[Model, torch.Tensor, int, torch.dtype]
XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]


class XliteModel(ABC):
    """Base adapter for converting vLLM models into xlite runtime models.

    Subclasses are responsible for mapping architecture-specific configuration and weights into the `xlite._C.Model`
    interface.

    Attributes:
        runnable (nn.Module): The original runnable model used by vLLM. Used as the source of truth for weight
            extraction for xlite model construction.
        vllm_config (VllmConfig): The configuration object provided by vLLM. Used to build xlite configuration at
            runtime.
        xlite_config (ModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (Model): Native xlite model container populated by subclasses.
    """

    def __init__(self, runnable: nn.Module, vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            runnable (nn.Module): The original runnable model used by vLLM.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.runnable = runnable
        self.vllm_config = vllm_config

        self.xlite_config = ModelConfig()
        self.xlite_model = Model()

    def initialize(self) -> XliteInitResult:
        """Initialize an xlite model and precomputed RoPE cache.

        Returns:
            XliteInitResult: A tuple of `(xlite_model, freq_cis, hidden_size, dtype)` required by `XliteWrapper`.
        """
        self._build_model_config()
        self._build_model()

        rank = torch.distributed.get_rank()
        self.xlite_model.init(self.xlite_config, rank)

        freq_cis = self._precompute_freqs_cis()
        return (self.xlite_model, freq_cis, self.xlite_config.hidden_size, self.vllm_config.model_config.dtype)

    @abstractmethod
    def _build_model_config(self) -> None:
        """Build architecture-specific xlite model configuration.

        This method extracts necessary configuration attributes from the vLLM config (e.g., HuggingFace metadata) and
        populates an xlite :class:`ModelConfig` object.

        Returns:
            None: `self` attribute :attr:`xlite_config` is updated in-place.
        """

    @abstractmethod
    def _build_model(self) -> None:
        """Build architecture-specific xlite model weights.

        This method traverses the runnable model's parameters and maps them into the xlite :class:`Model` interface
        according to the architecture's specific structure.

        Returns:
            None: `self` attribute :attr:`xlite_model` is updated in-place.

        Notes:
            :meth:`_build_model_config` should be called prior to this method to ensure the xlite configuration is
            populated before weight mapping.
        """

    def _get_layers_and_model_prefix(self) -> tuple[Sequence[nn.Module], str]:
        """Extract transformer layers and parameter prefix from runnable.

        Returns:
            tuple[Sequence[nn.Module], str]: A pair of `(layers, model_prefix)` for model traversal.
        """
        if hasattr(self.runnable, "language_model"):
            layers = cast(
                Sequence[nn.Module], _get_nested_attr(self.runnable.language_model, "model", "layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], _get_nested_attr(self.runnable, "model", "layers", default=[]))
            prefix = ""
        return layers, prefix

    @abstractmethod
    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precomputes frequency-based complex exponential values for rotary positional embeddings (RoPE).

        This method generates the RoPE frequency cache (cosine and sine values) required by the xlite attention
        implementation. The cache should be precomputed on the NPU device to avoid unnecessary host-device transfers
        during inference.

        Returns:
            torch.Tensor: The precomputed RoPE frequency cache tensor ready for use in xlite attention computations.

        Notes:
            :meth:`_build_model_config` should be called prior to this method.
        """

    @property
    def hf_text_config(self) -> PretrainedConfig:
        """Convenience property to access HuggingFace text configuration from vLLM config.

        Returns:
            PretrainedConfig: The HuggingFace text configuration object extracted from vLLM config.
        """
        hf_config = self.vllm_config.model_config.hf_text_config
        return cast(PretrainedConfig, getattr(hf_config, "text_config", hf_config))

    @property
    def hf_vision_config(self) -> PretrainedConfig | None:
        """Convenience property to access HuggingFace vision configuration from vLLM config, if exists.

        Returns:
            PretrainedConfig | None: The HuggingFace vision configuration object extracted from vLLM config, or None if
                not present.
        """
        return getattr(self.vllm_config.model_config.hf_config, "vision_config", None)


class LlamaXliteModel(XliteModel):
    """xlite adapter for Llama-like dense transformer architectures."""

    def _build_model_config(self) -> None:
        vllm_config = self.vllm_config
        hf_config = self.hf_text_config
        xlite_config = self.xlite_config

        xlite_config.vocab_size = hf_config.vocab_size
        xlite_config.hidden_size = hf_config.hidden_size
        xlite_config.n_layers = hf_config.num_hidden_layers
        xlite_config.n_heads = hf_config.num_attention_heads
        xlite_config.n_kv_heads = hf_config.num_key_value_heads
        if hasattr(hf_config, "head_dim"):
            xlite_config.head_dim = hf_config.head_dim
        else:
            xlite_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        xlite_config.rope_head_dim = xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.def_tp_size = get_tensor_model_parallel_world_size()
        xlite_config.def_dp_size = 1
        xlite_config.moe_ep_size = 1
        xlite_config.moe_tp_size = 1

        xlite_config.attn_type = AttnMHA
        xlite_config.weight_nz = envs_ascend.VLLM_ASCEND_ENABLE_NZ == 2
        scheduler_config = vllm_config.scheduler_config
        max_batch_size = scheduler_config.max_num_seqs
        max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.max_m = (
            scheduler_config.max_num_batched_tokens
            if get_ascend_config().xlite_graph_config.full_mode
            else scheduler_config.max_num_seqs
        )
        xlite_config.max_batch_size = max_batch_size
        xlite_config.max_seq_len = max_seq_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        if hasattr(xlite_config, "deepstack_num_level"):
            xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        if hasattr(xlite_config, "mrope_section"):
            xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        if hasattr(xlite_config, "mrope_interleaved"):
            xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)

    def _build_model(self) -> None:
        hf_config = self.hf_text_config
        xlite_config = self.xlite_config
        xlite_model = self.xlite_model

        params_dict = dict(self.runnable.named_parameters())
        layers, model_prefix = self._get_layers_and_model_prefix()

        def _require_param(param_name: str) -> torch.Tensor:
            param = params_dict.get(param_name)
            if not isinstance(param, torch.Tensor):
                raise ValueError(f"Required parameter not found in the runnable: {param_name}")
            return param

        xlite_model.embed = _require_param(f"{model_prefix}model.embed_tokens.weight")
        xlite_model.norm = _require_param(f"{model_prefix}model.norm.weight")
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = _require_param(f"{model_prefix}lm_head.weight")

        xlite_model.attn_norm = [
            weight for layer in layers if (weight := _get_nested_attr(layer, "input_layernorm", "weight")) is not None
        ]
        xlite_model.attn_out = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "self_attn", "o_proj", "weight")) is not None
        ]

        xlite_model.mha_qkv = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "self_attn", "qkv_proj", "weight")) is not None
        ]
        mha_qkv_bias = [
            bias for layer in layers if (bias := _get_nested_attr(layer, "self_attn", "qkv_proj", "bias")) is not None
        ]
        q_norm = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "self_attn", "q_norm", "weight")) is not None
        ]
        k_norm = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "self_attn", "k_norm", "weight")) is not None
        ]

        if len(mha_qkv_bias) != xlite_config.n_layers:
            xlite_config.qkv_bias = False
        else:
            xlite_config.qkv_bias = True
            xlite_model.mha_qkv_bias = mha_qkv_bias

        if len(q_norm) != xlite_config.n_layers or len(k_norm) != xlite_config.n_layers:
            xlite_config.qk_norm = False
        else:
            xlite_config.qk_norm = True
            xlite_model.mha_q_norm = q_norm
            xlite_model.mha_k_norm = k_norm

        xlite_model.mlp_norm = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "post_attention_layernorm", "weight")) is not None
        ]
        xlite_model.mlp_up_gate = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "mlp", "gate_up_proj", "weight")) is not None
        ]
        xlite_model.mlp_down = [
            weight for layer in layers if (weight := _get_nested_attr(layer, "mlp", "down_proj", "weight")) is not None
        ]

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute rotary cosine/sine cache on NPU.

        Returns:
            torch.Tensor: Concatenated cosine/sine RoPE cache on NPU.

        Raises:
            ValueError: If rope dimensions, sequence length, or theta are invalid.
        """
        base = self.xlite_config.rope_theta
        rotary_dim = self.xlite_config.rope_head_dim
        max_position_embeddings = self.xlite_config.max_seq_len
        dtype = self.vllm_config.model_config.dtype

        if rotary_dim <= 0 or max_position_embeddings <= 0 or base <= 0:
            raise ValueError(
                f"Invalid RoPE configuration: head_dim={rotary_dim}, max_seq_len={max_position_embeddings}, "
                f"rope_theta={base}"
            )

        # Keep cache construction on CPU, then transfer once to NPU.
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        cos_cache = freqs.cos().to(dtype)
        sin_cache = freqs.sin().to(dtype)
        freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
        return freq_cis.to(device="npu")


class QwenMoeXliteModel(LlamaXliteModel):
    """xlite adapter for Qwen MoE architectures."""

    def _build_model_config(self) -> None:
        super()._build_model_config()

        vllm_config = self.vllm_config
        hf_config = self.hf_text_config
        xlite_config = self.xlite_config

        ep_group = get_ep_group()
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        xlite_config.moe_ep_size = ep_group.world_size if vllm_config.parallel_config.enable_expert_parallel else 1
        xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_group.world_size
        xlite_config.experts_weight_transpose = True
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSoftmax

    def _build_model(self) -> None:
        super()._build_model()

        layers, _ = self._get_layers_and_model_prefix()
        xlite_model = self.xlite_model
        xlite_model.gate = [
            weight for layer in layers if (weight := _get_nested_attr(layer, "mlp", "gate", "weight")) is not None
        ]
        xlite_model.re_up_gate = [
            weight
            for layer in layers
            if (w13_weight := _get_nested_attr(layer, "mlp", "experts", "w13_weight")) is not None
            for weight in w13_weight[: _get_nested_attr(layer, "mlp", "experts", "local_num_experts", default=0)]
        ]
        xlite_model.re_down = [
            weight
            for layer in layers
            if (w2_weight := _get_nested_attr(layer, "mlp", "experts", "w2_weight")) is not None
            for weight in w2_weight[: _get_nested_attr(layer, "mlp", "experts", "local_num_experts", default=0)]
        ]

        if xlite_model.re_up_gate:
            self.xlite_config.experts_weight_nz = (
                torch_npu.get_npu_format(xlite_model.re_up_gate[0]) == ACL_FORMAT_FRACTAL_NZ
            )


class Glm4MoeXliteModel(LlamaXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    def _build_model_config(self) -> None:
        super()._build_model_config()

        vllm_config = self.vllm_config
        hf_config = self.hf_text_config
        xlite_config = self.xlite_config

        ep_group = get_ep_group()
        if hasattr(hf_config, "partial_rotary_factor"):
            partial_rotary_factor = hf_config.partial_rotary_factor
        else:
            partial_rotary_factor = getattr(hf_config, "rope_parameters", {}).get("partial_rotary_factor", 1.0)
        xlite_config.rope_head_dim = int(hf_config.head_dim * partial_rotary_factor)
        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        xlite_config.moe_ep_size = ep_group.world_size if vllm_config.parallel_config.enable_expert_parallel else 1
        xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_group.world_size
        xlite_config.experts_weight_transpose = True
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor

    def _build_model(self) -> None:
        super()._build_model()

        layers, _ = self._get_layers_and_model_prefix()
        xlite_model = self.xlite_model
        xlite_model.gate = [
            weight for layer in layers if (weight := _get_nested_attr(layer, "mlp", "gate", "weight")) is not None
        ]
        xlite_model.gate_bias = [
            bias.to(torch.float32)  # NOTE: type conversion for numerical stability in xlite's implementation
            for layer in layers
            if (bias := _get_nested_attr(layer, "mlp", "gate", "e_score_correction_bias")) is not None
        ]
        xlite_model.se_up_gate = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "mlp", "shared_experts", "gate_up_proj", "weight")) is not None
        ]
        xlite_model.se_down = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "mlp", "shared_experts", "down_proj", "weight")) is not None
        ]
        xlite_model.re_up_gate = [
            w13_weight_i
            for layer in layers
            if (w13_weight := _get_nested_attr(layer, "mlp", "experts", "w13_weight")) is not None
            for w13_weight_i in w13_weight[: _get_nested_attr(layer, "mlp", "experts", "local_num_experts", default=0)]
        ]
        xlite_model.re_down = [
            w2_weight_i
            for layer in layers
            if (w2_weight := _get_nested_attr(layer, "mlp", "experts", "w2_weight")) is not None
            for w2_weight_i in w2_weight[: _get_nested_attr(layer, "mlp", "experts", "local_num_experts", default=0)]
        ]

        if xlite_model.re_up_gate:
            self.xlite_config.experts_weight_nz = (
                torch_npu.get_npu_format(xlite_model.re_up_gate[0]) == ACL_FORMAT_FRACTAL_NZ
            )


class MiniMaxM2XliteModel(LlamaXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    def _build_model_config(self) -> None:
        super()._build_model_config()

        vllm_config = self.vllm_config
        hf_config = self.hf_text_config
        xlite_config = self.xlite_config

        ep_group = get_ep_group()
        xlite_config.rope_head_dim = hf_config.rotary_dim
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_local_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        xlite_config.moe_ep_size = ep_group.world_size if vllm_config.parallel_config.enable_expert_parallel else 1
        xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_group.world_size
        xlite_config.experts_weight_transpose = True
        xlite_config.moe_intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.qk_norm_full = True
        xlite_config.scoring_func = ScoringFuncSigmoid

    def _build_model(self) -> None:
        super()._build_model()

        layers, _ = self._get_layers_and_model_prefix()
        xlite_model = self.xlite_model
        xlite_model.gate = [
            weight
            for layer in layers
            if (weight := _get_nested_attr(layer, "block_sparse_moe", "gate", "weight")) is not None
        ]
        xlite_model.gate_bias = [
            bias.to(torch.float32)  # NOTE: type conversion for numerical stability in xlite's implementation
            for layer in layers
            if (bias := _get_nested_attr(layer, "block_sparse_moe", "e_score_correction_bias")) is not None
        ]
        xlite_model.re_up_gate = [
            w13_weight_i
            for layer in layers
            if (w13_weight := _get_nested_attr(layer, "block_sparse_moe", "experts", "w13_weight")) is not None
            for w13_weight_i in w13_weight[
                : _get_nested_attr(layer, "block_sparse_moe", "experts", "local_num_experts", default=0)
            ]
        ]
        xlite_model.re_down = [
            w2_weight_i
            for layer in layers
            if (w2_weight := _get_nested_attr(layer, "block_sparse_moe", "experts", "w2_weight")) is not None
            for w2_weight_i in w2_weight[
                : _get_nested_attr(layer, "block_sparse_moe", "experts", "local_num_experts", default=0)
            ]
        ]

        if xlite_model.re_up_gate:
            self.xlite_config.experts_weight_nz = (
                torch_npu.get_npu_format(xlite_model.re_up_gate[0]) == ACL_FORMAT_FRACTAL_NZ
            )


def xlite_model_init(runnable: nn.Module, vllm_config: VllmConfig) -> XliteInitResult:
    """Construct and initialize an architecture-specific xlite model adapter.

    Args:
        runnable (nn.Module): The runnable model instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteInitResult: Initialized xlite model, RoPE cache, hidden size and dtype.
    """
    strategy_map: dict[str, type[XliteModel]] = {
        "LlamaForCausalLM": LlamaXliteModel,
        "Qwen2ForCausalLM": LlamaXliteModel,
        "Qwen3ForCausalLM": LlamaXliteModel,
        "Qwen3VLForConditionalGeneration": LlamaXliteModel,
        "Qwen3MoeForCausalLM": QwenMoeXliteModel,
        "Qwen3VLMoeForConditionalGeneration": QwenMoeXliteModel,
        "Glm4MoeForCausalLM": Glm4MoeXliteModel,
        "MiniMaxM2ForCausalLM": MiniMaxM2XliteModel,
    }

    architecture = vllm_config.model_config.architectures[0]
    strategy_class = strategy_map.get(architecture)
    if not strategy_class:
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(runnable, vllm_config).initialize()


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths."""

    def __init__(self, runnable: nn.Module, vllm_config: VllmConfig):
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            runnable (nn.Module): The runnable model implementation.
            vllm_config (VllmConfig): Runtime configuration for execution.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.runnable = runnable
        self.full_mode = get_ascend_config().xlite_graph_config.full_mode

        rank = torch.distributed.get_rank()
        local_rank = get_world_group().local_rank
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size
        self.xlite_rt = Runtime(local_rank, 0, rank, get_tensor_model_parallel_world_size(), self.data_parallel_size)

        (self.xlite_model, self.freq_cis, hidden_size, dtype) = xlite_model_init(runnable, vllm_config)

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if rank == 0:
            logger.info(f"xlite runtime pool size: {rt_pool_size} MB")
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_states = torch.empty(max_num_tokens, hidden_size, device=f"npu:{local_rank}", dtype=dtype)

    def __getattr__(self, key: str) -> Any:
        """Proxy unknown attributes to the wrapped runnable model.

        Args:
            key (str): The attribute name requested by the caller.

        Raises:
            AttributeError: If neither wrapper nor runnable has the attribute.

        Returns:
            Any: Attribute value resolved from the runnable.
        """
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of xlite wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        """Return the original runnable callable.

        Returns:
            Callable: Original model runnable.
        """
        # in case we need to access the original runnable.
        return self.runnable

    def register_kv_caches(self, kv_caches: Any) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (Any): Runtime KV cache handles or tensors.
        """
        self.kv_caches = kv_caches

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor): Token IDs for current step.
            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (Optional[IntermediateTensors]): Optional intermediate tensors from pipeline stages.
            inputs_embeds (Optional[torch.Tensor]): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        attn_metadata: Any = forward_context.attn_metadata
        if attn_metadata is None:
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds)

        attn_metadata = next(iter(attn_metadata.values()), None)
        if attn_metadata is None or not isinstance(attn_metadata, AscendMetadata):
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds)

        with_prefill = attn_metadata.attn_state not in [
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        ]

        # Full: graph for prefill and decode
        # Decode-Only: runnable for prefill, graph for decode
        if not self.full_mode and self.data_parallel_size > 1:
            num_tokens = forward_context.batch_descriptor.num_tokens
            num_reqs = forward_context.batch_descriptor.num_reqs
            use_xlite_graph = num_reqs is not None and num_tokens <= num_reqs
        else:
            use_xlite_graph = not with_prefill or self.full_mode

        if use_xlite_graph:
            # TODO: When vllm_ascend enables graph mode, attn_metadata.num_decodes
            # will be padded in decode requests. Therefore, it is first fixed using
            # num_decode_tokens. However, in the future, when MTP is enabled, there
            # may be cases where a single request involves multiple tokens, which
            # will need to be solved.
            num_decodes = attn_metadata.num_decode_tokens
            num_prefills = attn_metadata.num_prefills
            batch = num_prefills + num_decodes
            seq_lens = attn_metadata.seq_lens[:batch]
            seq_tensor = torch.cat([torch.tensor([0]), torch.tensor(attn_metadata.actual_seq_lengths_q)], dim=0)
            query_lens = seq_tensor[1:] - seq_tensor[:-1]
            query_lens = query_lens[:batch]
            cached_lens = seq_lens - query_lens

            num_tokens = forward_context.batch_descriptor.num_tokens
            num_actual_tokens = attn_metadata.num_actual_tokens
            xlite_attn_metadata = AttnMeta()
            xlite_attn_metadata.lens = query_lens.tolist()
            xlite_attn_metadata.cached_lens = cached_lens.tolist()
            xlite_attn_metadata.is_prefills = [False] * num_decodes + [True] * num_prefills
            xlite_attn_metadata.block_tables_cpu = attn_metadata.block_tables.cpu().tolist()
            if positions.ndim == 2:
                xlite_attn_metadata.positions = positions[:, : attn_metadata.num_actual_tokens].contiguous()
            else:
                xlite_attn_metadata.positions = positions

            # Compatibility between DP and Non-DP scenarios
            h = self.hidden_states[:num_tokens]
            stream = torch.npu.current_stream().npu_stream
            if inputs_embeds is None:
                self.xlite_model.forward(
                    self.xlite_rt, input_ids, xlite_attn_metadata, self.kv_caches, self.freq_cis, h, stream
                )
            else:
                deepstack_input_embeds = getattr(self.runnable, "deepstack_input_embeds", [])
                xlite_deepstack_input_embeds = [
                    deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
                ]
                self.xlite_model.forward_with_inputs_embeds(
                    self.xlite_rt,
                    inputs_embeds,
                    xlite_attn_metadata,
                    self.kv_caches,
                    self.freq_cis,
                    h,
                    stream,
                    xlite_deepstack_input_embeds,
                )
                if xlite_deepstack_input_embeds and hasattr(self.runnable, "_clear_deepstack_input_embeds"):
                    self.runnable._clear_deepstack_input_embeds(inputs_embeds.size(0))
            return h[:num_actual_tokens]
        else:
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds)
