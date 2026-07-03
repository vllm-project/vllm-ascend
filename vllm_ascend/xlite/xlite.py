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
from xlite._C import AttnMeta, AttnMHA, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.xlite.utils import (
    AttnMetadataRouter,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    get_dotted_attr,
    get_layer_weights,
)

XliteInitResult: TypeAlias = tuple[XModel, torch.Tensor, int, torch.dtype]
XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]

_architecture_strategy_map: dict[str, type[XliteModel]] = {}
"""Mapping from model architecture names in `config.json` to their corresponding xlite adapter classes."""


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

    _attn_metadata_type: type | tuple[type, ...]
    """The expected type of attention metadata in the forward context for this architecture. Used for runtime checks
    before forwarding. See :meth:`XliteWrapper.__call__` for usage."""
    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map and metadata type set."""
        ts = getattr(cls, "_attn_metadata_type", None)
        if ts is None or (not isinstance(ts, type) and not all(isinstance(t, type) for t in ts)):
            raise ValueError(
                f"XliteModel subclass {cls.__name__} must define _attn_metadata_type as a type or a tuple of types."
            )

        arcs = getattr(cls, "_supported_architectures", None)
        if arcs is None:
            raise ValueError(f"XliteModel subclass {cls.__name__} must define _supported_architectures attribute.")
        if isinstance(arcs, str):
            arcs = [arcs]
        for arc in arcs:
            if arc in _architecture_strategy_map:
                raise ValueError(f"Duplicate xlite adapter for architecture {arc}: {_architecture_strategy_map[arc]}")
            _architecture_strategy_map[arc] = cls
        super().__init_subclass__(**kwargs)

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

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()

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
                Sequence[nn.Module], get_dotted_attr(self.runnable.language_model, "model.layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], get_dotted_attr(self.runnable, "model.layers", default=[]))
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

    @staticmethod
    def is_tensor_nz(t: torch.Tensor) -> bool:
        """Check if a tensor is in NZ format.

        Args:
            t (torch.Tensor): The tensor to check.

        Returns:
            bool: True if the tensor is in NZ format, False otherwise.
        """
        format = torch_npu.get_npu_format(t)
        return format == torch_npu.Format.FRACTAL_NZ

    @staticmethod
    def all_tensors_zero(tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None) -> bool:
        """Check if all tensors in the list/tuple are zero tensors.

        Args:
            tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None): The tensors to check.

        Returns:
            bool: True if all tensors are zero tensors (or empty), False otherwise.
        """
        if tensors is None:
            return True
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        if len(tensors) == 0:
            return True
        return all(torch.allclose(t, t.new_zeros(1)) for t in tensors)

    @staticmethod
    def _transform_deq_scale(deq_scale: torch.Tensor) -> torch.Tensor:
        """
        The data format required by the fixpipe hardware is as follows:

        Data is stored in uint64_t, with the upper 32 bits being 0 and the lower 32 bits storing the FP32 format. The
        lower 10 bits of the FP32 format are not involved in computation, and the actual data format is TF32.
        """
        deq_scale_fp32 = deq_scale.to(torch.float32)
        scale = deq_scale_fp32.new_zeros(deq_scale.shape[0] * 2)
        scale[0::2] = deq_scale_fp32[0::1]
        return scale

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
    """xlite adapter base for Llama-like architectures.

    This is the *de facto* base adapter for all xlite-supported architectures and may contain configurations beyond
    Llama-like dense models. `XliteModel` subclasses should inherit from this class unless there is a major divergence.
    """

    _attn_metadata_type = AscendMetadata
    _supported_architectures = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3VLForConditionalGeneration",
    ]

    def _build_model_config(self) -> None:
        xlite_config, vllm_config, hf_config = self.xlite_config, self.vllm_config, self.hf_text_config

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
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        try:
            ep_word_size = get_ep_group().world_size
            xlite_config.moe_ep_size = ep_word_size if vllm_config.parallel_config.enable_expert_parallel else 1
            xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_word_size
        except AssertionError:
            xlite_config.moe_ep_size, xlite_config.moe_tp_size = 1, 1
        xlite_config.experts_weight_transpose = True

        xlite_config.attn_type = AttnMHA
        xlite_config.scoring_func = ScoringFuncSoftmax
        xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
        xlite_config.max_m = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if get_ascend_config().xlite_graph_config.full_mode
            else vllm_config.scheduler_config.max_num_seqs
        )
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

    def _build_model(self) -> None:
        xlite_model, xlite_config, hf_config = self.xlite_model, self.xlite_config, self.hf_text_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        xlite_model.embed = get_dotted_attr(self.runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.runnable, f"{model_prefix}model.norm.weight", raises=True)
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj")
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj")

        mha_qkv_bias = get_layer_weights(layers, "self_attn.qkv_proj.bias")
        xlite_config.qkv_bias = len(mha_qkv_bias) == xlite_config.n_layers
        xlite_model.mha_qkv_bias = mha_qkv_bias if xlite_config.qkv_bias else []
        q_norm = get_layer_weights(layers, "self_attn.q_norm.weight")
        k_norm = get_layer_weights(layers, "self_attn.k_norm.weight")
        xlite_config.qk_norm = len(q_norm) == len(k_norm) == xlite_config.n_layers
        xlite_model.mha_q_norm = q_norm if xlite_config.qk_norm else []
        xlite_model.mha_k_norm = k_norm if xlite_config.qk_norm else []

        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        if not self.quantization:
            return

        if xlite_model.mha_qkv:
            xlite_config.quant_attn_weight_nz = self.is_tensor_nz(xlite_model.mha_qkv[0])
            xlite_config.quant_attn_weight_transpose = True

        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.norm_bias = get_dotted_attr(self.runnable, f"{model_prefix}model.norm.bias", raises=True)
            xlite_model.attn_norm_bias = get_layer_weights(layers, "input_layernorm.bias")
            xlite_model.mlp_norm_bias = get_layer_weights(layers, "post_attention_layernorm.bias")
            if xlite_config.qk_norm:
                xlite_model.mha_q_norm_bias = get_layer_weights(layers, "self_attn.q_norm.bias")
                xlite_model.mha_k_norm_bias = get_layer_weights(layers, "self_attn.k_norm.bias")

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

    def init_matmul_weights(self, layers: Sequence[torch.nn.Module], xlite_prefix: str, model_prefix: str) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
        """
        xlite_model = self.xlite_model
        setattr(xlite_model, xlite_prefix, get_layer_weights(layers, f"{model_prefix}.weight"))
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str):
            setattr(xlite_model, xlite_attr, get_layer_weights(layers, layer_attr))

        deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", post_processor=self._transform_deq_scale)
        if len(deq_scale) > 0:  # static quant
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
        else:
            weight_scale = get_layer_weights(
                layers, f"{model_prefix}.weight_scale", post_processor=self._transform_deq_scale
            )
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)


class QwenMoeXliteModel(LlamaXliteModel):
    """xlite adapter for Qwen MoE architectures."""

    _attn_metadata_type = AscendMetadata
    _supported_architectures = ["Qwen3MoeForCausalLM", "Qwen3VLMoeForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        xlite_model.gate = get_layer_weights(layers, "mlp.gate.weight")
        prefix = "mlp.experts."
        kwargs: WeightGetterConfig = {"secondary_flattening": f"{prefix}local_num_experts", "post_processor": None}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{prefix}w13_weight", **kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{prefix}w2_weight", **kwargs)
        xlite_config.experts_weight_nz = self.is_tensor_nz(xlite_model.re_up_gate[0])

        if self.quantization:
            kwargs["post_processor"] = self._transform_deq_scale
            xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{prefix}w13_weight_scale_fp32", **kwargs)
            xlite_model.re_down_scale = get_layer_weights(layers, f"{prefix}w2_weight_scale", **kwargs)


class Glm4MoeXliteModel(LlamaXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    _attn_metadata_type = AscendMetadata
    _supported_architectures = ["Glm4MoeForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        if hasattr(hf_config, "partial_rotary_factor"):
            partial_rotary_factor = hf_config.partial_rotary_factor
        else:
            partial_rotary_factor = getattr(hf_config, "rope_parameters", {}).get("partial_rotary_factor", 1.0)
        xlite_config.rope_head_dim = int(xlite_config.head_dim * partial_rotary_factor)
        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor
        xlite_config.gate_captured = False

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        xlite_model.gate = get_layer_weights(layers, "mlp.gate.weight")
        # NOTE: type conversion for numerical stability in xlite's implementation
        xlite_model.gate_bias = get_layer_weights(
            layers, "mlp.gate.e_score_correction_bias", post_processor=lambda b: b.to(torch.float32)
        )
        self.init_matmul_weights(layers, "se_up_gate", "mlp.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", "mlp.shared_experts.down_proj")

        prefix = "mlp.experts."
        kwargs: WeightGetterConfig = {"secondary_flattening": f"{prefix}local_num_experts", "post_processor": None}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{prefix}w13_weight", **kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{prefix}w2_weight", **kwargs)
        if xlite_model.re_up_gate:
            xlite_config.experts_weight_nz = self.is_tensor_nz(xlite_model.re_up_gate[0])

        if self.quantization:
            kwargs["post_processor"] = self._transform_deq_scale
            xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{prefix}w13_weight_scale_fp32", **kwargs)
            xlite_model.re_down_scale = get_layer_weights(layers, f"{prefix}w2_weight_scale", **kwargs)


class MiniMaxM2XliteModel(LlamaXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    _attn_metadata_type = AscendMetadata
    _supported_architectures = ["MiniMaxM2ForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.rope_head_dim = hf_config.rotary_dim
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_local_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.qk_norm_full = True
        xlite_config.scoring_func = ScoringFuncSigmoid

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        xlite_model.gate = get_layer_weights(layers, "block_sparse_moe.gate.weight")
        # NOTE: type conversion for numerical stability in xlite's implementation
        xlite_model.gate_bias = get_layer_weights(
            layers, "block_sparse_moe.e_score_correction_bias", post_processor=lambda b: b.to(torch.float32)
        )

        prefix = "block_sparse_moe.experts."
        kwargs: WeightGetterConfig = {"secondary_flattening": f"{prefix}local_num_experts", "post_processor": None}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{prefix}w13_weight", **kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{prefix}w2_weight", **kwargs)
        if xlite_model.re_up_gate:
            xlite_config.experts_weight_nz = self.is_tensor_nz(xlite_model.re_up_gate[0])

        if self.quantization:
            kwargs["post_processor"] = self._transform_deq_scale
            xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{prefix}w13_weight_scale_fp32", **kwargs)
            xlite_model.re_down_scale = get_layer_weights(layers, f"{prefix}w2_weight_scale", **kwargs)


def get_adapter_xlite_model(runnable: nn.Module, vllm_config: VllmConfig) -> XliteModel:
    """Look up and initialize the appropriate xlite model adapter based on the architecture specified in vLLM config and
    the runnable model.

    Args:
        runnable (nn.Module): The runnable model instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteModel: An initialized xlite model adapter ready for inference.
    """
    architecture = vllm_config.model_config.architectures[0]
    if not (strategy_class := _architecture_strategy_map.get(architecture)):
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(runnable, vllm_config)


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths."""

    def __init__(self, runnable: nn.Module, vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            runnable (nn.Module): The runnable model implementation.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.runnable = runnable
        self.device = device
        self.full_mode = get_ascend_config().xlite_graph_config.full_mode

        rank = torch.distributed.get_rank()
        local_rank = get_world_group().local_rank
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size

        self.adapter_xlite_model = get_adapter_xlite_model(runnable, vllm_config)
        (self.xlite_model, self.freq_cis, hidden_size, dtype) = self.adapter_xlite_model.initialize()
        xlite_config = self.adapter_xlite_model.xlite_config
        self.xlite_rt = Runtime(
            devid=local_rank,
            size=0,
            rank=rank,
            tp_size=xlite_config.def_tp_size,
            dp_size=xlite_config.def_dp_size,
            moe_tp_size=xlite_config.moe_tp_size,
            moe_ep_size=xlite_config.moe_ep_size,
        )

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if rank == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_states = torch.empty(max_num_tokens, hidden_size, device=self.device, dtype=dtype)

    def __getattr__(self, key: str) -> Any:
        """Proxy unknown attributes to the wrapped runnable model.

        Args:
            key (str): The attribute name requested by the caller.

        Raises:
            AttributeError: If neither wrapper nor runnable has the attribute.

        Returns:
            Any: Attribute value resolved from the runnable.
        """
        try:
            return getattr(self.runnable, key)
        except Exception:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from None

    def unwrap(self) -> Callable:
        """Return the original runnable callable. See :meth:`ACLGraphWrapper.unwrap` for details.

        Returns:
            Callable: Original model runnable.
        """
        # in case we need to access the original runnable.
        if isinstance(runnable := self.runnable, ACLGraphWrapper):
            return runnable.unwrap()
        return runnable

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
        **model_kwargs: Any,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor): Token IDs for current step.
            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (Optional[IntermediateTensors]): Optional intermediate tensors from pipeline stages.
            inputs_embeds (Optional[torch.Tensor]): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).
            **model_kwargs (Any): Additional keyword arguments for the runnable.

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        if getattr(forward_context, "in_profile_run", False):
            if self.full_mode:
                # In full mode, xlite handles both prefill and decode, and aclgraph runnable should not reserve memory.
                # This is to avoid redundant memory allocation that reduces KV cache capacity and regresses performance.
                # NOTE: returning a single hidden state tensor may break the vLLM pipeline if the runnable expects a
                # tuple of outputs, e.g., (hidden_states, aux_hidden_states) under certain speculative scenarios
                return self.hidden_states
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata: Any = forward_context.attn_metadata
        if attn_metadata is None:
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        attn_metadata = next(iter(attn_metadata.values()), None)
        if not isinstance(attn_metadata, self.adapter_xlite_model._attn_metadata_type):
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        with_prefill = attn_metadata.attn_state not in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )

        # Full: graph for prefill and decode
        # Decode-Only: runnable for prefill, graph for decode
        if not self.full_mode and self.data_parallel_size > 1:
            num_tokens = forward_context.batch_descriptor.num_tokens
            num_reqs = forward_context.batch_descriptor.num_reqs
            use_xlite_graph = num_reqs is not None and num_tokens <= num_reqs
        else:
            use_xlite_graph = not with_prefill or self.full_mode

        if not use_xlite_graph:
            # fall back to runnable for prefill in decode-only mode
            # or when the number of tokens exceeds the graph capacity in non-full mode
            return self.runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        seq_lens = attn_metadata_router.seq_lens
        cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :].to(device=seq_lens.device)
        query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
        cached_lens = torch.clamp(seq_lens - query_lens, min=0)

        num_tokens = forward_context.batch_descriptor.num_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        xlite_attn_metadata = AttnMeta()
        xlite_attn_metadata.lens = query_lens.tolist()
        xlite_attn_metadata.cached_lens = cached_lens.tolist()
        xlite_attn_metadata.block_tables_cpu = attn_metadata_router.block_tables.tolist()
        if positions.ndim == 2:
            xlite_attn_metadata.positions = positions[:, :num_actual_tokens].contiguous()
            positions = positions[0]
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
