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

import math
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import regex as re
import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import AttnDSA, AttnHybrid, AttnMeta, AttnMHA, AttnMLA, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState, AscendMetadata
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.xlite.utils import (
    AttnMetadataRouter,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    get_dotted_attr,
    get_layer_weights,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunner

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]

_architecture_strategy_map: dict[str, type["XliteModelBase"]] = {}
"""Mapping from model architecture names in `config.json` to their corresponding xlite adapter classes."""


class XliteModelBase(ABC):
    """Base adapter for converting vLLM models into xlite runtime models.

    Subclasses are responsible for mapping architecture-specific configuration and weights into the `xlite._C.Model`
    interface.

    Attributes:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        npu_runnable (nn.Module): The original runnable model used by vLLM-ascend. Used as the source of truth for
            weight extraction for xlite model construction.
        vllm_config (VllmConfig): The configuration object provided by vLLM. Used to build xlite configuration at
            runtime.
        device (torch.device): The device on which the model is loaded. Typically an Ascend NPU device.
        dtype (torch.dtype): The data type used for model weights and computations. Derived from `vllm_config`.
        rottary_embed (AscendRotaryEmbedding | None): The rotary embedding module extracted from the NPU runnable model.
        xlite_config (XModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (XModel): Native xlite model container populated by subclasses.
        rope_cossin_cache (torch.Tensor): Precomputed RoPE frequency cache tensor for rotary positional embeddings.
    """

    from vllm_ascend.attention.attention_v1 import AscendMetadata

    _attn_metadata_type: type | tuple[type, ...] = AscendMetadata
    """The expected type of attention metadata in the forward context for this architecture. Used for runtime checks
    before forwarding. See :meth:`XliteWrapper.__call__` for usage."""
    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

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

    def __init__(self, npu_runner: "XliteModelRunner", vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            npu_runner (XliteModelRunner): The NPU model runner instance.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner = npu_runner
        self.npu_runnable = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = self.npu_runner.device
        self.dtype = vllm_config.model_config.dtype
        self.rottary_embed: AscendRotaryEmbedding | None = self._get_rope()
        if self.rottary_embed:
            logger.info_once(
                "xlite: found rotary embedding module `%s` in the NPU runnable.",
                re.sub(r"\s+", " ", repr(self.rottary_embed).replace("\n", "").strip()),
            )
        else:
            logger.warning_once("xlite: no rotary embedding module found in `%s`", self.npu_runnable)

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()
        self._build_model_config()
        self._build_model()
        self.xlite_model.init(self.xlite_config, torch.distributed.get_rank())
        self.rope_cossin_cache = self._precompute_rope_cache()

        if torch.distributed.get_rank() == 0:
            logger.info_once("xlite: built model config: %s", self.xlite_config)
            logger.info_once("xlite: built model weights: %s", self.xlite_model)

    def extract_kv_cache(self, kv_caches: list[tuple[torch.Tensor, ...]], /) -> list[tuple[torch.Tensor, ...]]:
        """Extract xlite-compatible KV cache from the vLLM-ascend KV cache.

        Args:
            kv_caches (list[tuple[torch.Tensor, ...]]): The KV cache from the upstream vLLM/vLLM-ascend.

        Returns:
            list[tuple[torch.Tensor, ...]]: The KV cache in the format expected by xlite model implementations.
        """
        # naively, the `kv_cache` is a list of <n_layers> tuples, each tuple contains two tensors of shape
        # [n_blocks, block_size, n_heads, head_dim] for K and V respectively.
        return kv_caches

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
        if hasattr(self.npu_runnable, "language_model"):
            layers = cast(
                Sequence[nn.Module], get_dotted_attr(self.npu_runnable.language_model, "model.layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], get_dotted_attr(self.npu_runnable, "model.layers", default=[]))
            prefix = ""
        return layers, prefix

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Extract the rotary embedding module from the NPU runnable model :data:`npu_runnable`.

        Args:
            layer_idx (int): The index of the transformer layer to extract the rotary embedding from. Defaults to 0.

        Returns:
            AscendRotaryEmbedding | None: The rotary embedding module found in the specified transformer layer, or None
            if not found.
        """
        layers, _ = self._get_layers_and_model_prefix()
        rottary_embed: AscendRotaryEmbedding | None = get_dotted_attr(layers, f"{layer_idx}.self_attn.rotary_emb")
        if not rottary_embed:
            logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return rottary_embed

    def _precompute_rope_cache(self) -> torch.Tensor:
        """Precomputes frequency-based complex exponential values for rotary positional embeddings (RoPE).

        This method generates the RoPE frequency cache (cosine and sine values) required by the xlite attention
        implementation. The cache should be precomputed on the NPU device to avoid unnecessary host-device transfers
        during inference.

        Returns:
            torch.Tensor: The precomputed RoPE frequency cache tensor ready for use in xlite attention computations.

        Notes:
            :meth:`_build_model_config` should be called prior to this method.
        """
        if not self.rottary_embed:
            return torch.empty(2, dtype=torch.float32, device=self.device)
        cossin_cache: torch.Tensor | None = getattr(self.rottary_embed, "cossin_cache", None)
        if not isinstance(cossin_cache, torch.Tensor):
            cossin_cache = self.rottary_embed._compute_cos_sin_cache()
        return cossin_cache.to(dtype=torch.float32, device=self.device)

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
    def all_tensors_zero(tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None) -> bool:
        """Check if all tensors in the list/tuple are zero tensors.

        Args:
            tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None): The tensors to check.

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
        """Repack a dequantization scale into the fixpipe hardware's expected format.

        Data is stored in ``uint64_t``: the upper 32 bits are 0 and the lower 32 bits hold an FP32 value whose lower 10
        bits are not involved in computation, making the effective data format TF32.

        Args:
            deq_scale (torch.Tensor): The original dequantization scale tensor.

        Returns:
            torch.Tensor: The repacked scale tensor for fixpipe computation.
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


class StandardXliteModel(XliteModelBase):
    """xlite adapter base for standard architectures.

    This is the *de facto* base adapter for all xlite-supported architectures. Configurations for various model types,
    dense and MoE, are included.

    Subclasses of :class:`XliteModelBase`, i.e., xlite model adapters, should inherit from :class:`StandardXliteModel`,
    unless there is a major divergence.

    **Developer Guide**: The :class:`StandardXliteModel` class is designed to be a flexible base for xlite adapters.
    When developing a new adapter for a specific architecture to dock with the xlite backend, consider adding weight
    loading lines in :meth:`StandardXliteModel._build_model` directly if feasible, using the highly flexible and
    error-checked `get_layer_weights` utility. For :meth:`_build_model_config`, override it in the subclass only if the
    architecture has unique configuration needs.
    """

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
        xlite_config.rope_head_dim = self.rottary_embed.rotary_dim if self.rottary_embed else xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = getattr(
            hf_config, "intermediate_size", getattr(hf_config, "moe_intermediate_size", 0)
        )
        xlite_config.def_tp_size = tp_size = get_tensor_model_parallel_world_size()
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
        max_m = vllm_config.scheduler_config.max_num_batched_tokens
        if not get_ascend_config().xlite_graph_config.full_mode:
            # decode-sized token budget: max_num_seqs * (1 + num_speculative_tokens) instead of max_num_batched_tokens
            n_spec_tokens = getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
            max_m = min(max_m, vllm_config.scheduler_config.max_num_seqs * (n_spec_tokens + 1))
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size
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

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj")
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj")

        with xlite_model.condition(lambda tensors: len(tensors) == xlite_config.n_layers):
            xlite_model.mha_q_norm = get_layer_weights(layers, "self_attn.q_norm.weight")
            xlite_model.mha_k_norm = get_layer_weights(layers, "self_attn.k_norm.weight")
            xlite_config.qk_norm = bool(xlite_model.mha_q_norm) and bool(xlite_model.mha_k_norm)

        # Dense MLP weights
        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        # MoE-specific weights
        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        xlite_model.gate_bias = get_layer_weights(
            layers,
            f"{mlp_prefix}.gate.e_score_correction_bias",
            f"{mlp_prefix}.e_score_correction_bias",  # for MiniMax-2.x compatibility
            post_processor=lambda b: b.to(torch.float32),  # type conversion for numerical stability in xlite backend
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")

        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])

        # bias terms
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mha_qkv_bias = get_layer_weights(layers, "self_attn.qkv_proj.bias")
            xlite_config.qkv_bias = len(xlite_model.mha_qkv_bias) == xlite_config.n_layers
            xlite_model.norm_bias = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.bias", raises=True)
            xlite_model.attn_norm_bias = get_layer_weights(layers, "input_layernorm.bias")
            xlite_model.mlp_norm_bias = get_layer_weights(layers, "post_attention_layernorm.bias")
            if xlite_config.qk_norm:
                xlite_model.mha_q_norm_bias = get_layer_weights(layers, "self_attn.q_norm.bias")
                xlite_model.mha_k_norm_bias = get_layer_weights(layers, "self_attn.k_norm.bias")

        if not self.quantization:
            return

        xlite_config.quant_attn_weight_nz = bool(xlite_model.mha_qkv) and self.is_tensor_nz(xlite_model.mha_qkv[0])
        xlite_config.quant_attn_weight_transpose = bool(xlite_model.mha_qkv)

        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def init_matmul_weights(
        self,
        layers: Sequence[torch.nn.Module],
        xlite_prefix: str,
        model_prefix: str,
        *,
        mask: Sequence[bool] | None = None,
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
        """
        xlite_model = self.xlite_model
        setattr(xlite_model, xlite_prefix, get_layer_weights(layers, f"{model_prefix}.weight", mask=mask))
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str) -> None:
            setattr(xlite_model, xlite_attr, get_layer_weights(layers, layer_attr, mask=mask))

        wt_kwargs: WeightGetterConfig = {"mask": mask, "post_processor": self._transform_deq_scale}
        deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
        if len(deq_scale) > 0:  # static quant
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
        else:
            weight_scale = get_layer_weights(layers, f"{model_prefix}.weight_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove the type conversion once `xlite` accepts and defaults the cache in FP32 format
        return super()._precompute_rope_cache().to(dtype=self.dtype, device=self.device)


class QwenMoeXliteModel(StandardXliteModel):
    """xlite adapter for Qwen MoE architectures."""

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


class Glm4MoeXliteModel(StandardXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    _supported_architectures = ["Glm4MoeForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor


class DeepseekV3XliteModel(Glm4MoeXliteModel):
    """xlite adapter for DeepseekV3 MoE architectures with MLA attention."""

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

    _attn_metadata_type = AscendMLAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV3ForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # MLA attention type
        xlite_config.attn_type = AttnMLA
        xlite_config.n_kv_heads = 1  # MLA uses latent cache
        xlite_config.head_dim = 0  # Ignored by MLA

        # MLA dimensions (override Llama's head_dim-based rope_head_dim)
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.qk_nope_head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.kv_lora_rank
        xlite_config.v_head_dim = hf_config.v_head_dim
        xlite_config.softmax_scale = (hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim) ** -0.5
        # correct softmax_scale for yarn-style RoPE if max_seq_len > original_max_position_embeddings
        rope_params: dict[str, int | float | str] = getattr(hf_config, "rope_parameters", {})
        original_max_len: int = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)  # type: ignore[assignment]
        if xlite_config.max_seq_len > original_max_len and "mscale" in rope_params and "factor" in rope_params:
            mscale: float = 1.0 + 0.1 * rope_params["mscale"] * math.log(rope_params["factor"])  # type: ignore[operator,arg-type]
            xlite_config.softmax_scale *= mscale**2

        # MoE configuration (from Glm4MoeXliteModel, adapted for DeepseekV3)
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        # MLA attention weights
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight")
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight")
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV")
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T")
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(wuv_lst[0])
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(wuk_lst[0])

        if not self.quantization:
            return

        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(wt_lst[0])
        self.xlite_config.quant_attn_weight_transpose = True
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mla_q_norm_bias = get_layer_weights(layers, "self_attn.q_a_layernorm.bias")
            xlite_model.mla_kv_norm_bias = get_layer_weights(layers, "self_attn.kv_a_layernorm.bias")

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove this method once `xlite` no longer requires complex64 format
        cossin_cache = super(StandardXliteModel, self)._precompute_rope_cache()
        if not self.rottary_embed:
            return cossin_cache.to(dtype=torch.complex64, device=self.device)
        # For MLA attention, we need to return the cache as a complex instead of separate cos/sin tensors
        if not (isinstance(cossin_cache, torch.Tensor) and cossin_cache.ndim == 2 and cossin_cache.shape[1] % 2 == 0):
            raise ValueError("RoPE cache must be a 2D tensor with even second dimension for MLA attention.")
        sep = cossin_cache.shape[-1] // 2
        return torch.complex(cossin_cache[:, :sep], cossin_cache[:, sep:]).to(dtype=torch.complex64, device=self.device)


class DeepseekV32XliteModel(DeepseekV3XliteModel):
    """xlite adapter for Deepseek-V3.2/GLM-5.x architectures with Deepseek sparse attention (DSA)."""

    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    _attn_metadata_type = AscendSFAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnDSA
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        index_types: list[str] = getattr(hf_config, "indexer_types", []) or []
        xlite_config.index_full_mask = list(map(lambda x: str(x).lower().startswith("full"), index_types))

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model = self.xlite_model
        layers, _ = self._get_layers_and_model_prefix()

        mask: list[bool] | None = getattr(self.xlite_config, "index_full_mask", None) or None
        self.init_matmul_weights(layers, "index_q_b", f"{(prefix := 'self_attn.indexer')}.wq_b", mask=mask)
        xlite_model.index_k_weights_proj = get_layer_weights(layers, f"{prefix}.wk_weights_proj.weight", mask=mask)
        # xlite backend expects index_k_norm and index_k_norm_bias to have dtype=torch.float32 or the model dtype
        xlite_model.index_k_norm = get_layer_weights(layers, f"{prefix}.k_norm.weight", mask=mask)
        xlite_model.index_k_norm_bias = get_layer_weights(layers, f"{prefix}.k_norm.bias", mask=mask)

    def extract_kv_cache(self, kv_caches: list[tuple[torch.Tensor, ...]], /) -> list[tuple[torch.Tensor, ...]]:
        from vllm_ascend.xlite.utils import _DUMMY_TENSOR

        # For DSA, the kv_caches are passed as [(indexer_k_cache,), (k_nope_cache, k_pe_cache), ..., <mtp_layer>]
        # TODO: consider the compatibility with `enable_sparse_sfa_c8` and `enable_sparse_li_c8`
        indexer_caches: list[tuple[torch.Tensor, ...]] = []
        mla_caches: list[tuple[torch.Tensor, ...]] = []

        idx = 0
        index_mask = getattr(self.xlite_config, "index_full_mask", None) or [True] * self.xlite_config.n_layers
        dummy_indexer_cache = (_DUMMY_TENSOR,)
        for mask in index_mask:
            indexer_caches.append(kv_caches[idx] if mask else dummy_indexer_cache)
            mla_caches.append(kv_caches[idx + int(mask)])
            idx += 1 + int(mask)
        return [mla_c[:2] + indexer_c[:1] for mla_c, indexer_c in zip(mla_caches, indexer_caches)]


class MiniMaxM2XliteModel(StandardXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    _supported_architectures = ["MiniMaxM2ForCausalLM"]
    _decoder_layer_mlp_module = "block_sparse_moe"

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


def _gemma_norm_to_xlite(weight: torch.Tensor) -> torch.Tensor:
    """Gemma RMSNorm ``x*(1+w)`` -> xlite ``x*w``."""
    return (weight.float() + 1.0).to(weight.dtype).contiguous()


def _flatten_expert_weights(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Unbind packed ``[E, ...]`` expert tensors into per-expert 2D views."""
    flat: list[torch.Tensor] = []
    for weight in weights:
        if weight.dim() >= 3:
            flat.extend(t.contiguous() for t in weight.unbind(0))
        else:
            flat.append(weight if weight.is_contiguous() else weight.contiguous())
    return flat


def _pack_mha_qkv_weight(qkv_weight: torch.Tensor, head_dim: int, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """vLLM interleaved [Q|Gate]/K|V -> xlite [Q|K|V|Gate] (weights)."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    expected = q_size * 2 + kv_size * 2
    if qkv_weight.shape[0] != expected:
        raise ValueError(f"Unexpected fused QKV rows {qkv_weight.shape[0]}, expected {expected}")
    qkv_weight = qkv_weight.contiguous()
    q_gate = qkv_weight[: q_size * 2].view(num_heads, 2, head_dim, -1)
    q_w = q_gate[:, 0].reshape(q_size, -1).contiguous()
    g_w = q_gate[:, 1].reshape(q_size, -1).contiguous()
    k_w = qkv_weight[q_size * 2 : q_size * 2 + kv_size]
    v_w = qkv_weight[q_size * 2 + kv_size :]
    return torch.cat([q_w, k_w, v_w, g_w], dim=0)


def _pack_mha_qkv_bias(qkv_bias: torch.Tensor, head_dim: int, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Reorder the fused QKV bias to match the packed [Q|K|V|Gate] weight layout."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate_b = qkv_bias[: q_size * 2].view(num_heads, 2, head_dim)
    return torch.cat(
        [
            q_gate_b[:, 0].reshape(-1),
            qkv_bias[q_size * 2 : q_size * 2 + kv_size],
            qkv_bias[q_size * 2 + kv_size :],
            q_gate_b[:, 1].reshape(-1),
        ],
        dim=0,
    )


def _split_linear_qkvz(weight: torch.Tensor, key_dim: int, value_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    qkv_rows = key_dim * 2 + value_dim
    if weight.shape[0] != qkv_rows + value_dim:
        raise ValueError(f"Unexpected in_proj_qkvz rows {weight.shape[0]}")
    return weight[:qkv_rows].contiguous(), weight[qkv_rows:].contiguous()


def _split_linear_ba(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.shape[0] % 2:
        raise ValueError(f"Unexpected in_proj_ba rows {weight.shape[0]}")
    b_w, a_w = weight.chunk(2, dim=0)
    return b_w.contiguous(), a_w.contiguous()


def _resolve_full_attention_interval(hf_config: PretrainedConfig) -> int:
    interval = getattr(hf_config, "full_attention_interval", None)
    if isinstance(interval, int) and interval > 0:
        return interval
    # Derive the period from the layer-type list. Use the distance between the
    # first two full-attention layers: models may start with a full layer
    # (index 0), where `idx + 1` would wrongly yield interval == 1.
    full_indices = [
        idx
        for idx, layer_type in enumerate(getattr(hf_config, "layer_types", None) or [])
        if layer_type == "full_attention"
    ]
    if len(full_indices) >= 2:
        return full_indices[1] - full_indices[0]
    if len(full_indices) == 1:
        return full_indices[0] + 1
    return 4


class Qwen3_5XliteModel(StandardXliteModel):
    """xlite adapter for the dense Qwen3.5 hybrid (MHA + GDN) architecture."""

    _attn_metadata_type = AscendMetadata
    _supported_architectures = ["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnHybrid
        xlite_config.attn_output_gate = True
        xlite_config.full_attention_interval = _resolve_full_attention_interval(hf_config)
        xlite_config.linear_num_k_heads = hf_config.linear_num_key_heads
        xlite_config.linear_num_v_heads = hf_config.linear_num_value_heads
        xlite_config.linear_key_head_dim = hf_config.linear_key_head_dim
        xlite_config.linear_value_head_dim = hf_config.linear_value_head_dim
        xlite_config.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        partial = getattr(hf_config, "partial_rotary_factor", None) or (
            getattr(hf_config, "rope_parameters", None) or {}
        ).get("partial_rotary_factor", 1.0)
        xlite_config.rope_head_dim = int(xlite_config.head_dim * partial)
        xlite_config.qk_norm = True
        xlite_config.qkv_bias = bool(
            getattr(hf_config, "attention_bias", False) or getattr(hf_config, "qkv_bias", False)
        )
        xlite_config.mrope_section = []
        xlite_config.mrope_interleaved = False
        # Full-hybrid: prefill and decode both run on xlite, so the graph must
        # cover the whole chunked-prefill budget instead of the decode-sized one.
        tp_size = max(int(xlite_config.def_tp_size), 1)
        max_m = self.vllm_config.scheduler_config.max_num_batched_tokens
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size
        kernel_sizes = AscendAttentionBackend.get_supported_kernel_block_sizes()
        if kernel_sizes and xlite_config.block_size != int(kernel_sizes[0]):
            logger.info("xlite hybrid block_size: manager=%s -> kernel=%s", xlite_config.block_size, kernel_sizes[0])
            xlite_config.block_size = int(kernel_sizes[0])

        # Per-layer MHA/GDN role mask (same pattern as DeepseekV32's index_full_mask).
        layers, _ = self._get_layers_and_model_prefix()
        if len(layers) != int(xlite_config.n_layers):
            raise ValueError(f"Layer count mismatch: {len(layers)} vs {xlite_config.n_layers}")
        self._full_attention_mask = [
            getattr(layer, "layer_type", None) == "full_attention" or hasattr(layer, "self_attn") for layer in layers
        ]

    def _build_model(self) -> None:
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        hf_config = self.hf_text_config
        layers, model_prefix = self._get_layers_and_model_prefix()
        mask = self._full_attention_mask
        gdn_mask = [not m for m in mask]
        # Use the first MHA layer's module attributes: under TP they carry the
        # per-rank shard sizes (config fields are the global values).
        mha_layer = next(layer for layer in layers if hasattr(layer, "self_attn"))
        qkv_kwargs = {
            "head_dim": int(mha_layer.self_attn.head_dim),
            "num_heads": int(mha_layer.self_attn.num_heads),
            "num_kv_heads": int(mha_layer.self_attn.num_kv_heads),
        }
        tp_size = max(int(xlite_config.def_tp_size), 1)
        key_dim = int(xlite_config.linear_num_k_heads) * int(xlite_config.linear_key_head_dim) // tp_size
        value_dim = int(xlite_config.linear_num_v_heads) * int(xlite_config.linear_value_head_dim) // tp_size
        to_model_dtype = lambda t: t.detach().to(dtype=self.dtype).contiguous()  # noqa: E731

        # Embeddings and head (tied embeddings reuse the embedding table).
        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = _gemma_norm_to_xlite(
            get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        )
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        # Every layer-level RMSNorm in Qwen3.5 is Gemma-style (x * (1 + w)).
        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight", post_processor=_gemma_norm_to_xlite)
        xlite_model.mlp_norm = get_layer_weights(
            layers, "post_attention_layernorm.weight", post_processor=_gemma_norm_to_xlite
        )

        # MHA layers (full attention only; GDN slots are dummy placeholders).
        # get_layer_weights feeds dummy placeholders through post_processor too;
        # guard with numel so the layout asserts only see real weights.
        xlite_model.mha_qkv = get_layer_weights(
            layers,
            "self_attn.qkv_proj.weight",
            mask=mask,
            post_processor=lambda w: w if w.numel() <= 1 else _pack_mha_qkv_weight(w, **qkv_kwargs),
        )
        xlite_model.attn_out = get_layer_weights(layers, "self_attn.o_proj.weight", mask=mask)
        xlite_model.mha_q_norm = get_layer_weights(
            layers, "self_attn.q_norm.weight", mask=mask, post_processor=_gemma_norm_to_xlite
        )
        xlite_model.mha_k_norm = get_layer_weights(
            layers, "self_attn.k_norm.weight", mask=mask, post_processor=_gemma_norm_to_xlite
        )
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mha_qkv_bias = get_layer_weights(
                layers,
                "self_attn.qkv_proj.bias",
                mask=mask,
                post_processor=lambda b: b if b.numel() <= 1 else _pack_mha_qkv_bias(b, **qkv_kwargs),
            )
            xlite_config.qkv_bias = bool(xlite_model.mha_qkv_bias) and not self.all_tensors_zero(
                xlite_model.mha_qkv_bias
            )

        # GDN layers (linear attention; MHA slots are dummy placeholders).
        # Note: linear_attn.norm is RMSNormGated (standard x * w), not Gemma-style.
        # get_layer_weights feeds dummy placeholders through post_processor too;
        # guard with numel so the layout asserts only see real weights.
        gdn_guard = lambda w, fn: w if w.numel() <= 1 else fn(w)  # noqa: E731
        xlite_model.linear_in_proj_qkv = get_layer_weights(
            layers,
            "linear_attn.in_proj_qkvz.weight",
            mask=gdn_mask,
            post_processor=lambda w: gdn_guard(w, lambda t: _split_linear_qkvz(t, key_dim, value_dim)[0]),
        )
        xlite_model.linear_in_proj_z = get_layer_weights(
            layers,
            "linear_attn.in_proj_qkvz.weight",
            mask=gdn_mask,
            post_processor=lambda w: gdn_guard(w, lambda t: _split_linear_qkvz(t, key_dim, value_dim)[1]),
        )
        xlite_model.linear_in_proj_b = get_layer_weights(
            layers,
            "linear_attn.in_proj_ba.weight",
            mask=gdn_mask,
            post_processor=lambda w: gdn_guard(w, lambda t: _split_linear_ba(t)[0]),
        )
        xlite_model.linear_in_proj_a = get_layer_weights(
            layers,
            "linear_attn.in_proj_ba.weight",
            mask=gdn_mask,
            post_processor=lambda w: gdn_guard(w, lambda t: _split_linear_ba(t)[1]),
        )
        xlite_model.linear_conv1d = get_layer_weights(
            layers,
            "linear_attn.conv1d.weight",
            mask=gdn_mask,
            post_processor=lambda w: gdn_guard(w, lambda t: (t.unsqueeze(1) if t.ndim == 2 else t).contiguous()),
        )
        xlite_model.linear_a_log = get_layer_weights(
            layers, "linear_attn.A_log", mask=gdn_mask, post_processor=to_model_dtype
        )
        xlite_model.linear_dt_bias = get_layer_weights(
            layers, "linear_attn.dt_bias", mask=gdn_mask, post_processor=to_model_dtype
        )
        xlite_model.linear_norm = get_layer_weights(layers, "linear_attn.norm.weight", mask=gdn_mask)
        xlite_model.linear_out_proj = get_layer_weights(layers, "linear_attn.out_proj.weight", mask=gdn_mask)

        # Dense MLP layers.
        xlite_model.mlp_up_gate = get_layer_weights(layers, "mlp.gate_up_proj.weight")
        xlite_model.mlp_down = get_layer_weights(layers, "mlp.down_proj.weight")

        # MoE layers (empty on dense models; populated for Qwen3_5MoeXliteModel).
        mlp_prefix = self._decoder_layer_mlp_module
        dbg_mlp = getattr(layers[0], mlp_prefix, None)
        logger.warning(
            "XLITE-DBG mlp_prefix=%s mlp_type=%s has(gate/shared_expert_gate/shared_expert/experts/routed)="
            "(%s/%s/%s/%s/%s)",
            mlp_prefix,
            type(dbg_mlp).__name__,
            hasattr(dbg_mlp, "gate") if dbg_mlp is not None else None,
            hasattr(dbg_mlp, "shared_expert_gate") if dbg_mlp is not None else None,
            hasattr(dbg_mlp, "shared_expert") if dbg_mlp is not None else None,
            hasattr(dbg_mlp, "experts") if dbg_mlp is not None else None,
            (hasattr(getattr(dbg_mlp, "experts", None), "routed_experts")) if dbg_mlp is not None else None,
        )
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        xlite_model.gate_bias = get_layer_weights(
            layers,
            f"{mlp_prefix}.gate.bias",
            f"{mlp_prefix}.gate.e_score_correction_bias",
            post_processor=lambda b: b.to(torch.float32),
        )
        # Qwen3 MoE models name the module shared_experts (plural); Qwen3.5
        # uses shared_expert (singular) — try both, first non-empty wins.
        xlite_model.se_up_gate = get_layer_weights(
            layers,
            f"{mlp_prefix}.shared_experts.gate_up_proj.weight",
            f"{mlp_prefix}.shared_expert.gate_up_proj.weight",
        )
        xlite_model.se_down = get_layer_weights(
            layers, f"{mlp_prefix}.shared_experts.down_proj.weight", f"{mlp_prefix}.shared_expert.down_proj.weight"
        )
        if xlite_model.se_up_gate and xlite_model.se_down:
            # y = routed + sigmoid(se_gate(x)) * shared(x): the gate must be bound
            # whenever a shared expert exists, otherwise shared contributes at full weight.
            xlite_model.se_gate = get_layer_weights(layers, f"{mlp_prefix}.shared_expert_gate.weight")
        re_kwargs: WeightGetterConfig = {
            "secondary_flattening": f"{mlp_prefix}.experts.routed_experts.local_num_experts"
        }
        xlite_model.re_up_gate = get_layer_weights(
            layers, f"{mlp_prefix}.experts.routed_experts.w13_weight", **re_kwargs
        )
        xlite_model.re_down = get_layer_weights(layers, f"{mlp_prefix}.experts.routed_experts.w2_weight", **re_kwargs)
        if not xlite_model.re_up_gate:
            re_kwargs = {"secondary_flattening": f"{mlp_prefix}.experts.local_num_experts"}  # type: ignore[assignment]
            xlite_model.re_up_gate = get_layer_weights(layers, f"{mlp_prefix}.experts.w13_weight", **re_kwargs)
            xlite_model.re_down = get_layer_weights(layers, f"{mlp_prefix}.experts.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])
        # xlite group_matmul transpose=True means [K, N] (in, out); infer from the actual slice.
        if xlite_model.re_up_gate:
            w0 = xlite_model.re_up_gate[0]
            if w0.dim() == 2 and w0.shape[0] == xlite_config.hidden_size:
                xlite_config.experts_weight_transpose = True
            elif w0.dim() == 2 and w0.shape[1] == xlite_config.hidden_size:
                xlite_config.experts_weight_transpose = False


class Qwen3_5MoeXliteModel(Qwen3_5XliteModel):
    """xlite adapter for the Qwen3.5 MoE hybrid (MHA + GDN + sparse MoE) architecture."""

    _supported_architectures = ["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = (
            1 if int(getattr(hf_config, "shared_expert_intermediate_size", 0) or 0) > 0 else 0
        )
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = bool(getattr(hf_config, "norm_topk_prob", True))


def get_adapter_xlite_model(npu_runner: "XliteModelRunner", vllm_config: VllmConfig) -> XliteModelBase:
    """Look up and initialize the appropriate xlite model adapter based on the architecture specified in vLLM config and
    the runnable model.

    Args:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteModelBase: An initialized xlite model adapter ready for inference.
    """
    architecture = vllm_config.model_config.architectures[0]
    if not (strategy_class := _architecture_strategy_map.get(architecture)):
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(npu_runner, vllm_config)


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths.

    Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention state:
    batches within the preallocated budget (:attr:`max_tokens`, sized by ``full_mode``) run on xlite, others fall
    back to the native runnable.
    """

    def __init__(self, npu_runner: "XliteModelRunner", vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            npu_runner (XliteModelRunner): The native NPU model runner.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.npu_runner = npu_runner
        self.npu_runnable = npu_runner.fallback_model
        self.device = device
        # `full_mode` only sizes the token budget (see the comment in `StandardXliteModel._build_config`).
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        # EXPERIMENTAL (A1 prototype): decode-only mode for hybrid (GDN+MHA)
        # models. Prefill runs the native runnable (piecewise ACLGraph kept);
        # decode steps run the xlite graph; GDN states produced by native
        # prefill are imported into the xlite persistent pool at handoff.
        self._hybrid_decode_only: bool = os.environ.get("XLITE_DECODE_ONLY_HYBRID", "0") == "1"
        self._xlite_native_dirty: set[int] = set()
        self._xlite_req_pages: dict[int, list[int | None]] = {}
        self._xlite_native_gdn_groups: list[list[torch.Tensor]] = []
        self._xlite_import_log_done = False
        # A1 decode-only: native steps record their per-step state-index
        # tensor (device) here instead of syncing to CPU every step; the ids
        # are materialized into the dirty set once at the next import.
        self._pending_native_ids: list[torch.Tensor] = []
        # A1 decode-only: key ids (device) of the LAST xlite decode batch —
        # bounds the pre-native-step export to one batch worth of requests.
        self._last_decode_keys: torch.Tensor | None = None
        if self._hybrid_decode_only:
            logger.warning(
                "xlite EXPERIMENTAL decode-only hybrid mode enabled "
                "(XLITE_DECODE_ONLY_HYBRID=1): prefill runs the native "
                "runnable, decode runs the xlite graph; GDN states are "
                "imported from the vllm mamba pools. Use full_mode=false."
            )

        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
        self.xlite_model = self.adapter_xlite_model.xlite_model  # xlite C++ model instance
        self.freq_cis = self.adapter_xlite_model.rope_cossin_cache
        xlite_config = self.adapter_xlite_model.xlite_config
        self.xlite_rt = Runtime(
            devid=device.index,
            size=0,
            rank=torch.distributed.get_rank(),
            tp_size=xlite_config.def_tp_size,
            dp_size=xlite_config.def_dp_size,
            moe_tp_size=xlite_config.moe_tp_size,
            moe_ep_size=xlite_config.moe_ep_size,
        )
        # The xlite token budget: batches with more tokens (max across DP ranks) are routed to the native runnable.
        self.max_tokens = xlite_config.max_m
        self.hidden_states = torch.empty(
            self.max_tokens, xlite_config.hidden_size, device=self.device, dtype=self.adapter_xlite_model.dtype
        )

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if torch.distributed.get_rank() == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

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
            return getattr(self.npu_runnable, key)
        except Exception as e:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from e

    def register_kv_caches(self, kv_caches: list[tuple[torch.Tensor, ...]]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[tuple[torch.Tensor, ...]]): Runtime KV cache handles or tensors.
        """
        # `extract_kv_cache` is declared to return a list, but the actual shape
        # depends on the strategy (vLLM may hand us a dict keyed by layer name).
        adapted: Any = self.adapter_xlite_model.extract_kv_cache(kv_caches)
        if self.adapter_xlite_model.xlite_config.attn_type == AttnHybrid:
            ordered = list(adapted.values()) if isinstance(adapted, dict) else list(adapted)
            converted = [list(c) if isinstance(c, (list, tuple)) else [c] for c in ordered]
            self.kv_caches = self._adapt_hybrid_kv_caches(converted)
        else:
            self.kv_caches = adapted

    def _adapt_hybrid_kv_caches(self, caches: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Map hybrid caches to xlite: paged K/V for full layers, slot-major GDN states."""
        cfg = self.adapter_xlite_model.xlite_config
        n_layers = int(cfg.n_layers)
        max_batch, tp = int(cfg.max_batch_size), max(int(cfg.def_tp_size), 1)
        n_v = int(cfg.linear_num_v_heads) // tp
        k_dim, v_dim = int(cfg.linear_key_head_dim), int(cfg.linear_value_head_dim)
        conv_dim = (int(cfg.linear_num_k_heads) // tp) * k_dim * 2 + n_v * v_dim
        kernel, block_size, head_dim = int(cfg.linear_conv_kernel_dim), int(cfg.block_size), int(cfg.head_dim)
        kv_heads = max(int(cfg.n_kv_heads) // tp, 1)
        # Slot 0 is NULL_BLOCK_ID; sparse mamba IDs are remapped into [1, max_batch].
        num_slots = max_batch + 1
        self._xlite_hybrid_block_to_slot: dict[int, int] = {}
        self._xlite_hybrid_free_slots = list(range(1, num_slots))
        self._xlite_hybrid_persistent: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._xlite_hybrid_batch: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._xlite_hybrid_state_refs: list[torch.Tensor] = []
        self._xlite_hybrid_last_ids: tuple[int, ...] | None = None
        self._xlite_hybrid_last_dense: tuple[int, ...] | None = None
        self._xlite_hybrid_last_idx: torch.Tensor | None = None
        self._xlite_hybrid_dirty = False
        self._xlite_hybrid_skip = False
        adapted: list[list[torch.Tensor]] = []
        z = self.hidden_states.new_zeros

        def as_kv(group: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            ts = [t for t in group if isinstance(t, torch.Tensor)]
            if len(ts) == 1 and ts[0].ndim == 5 and ts[0].shape[0] == 2:
                k, v = ts[0][0], ts[0][1]
            elif len(ts) >= 2:
                k, v = ts[0], ts[1]
            else:
                raise RuntimeError(f"bad full-attn cache shapes {[tuple(t.shape) for t in ts]}")
            # Already kernel layout, or split manager page into kernel blocks.
            if k.shape[1] == block_size and k.shape[2:] == (kv_heads, head_dim):
                return k, v
            if (
                k.ndim == 4
                and k.shape[1] > block_size
                and k.shape[1] % block_size == 0
                and k.shape[2:] == (kv_heads, head_dim)
            ):
                chunk = k.shape[1] // block_size
                return (
                    k.view(k.shape[0] * chunk, block_size, kv_heads, head_dim),
                    v.view(v.shape[0] * chunk, block_size, kv_heads, head_dim),
                )
            raise RuntimeError(
                f"full-attn KV layout mismatch: {tuple(k.shape)}, expect [*,{block_size},{kv_heads},{head_dim}]"
            )

        # Layer roles must come from the same source as the weight bindings
        # (_full_attention_mask), not from the inferred interval — the two
        # disagree for non-canonical layer arrangements.
        role_mask = self._full_attention_mask
        if len(role_mask) != len(caches):
            raise RuntimeError(f"full-attention mask length {len(role_mask)} != cache layers {len(caches)}")
        for i in range(n_layers):
            if role_mask[i]:
                adapted.append(list(as_kv(caches[i])))
            else:
                # A1 decode-only: native per-layer mamba pools are resolved
                # lazily from the model's bound GDN modules
                # (_resolve_native_gdn_groups) — the runner's kv_caches
                # entries here are GROUP buffers shared by 3 GDN layers.
                conv_p, ssm_p = z(num_slots, conv_dim, kernel), z(num_slots, n_v, k_dim, v_dim)
                conv_b, ssm_b = z(max_batch, conv_dim, kernel), z(max_batch, n_v, k_dim, v_dim)
                self._xlite_hybrid_persistent.append((conv_p, ssm_p))
                self._xlite_hybrid_batch.append((conv_b, ssm_b))
                self._xlite_hybrid_state_refs.extend([conv_p, ssm_p, conv_b, ssm_b])
                adapted.append([conv_b, ssm_b])
        logger.info(
            "xlite hybrid KV adapted: layers=%s max_batch=%s gdn_slots=%s native_gdn_groups=%s caches_len=%s",
            n_layers,
            max_batch,
            num_slots,
            len(self._xlite_native_gdn_groups),
            len(caches),
        )

        return adapted

    @staticmethod
    def _pick_attn_metadata(attn_metadata: Any, expected_type: type | tuple[type, ...]) -> Any | None:
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
        if isinstance(attn_metadata, expected_type):
            return attn_metadata
        if isinstance(attn_metadata, dict):
            # Prefer full-attention AscendMetadata (hybrid also has GDN entries).
            cands = [m for m in attn_metadata.values() if isinstance(m, expected_type)]
            if not cands:
                return None
            if len(cands) == 1 or expected_type is not AscendMetadata:
                return cands[0]

            def score(m: Any) -> int:
                bt = getattr(m, "block_tables", None)
                return int(bt.shape[-1]) if bt is not None and hasattr(bt, "shape") else -1

            return max(cands, key=score)
        return None

    @staticmethod
    def _gdn_state_ids(attn_metadata_raw: Any, num_reqs: int) -> torch.Tensor | None:
        if num_reqs <= 0:
            return None
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        metas = (
            attn_metadata_raw.values()
            if isinstance(attn_metadata_raw, dict)
            else attn_metadata_raw
            if isinstance(attn_metadata_raw, list)
            else (attn_metadata_raw,)
        )
        for meta in metas:
            if not isinstance(meta, GDNAttentionMetadata):
                continue
            idx = meta.non_spec_state_indices_tensor
            if idx is None and meta.spec_state_indices_tensor is not None:
                idx = meta.spec_state_indices_tensor[:, 0]
            if idx is not None and idx.numel() >= num_reqs:
                return idx[:num_reqs]
        return None

    def _resolve_native_gdn_groups(self) -> None:
        """A1 decode-only: locate the native per-layer mamba state pools.

        The runner's kv_caches entries for GDN layers are GROUP buffers packed
        per 4-layer block (3 GDN layers share one buffer; data_ptr-verified),
        so they cannot be sliced per layer here. Instead read the bound
        per-layer views from the native model's GDN modules
        (``layer.linear_attn.kv_cache``), which the native path itself uses.
        """
        inner = self.npu_runnable
        if hasattr(inner, "unwrap"):
            inner = inner.unwrap()
        # candidate decoder containers: text decoder may sit at .model,
        # .language_model, or .model.language_model (VL-style wrappers)
        cands = []
        obj = inner
        for _ in range(4):
            if obj is None:
                break
            cands.append(obj)
            for attr in ("model", "language_model"):
                nxt = getattr(obj, attr, None)
                if nxt is not None:
                    cands.append(nxt)
            obj = getattr(obj, "language_model", None) or getattr(obj, "model", None)
        layers = None
        for c in cands:
            layers = getattr(c, "layers", None)
            if layers is not None:
                break
        if layers is None:
            raise RuntimeError(
                "xlite decode-only: cannot locate native model layers "
                f"(inner type={type(inner).__name__}, "
                f"candidates={[type(c).__name__ for c in cands]})"
            )
        groups = []
        prefixes = []
        for layer in layers:
            gdn = getattr(layer, "linear_attn", None)
            if gdn is not None and hasattr(gdn, "kv_cache"):
                groups.append([t for t in gdn.kv_cache if isinstance(t, torch.Tensor)])
                prefixes.append(getattr(gdn, "prefix", None))
        if not groups:
            raise RuntimeError("xlite decode-only: no native GDN layers found")
        self._xlite_native_gdn_groups = groups
        self._xlite_gdn_meta_keys = prefixes
        if os.environ.get("XLITE_DO_DEBUG", "0") == "1":
            logger.warning(
                "xlite DO-DEBUG: resolved ptrs=%s shapes=%s types=%s",
                [g[0].data_ptr() for g in groups[:6]],
                [tuple(g[0].shape) for g in groups[:6]],
                [type(g[0]).__name__ for g in groups[:2]],
            )
        logger.info("xlite decode-only: resolved %s native GDN layer pools", len(groups))

    def xlite_hybrid_prepare(self) -> None:
        """A1 decode-only: pre-resolve the native GDN pools and warm the D2H
        sync path at load time so the first real request does not pay the
        lazy-init costs (the first .cpu() of the process initializes the NPU
        synchronize context; measured as a ~118ms one-time TTFT penalty on
        the first benchmark run when left lazy)."""
        if not self._hybrid_decode_only:
            return
        try:
            self._resolve_native_gdn_groups()
        except RuntimeError as e:
            logger.warning("xlite decode-only: pool pre-resolve deferred: %s", e)
            return
        # warm the D2H synchronize path (first sync pays context init)
        _ = int(torch.zeros(1, device=self.device).sum().item())
        logger.info("xlite decode-only: warmed up (native pools resolved, sync path ready)")

    @staticmethod
    def _gdn_layer_indices(attn_metadata_raw: Any, key: str | None, num_reqs: int) -> list[int] | None:
        """Per-request state page ids of ONE GDN layer (its own metadata)."""
        if key is None or num_reqs <= 0:
            return None
        try:
            meta = attn_metadata_raw[key]
        except (KeyError, TypeError, IndexError):
            return None
        idx = getattr(meta, "non_spec_state_indices_tensor", None)
        if idx is None or idx.numel() < num_reqs:
            return None
        return [int(x) for x in idx[:num_reqs].detach().cpu().tolist()]

    def _hybrid_import_native_states(self, attn_metadata_raw: Any, layer0_idx: list[int] | None, num_reqs: int) -> None:
        """A1 decode-only: import native mamba-pool GDN states into the xlite
        persistent pool.

        The pool backing is SHARED across GDN layers; each layer addresses it
        with its OWN layer-scoped state indices (from its per-layer metadata),
        so the import resolves per-layer indices instead of reusing layer 0's.
        Conversion per GDN layer:
          conv: native [K-1, C] (SD) or [C, K-1] (DS)  -> xlite [C, K] cols 1..K-1
                (xlite window = [state[1:], input]; col 0 is oldest/scratch)
          ssm:  native [n_v, V, K] ("vllm convention") -> xlite [n_v, K, V]
        """
        if num_reqs <= 0:
            return
        if not self._xlite_native_gdn_groups:
            self._resolve_native_gdn_groups()
        # Materialize the state-index tensors recorded by native steps since
        # the last import (one sync per prefill->decode transition instead of
        # one per native step).
        if self._pending_native_ids:
            for _t in self._pending_native_ids:
                self._xlite_native_dirty.update(int(x) for x in _t.detach().cpu().tolist())
            self._pending_native_ids.clear()
        if len(self._xlite_native_gdn_groups) != len(self._xlite_hybrid_persistent):
            raise RuntimeError(
                f"xlite decode-only: native GDN group count "
                f"{len(self._xlite_native_gdn_groups)} != persistent pool count "
                f"{len(self._xlite_hybrid_persistent)}"
            )
        if not self._xlite_native_dirty:
            return
        key_ids = layer0_idx
        if key_ids is None:
            key_ids = self._gdn_layer_indices(
                attn_metadata_raw,
                self._xlite_gdn_meta_keys[0] if self._xlite_gdn_meta_keys else None,
                num_reqs,
            )
        if key_ids is None:
            return
        dirty_keys = [b for b in key_ids if b and b in self._xlite_native_dirty]
        if not dirty_keys:
            return
        slots = self._hybrid_slots(tuple(b for b in key_ids if b))
        first_log = not self._xlite_import_log_done
        imported = 0
        # PERF: materialize ALL layers' state indices with ONE device->host
        # sync. Previously each layer resolved its indices with its own
        # .cpu(), so every import event paid one full-stream sync per layer;
        # under KV-preemption storms the import runs constantly and those
        # syncs shatter the decode pipeline. Stack on-device, sync once,
        # slice per layer.
        layer_map: dict[int, list[int]] = {}
        _stacked: list[torch.Tensor] = []
        _stacked_li: list[int] = []
        for li in range(len(self._xlite_native_gdn_groups)):
            key = self._xlite_gdn_meta_keys[li] if li < len(self._xlite_gdn_meta_keys) else None
            if key is None:
                continue
            try:
                meta = attn_metadata_raw[key]
            except (KeyError, TypeError, IndexError):
                continue
            idx = getattr(meta, "non_spec_state_indices_tensor", None)
            if idx is None or idx.numel() < num_reqs:
                continue
            _stacked.append(idx[:num_reqs].to(torch.long))
            _stacked_li.append(li)
        if _stacked:
            _all = torch.stack(_stacked, dim=0).detach().cpu().tolist()  # ONE sync
            layer_map = {li: _all[j] for j, li in enumerate(_stacked_li)}
        for li, ((conv_p, ssm_p), group) in enumerate(
            zip(self._xlite_hybrid_persistent, self._xlite_native_gdn_groups)
        ):
            layer_ids = layer_map.get(li)
            if layer_ids is None:
                continue
            nconv, nssm = self._pick_native_conv_ssm(group, int(conv_p.shape[1]), int(ssm_p.shape[1]))
            conv_w = conv_p.shape[2] - 1
            for r, slot in enumerate(slots):
                if not slot or r >= len(layer_ids) or key_ids[r] not in self._xlite_native_dirty:
                    continue
                page = layer_ids[r]
                nc = nconv[page] if nconv.ndim == 3 else nconv  # [K-1, C] or [C, K-1]
                ns = nssm[page] if nssm.ndim == 4 else nssm  # [n_v, V, K]
                if nc.shape[0] == conv_w:
                    src = nc.t()  # [K-1, C] -> [C, K-1]
                elif nc.shape[1] == conv_w:
                    src = nc  # [C, K-1]
                else:
                    raise RuntimeError(
                        f"native conv state shape {tuple(nc.shape)} does not match xlite conv width {conv_w}"
                    )
                src = src.to(dtype=conv_p.dtype)
                conv_p[slot, :, 1:] = src
                conv_p[slot, :, 0] = src[:, 0]
                # native temporal state is [n_v, head_v_dim, head_k_dim]
                # ("vllm convention"); xlite pool is [n_v, k_dim, v_dim].
                ssm_p[slot] = ns.transpose(1, 2).to(dtype=ssm_p.dtype)
                imported += 1
                self._xlite_req_pages.setdefault(key_ids[r], [None] * len(self._xlite_native_gdn_groups))[li] = page
        self._xlite_native_dirty.difference_update(b for b in key_ids if b in self._xlite_native_dirty)
        # note: dirty keys are request-level (layer-0 scoped ids); clearing the
        # key clears ALL layers of that request at once
        if first_log:
            g0 = self._xlite_native_gdn_groups[0]
            logger.info(
                "xlite decode-only: imported states for %s requests x %s layers (native group[0] tensors: %s)",
                len(key_ids),
                imported // max(1, len(key_ids)),
                [(tuple(t.shape), str(t.dtype)) for t in g0],
            )
            self._xlite_import_log_done = True

    def _hybrid_export_to_native(self, key_ids: list[int] | None) -> None:
        """A1 decode-only: lazily write the xlite persistent states back to
        the native mamba pools right before a native step runs.

        BOUNDED: ``key_ids`` are the requests of the LAST xlite decode batch
        (cached as a device tensor by the decode path, materialized once per
        native<->decode transition — no per-step D2H). Export size never
        exceeds one batch regardless of how many requests the server has
        served; requests absent from the batch either have native-fresh
        states (new prefills) or no longer run (finished)."""
        if self._xlite_hybrid_dirty and self._xlite_hybrid_last_idx is not None:
            self._hybrid_copy(self._xlite_hybrid_last_idx, to_batch=False)
            self._xlite_hybrid_dirty = False
        if not key_ids or not self._xlite_native_gdn_groups:
            return
        for key in key_ids:
            if not key:
                continue
            slot = self._xlite_hybrid_block_to_slot.get(key)
            pages = self._xlite_req_pages.get(key)
            if not slot or not pages:
                continue
            for li, ((conv_p, ssm_p), group) in enumerate(
                zip(self._xlite_hybrid_persistent, self._xlite_native_gdn_groups)
            ):
                page = pages[li] if li < len(pages) else None
                if not page:
                    continue
                nconv, nssm = self._pick_native_conv_ssm(group, int(conv_p.shape[1]), int(ssm_p.shape[1]))
                if nconv.ndim == 3:
                    nconv[page] = conv_p[slot][:, 1:].t().to(nconv.dtype)
                else:
                    nconv[page] = conv_p[slot][:, 1:].to(nconv.dtype)
                if nssm.ndim == 4:
                    nssm[page] = ssm_p[slot].transpose(1, 2).to(nssm.dtype)
                else:
                    nssm[page] = ssm_p[slot].to(nssm.dtype)

    @staticmethod
    def _pick_native_conv_ssm(
        group: list[torch.Tensor],
        conv_dim: int,
        n_v: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identify conv / temporal state tensors inside a native mamba group.

        conv carries the conv_dim axis (2560/TP layout); temporal carries the
        n_v axis and no conv_dim.
        """
        ts = [t for t in group if isinstance(t, torch.Tensor)]
        conv = ssm = None
        for t in ts:
            last = t.shape[-2:] if t.ndim >= 2 else t.shape
            if conv is None and conv_dim in last:
                conv = t
            elif ssm is None and t.ndim >= 3 and t.shape[-3] == n_v and conv_dim not in t.shape[-2:]:
                ssm = t
        if conv is None or ssm is None:
            raise RuntimeError(
                "cannot identify conv/ssm in native mamba group: "
                f"{[(tuple(t.shape), str(t.dtype)) for t in ts]} "
                f"(conv_dim={conv_dim}, n_v={n_v})"
            )
        return conv, ssm

    def _hybrid_slots(self, bids: tuple[int, ...]) -> tuple[int, ...]:
        """Remap sparse mamba block IDs into dense slots; 0 is NULL."""
        active = {b for b in bids if b}
        out: list[int] = []
        for bid in bids:
            if not bid:
                out.append(0)
                continue
            slot = self._xlite_hybrid_block_to_slot.get(bid)
            if slot is None:
                if not self._xlite_hybrid_free_slots:
                    for old_bid, old_slot in list(self._xlite_hybrid_block_to_slot.items()):
                        if old_bid not in active:
                            del self._xlite_hybrid_block_to_slot[old_bid]
                            self._xlite_hybrid_free_slots.append(old_slot)
                            break
                if not self._xlite_hybrid_free_slots:
                    raise RuntimeError(f"xlite hybrid: no free GDN slots (active={len(active)})")
                slot = self._xlite_hybrid_free_slots.pop()
                self._xlite_hybrid_block_to_slot[bid] = slot
                for conv_p, ssm_p in self._xlite_hybrid_persistent:
                    conv_p[slot].zero_()
                    ssm_p[slot].zero_()
            out.append(slot)
        return tuple(out)

    def _hybrid_copy(self, idx: torch.Tensor, to_batch: bool) -> None:
        n = int(idx.numel())
        for (conv_p, ssm_p), (conv_b, ssm_b) in zip(self._xlite_hybrid_persistent, self._xlite_hybrid_batch):
            if to_batch:
                conv_b[:n].copy_(conv_p.index_select(0, idx))
                ssm_b[:n].copy_(ssm_p.index_select(0, idx))
            else:
                conv_p.index_copy_(0, idx, conv_b[:n])
                ssm_p.index_copy_(0, idx, ssm_b[:n])

    def _hybrid_zero_rows(self, rows: list[int], dense: tuple[int, ...]) -> None:
        for (conv_p, ssm_p), (conv_b, ssm_b) in zip(self._xlite_hybrid_persistent, self._xlite_hybrid_batch):
            for i in rows:
                conv_b[i].zero_()
                ssm_b[i].zero_()
                if dense[i]:
                    conv_p[dense[i]].zero_()
                    ssm_p[dense[i]].zero_()

    def _hybrid_gather(self, raw_idx: torch.Tensor | None, cached_lens: list[int], num_reqs: int) -> None:
        """Gather GDN states; skip copies when state_indices order is unchanged."""
        self._xlite_hybrid_skip = False
        if num_reqs <= 0 or not getattr(self, "_xlite_hybrid_persistent", None):
            self._xlite_hybrid_skip = True
            return
        bids = (
            tuple(range(1, num_reqs + 1))
            if raw_idx is None
            else tuple(int(x) for x in raw_idx[:num_reqs].detach().cpu().tolist())
        )
        fresh = [i for i, length in enumerate(cached_lens[:num_reqs]) if int(length) == 0 and bids[i]]
        last_idx = self._xlite_hybrid_last_idx
        last_dense = self._xlite_hybrid_last_dense
        if self._xlite_hybrid_last_ids == bids and last_idx is not None and last_dense is not None:
            if fresh:
                self._hybrid_zero_rows(fresh, last_dense)
            self._xlite_hybrid_skip = True
            return
        if self._xlite_hybrid_dirty and last_idx is not None:
            self._hybrid_copy(last_idx, to_batch=False)
            self._xlite_hybrid_dirty = False
        dense = bids if raw_idx is None else self._hybrid_slots(bids)
        idx = (
            last_idx
            if last_dense == dense and last_idx is not None
            else torch.tensor(dense, dtype=torch.long, device=self.device)
        )
        self._hybrid_copy(idx, to_batch=True)
        if fresh:
            self._hybrid_zero_rows(fresh, dense)
        self._xlite_hybrid_last_ids = bids
        self._xlite_hybrid_last_dense = dense
        self._xlite_hybrid_last_idx = idx

    def _hybrid_scatter(self) -> None:
        if self._xlite_hybrid_skip:
            self._xlite_hybrid_dirty = True
            return
        idx = self._xlite_hybrid_last_idx
        if idx is not None:
            self._hybrid_copy(idx, to_batch=False)
            self._xlite_hybrid_dirty = False

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor | None): Token IDs for current step.

            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (IntermediateTensors | None): Optional intermediate tensors from pipeline stages.
            inputs_embeds (torch.Tensor | None): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).
            **model_kwargs (Any): Additional keyword arguments for the runnable.

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        is_hybrid = self.adapter_xlite_model.xlite_config.attn_type == AttnHybrid

        def _ensure_output(out: Any, path: str) -> XliteForwardResult:
            if out is None:
                raise RuntimeError(
                    f"XliteWrapper got None from {path} "
                    f"(full_mode={self.full_mode}, hybrid={is_hybrid}, "
                    f"runnable={type(self.npu_runnable).__name__})"
                )
            return out

        if getattr(forward_context, "in_profile_run", False) or (
            is_hybrid and getattr(forward_context, "capturing", False)
        ):
            # full_mode hybrid: never capture/run Ascend-GDN runnable here — state is not
            # sync-compatible with xlite, and ACLGraph weak-refs of runnable output can
            # become None on replay.
            if self.full_mode and (
                getattr(forward_context, "in_profile_run", False)
                or (is_hybrid and getattr(forward_context, "capturing", False))
            ):
                # NOTE: returning a single hidden state tensor may break the vLLM pipeline if the runnable expects a
                # tuple of outputs, e.g., (hidden_states, aux_hidden_states) under certain speculative scenarios
                return _ensure_output(self.hidden_states, "full_mode profile/capture placeholder")
            return _ensure_output(
                self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                "runnable(profile/capturing)",
            )

        attn_metadata_raw: Any = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return _ensure_output(
                self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                "runnable(attn_metadata=None)",
            )

        if is_hybrid:
            attn_metadata = self._pick_attn_metadata(attn_metadata_raw, self.adapter_xlite_model._attn_metadata_type)
        else:
            attn_metadata = attn_metadata_raw[0] if isinstance(attn_metadata_raw, list) else attn_metadata_raw
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(
                    "model.layers.0.self_attn.attn", next(iter(attn_metadata.values()), None)
                )
            if not isinstance(attn_metadata, self.adapter_xlite_model._attn_metadata_type):
                attn_metadata = None
        if attn_metadata is None:
            return _ensure_output(
                self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                "runnable(attn_metadata type mismatch)",
            )

        with_prefill = attn_metadata.attn_state not in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )

        # Since upstream v0.28.0, non-hybrid batches are routed by token count
        # (max across DP ranks) instead of the batch attention state: batches
        # beyond the preallocated budget fall back to the native runnable.
        if not is_hybrid:
            num_tokens = getattr(forward_context, "max_tokens_across_dp", None)
            if num_tokens is not None and num_tokens > self.max_tokens:
                if self.full_mode:
                    logger.warning_once(
                        "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                        num_tokens,
                        self.max_tokens,
                        "https://atomgit.com/openeuler/GVirt",
                    )
                return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        # Full: graph for prefill and decode
        # Decode-Only: runnable for prefill, graph for decode
        # Hybrid: always graph (Ascend-GDN state is not sync-compatible with xlite).
        if is_hybrid:
            use_xlite_graph = True
        elif not self.full_mode and self.data_parallel_size > 1:
            num_tokens = forward_context.batch_descriptor.num_tokens
            num_reqs = forward_context.batch_descriptor.num_reqs
            use_xlite_graph = num_reqs is not None and num_tokens <= num_reqs
        else:
            use_xlite_graph = not with_prefill or self.full_mode

        if not use_xlite_graph:
            # fall back to runnable for prefill in decode-only mode
            # or when the number of tokens exceeds the graph capacity in non-full mode
            return _ensure_output(
                self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                "runnable(not use_xlite_graph)",
            )

        if is_hybrid:
            num_reqs = int(getattr(attn_metadata, "num_decodes", 0) or 0) + int(
                getattr(attn_metadata, "num_prefills", 0) or 0
            )
            if num_reqs <= 0:
                num_reqs = int(getattr(getattr(forward_context, "batch_descriptor", None), "num_reqs", 0) or 0)
            seq_lens_list = list(getattr(attn_metadata, "seq_lens_list", None) or [])[:num_reqs]
            actual_q = list(getattr(attn_metadata, "actual_seq_lengths_q", None) or [])
            if len(actual_q) >= num_reqs > 0:
                cum = [0] + [int(x) for x in actual_q[:num_reqs]]
                query_lens_list = [cum[i + 1] - cum[i] for i in range(num_reqs)]
            else:
                router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
                seq = router.seq_lens[:num_reqs]
                cum_t = router.cu_query_lens[-seq.size(0) :][: num_reqs + 1]
                if cum_t.device.type != "cpu":
                    cum_t = cum_t.to("cpu")
                query_lens_list = torch.diff(cum_t, prepend=cum_t.new_zeros(1)).tolist()
                if not seq_lens_list:
                    seq_lens_list = seq.tolist()
            query_lens_list = [int(x) for x in query_lens_list[:num_reqs]]
            seq_lens_list = [int(x) for x in seq_lens_list[:num_reqs]]
            if len(seq_lens_list) < num_reqs or len(query_lens_list) < num_reqs:
                # Falling back here means the native GDN layers would resume from
                # state this request never produced on xlite (the recurrent state
                # is not transferred across backends in full-hybrid mode) — the
                # response quality for this request is not guaranteed. Full state
                # bridging lands with the decode-only pool work.
                logger.error(
                    "xlite hybrid: metadata lens incomplete (reqs=%s seq=%s query=%s); "
                    "falling back to native with unbridged GDN state — output "
                    "quality for in-flight requests is not guaranteed",
                    num_reqs,
                    len(seq_lens_list),
                    len(query_lens_list),
                )
                return _ensure_output(
                    self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                    "runnable(hybrid lens mismatch)",
                )
            cached_lens_list = [max(seq_lens_list[i] - query_lens_list[i], 0) for i in range(num_reqs)]
            if input_ids is None:
                return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
            bt = getattr(attn_metadata, "block_tables", None)
            if bt is None:
                return _ensure_output(
                    self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                    "runnable(hybrid block_tables=None)",
                )
            block_tables_list = (
                bt[:num_reqs].detach().to("cpu").tolist()
                if hasattr(bt, "device") and bt.device.type != "cpu"
                else bt[:num_reqs].tolist()
            )
            num_actual_tokens = int(attn_metadata.num_actual_tokens)
            if sum(query_lens_list) != num_actual_tokens:
                return _ensure_output(
                    self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                    "runnable(hybrid query_lens sum mismatch)",
                )
            # Refuse capture/dummy all-zero tables on decode.
            for i, cached in enumerate(cached_lens_list):
                if cached > 0 and (not block_tables_list[i] or int(block_tables_list[i][0]) == 0):
                    return _ensure_output(
                        self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs),
                        "runnable(hybrid dummy block_table)",
                    )
            # slot-major state management: gather from the persistent pool;
            # decode-only mode additionally imports/exports native state here
            _gdn_idx = self._gdn_state_ids(attn_metadata_raw, num_reqs)
            if self._hybrid_decode_only:
                # PERF: materialize the layer-0 ids only when the dirty/
                # pending book-keeping actually needs them — an unconditional
                # .cpu() here would sync the stream on EVERY decode step.
                if self._xlite_native_dirty or self._pending_native_ids:
                    _layer0_ids = None
                    if _gdn_idx is not None:
                        _layer0_ids = [int(x) for x in _gdn_idx[:num_reqs].detach().cpu().tolist()]
                    self._hybrid_import_native_states(attn_metadata_raw, _layer0_ids, num_reqs)
                if os.environ.get("XLITE_DO_DEBUG", "0") == "1":
                    self._dbg_dec_steps = getattr(self, "_dbg_dec_steps", 0) + 1
                    _dbg_ids = (
                        [int(x) for x in _gdn_idx[:num_reqs].detach().cpu().tolist()] if _gdn_idx is not None else None
                    )
                    logger.warning(
                        "xlite DO-STEP: decode step #%s (num_reqs=%s, ids=%s)",
                        self._dbg_dec_steps,
                        num_reqs,
                        None if _dbg_ids is None else _dbg_ids[:6],
                    )
                # Remember this decode batch's key ids (device, no sync): the
                # export before the next native step covers exactly these —
                # every running decode-phase request appears in every decode
                # step, so the last batch covers all of them.
                if _gdn_idx is not None:
                    self._last_decode_keys = _gdn_idx[:num_reqs]
            if os.environ.get("XLITE_DO_DEBUG", "0") == "1" and _gdn_idx is not None:
                _ids_l = [int(x) for x in _gdn_idx[:num_reqs].detach().cpu().tolist()]
                for _i, _b in enumerate(_ids_l):
                    if not _b:
                        continue
                    _slot = self._xlite_hybrid_block_to_slot.get(_b, 0)
                    if not _slot:
                        continue
                    _cs = float(self._xlite_hybrid_persistent[0][0][_slot].float().sum())
                    _ss = float(self._xlite_hybrid_persistent[0][1][_slot].float().sum())
                    _cols = [
                        float(self._xlite_hybrid_persistent[0][0][_slot, :, c].float().sum())
                        for c in range(self._xlite_hybrid_persistent[0][0].shape[2])
                    ]
                    logger.warning(
                        "xlite DO-STATE: bid=%s slot=%s L0 conv_sum=%.6f ssm_sum=%.6f conv_cols=%s",
                        _b,
                        _slot,
                        _cs,
                        _ss,
                        [round(v, 4) for v in _cols],
                    )
                    if not getattr(self, "_dbg_alllayer_logged", False):
                        _ss_all = [round(float(p[1][_slot].float().sum()), 4) for p in self._xlite_hybrid_persistent]
                        _cv_all = [round(float(p[0][_slot].float().sum()), 4) for p in self._xlite_hybrid_persistent]
                        logger.warning("xlite DO-STATE-ALL: ssm=%s", _ss_all)
                        logger.warning("xlite DO-STATE-ALL: conv=%s", _cv_all)
                        self._dbg_alllayer_logged = True
                    break
            elif os.environ.get("XLITE_DO_DEBUG", "0") == "1" and not getattr(self, "_dbg_imp_logged", False):
                logger.warning(
                    "xlite DO-DEBUG: decode step skip import (dirty=%s, ids=%s)",
                    len(self._xlite_native_dirty),
                    None if _gdn_idx is None else _gdn_idx[:6].tolist(),
                )
                if getattr(self, "_dbg_imp_logged_n", 0) >= 2:
                    self._dbg_imp_logged = True
                self._dbg_imp_logged_n = getattr(self, "_dbg_imp_logged_n", 0) + 1
            self._hybrid_gather(_gdn_idx, cached_lens_list, num_reqs)
            stream = torch.npu.current_stream().npu_stream
            xlite_attn_metadata = AttnMeta()
            xlite_attn_metadata.lens = query_lens_list
            xlite_attn_metadata.cached_lens = cached_lens_list
            xlite_attn_metadata.block_tables_cpu = block_tables_list
            pos = positions[0] if positions.ndim == 2 else positions
            xlite_attn_metadata.positions = pos[:num_actual_tokens].contiguous()
            num_tokens = (
                getattr(forward_context, "max_tokens_across_dp", None) or forward_context.batch_descriptor.num_tokens
            )
            h = self.hidden_states[:num_tokens]
            stream = torch.npu.current_stream().npu_stream
            if inputs_embeds is None:
                # Text path requires token ids; VL batches arrive as
                # inputs_embeds (input_ids is None) and take the branch below.
                if input_ids is None:
                    return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
                self.xlite_model.forward(
                    self.xlite_rt,
                    input_ids[:num_actual_tokens],
                    xlite_attn_metadata,
                    self.kv_caches,
                    self.freq_cis,
                    h,
                    stream,
                )
            else:
                emb = inputs_embeds[:num_actual_tokens]
                deepstack = getattr(self.npu_runnable, "deepstack_input_embeds", [])
                xlite_deepstack = [d[: emb.size(0)] for d in deepstack]
                self.xlite_model.forward_with_inputs_embeds(
                    self.xlite_rt, emb, xlite_attn_metadata, self.kv_caches, self.freq_cis, h, stream, xlite_deepstack
                )
                if xlite_deepstack and hasattr(self.npu_runnable, "_clear_deepstack_input_embeds"):
                    self.npu_runnable._clear_deepstack_input_embeds(emb.size(0))
            self._hybrid_scatter()
            xlite_attn_metadata = AttnMeta()
            xlite_attn_metadata.lens = query_lens_list
            xlite_attn_metadata.cached_lens = cached_lens_list
            xlite_attn_metadata.block_tables_cpu = block_tables_list
            pos = positions[0] if positions.ndim == 2 else positions
            xlite_attn_metadata.positions = pos[:num_actual_tokens].contiguous()
            num_tokens = (
                getattr(forward_context, "max_tokens_across_dp", None) or forward_context.batch_descriptor.num_tokens
            )
            h = self.hidden_states[:num_tokens]
            stream = torch.npu.current_stream().npu_stream
            if inputs_embeds is None:
                # Text path requires token ids; VL batches arrive as
                # inputs_embeds (input_ids is None) and take the branch below.
                if input_ids is None:
                    return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
                self.xlite_model.forward(
                    self.xlite_rt,
                    input_ids[:num_actual_tokens],
                    xlite_attn_metadata,
                    self.kv_caches,
                    self.freq_cis,
                    h,
                    stream,
                )
            else:
                emb = inputs_embeds[:num_actual_tokens]
                deepstack = getattr(self.npu_runnable, "deepstack_input_embeds", [])
                xlite_deepstack = [d[: emb.size(0)] for d in deepstack]
                self.xlite_model.forward_with_inputs_embeds(
                    self.xlite_rt, emb, xlite_attn_metadata, self.kv_caches, self.freq_cis, h, stream, xlite_deepstack
                )
                if xlite_deepstack and hasattr(self.npu_runnable, "_clear_deepstack_input_embeds"):
                    self.npu_runnable._clear_deepstack_input_embeds(emb.size(0))
            return _ensure_output(h[:num_actual_tokens], "xlite hybrid forward")

        attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        num_actual_tokens = attn_metadata_router.num_actual_tokens
        seq_lens = attn_metadata_router.seq_lens
        cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :]
        query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
        cached_lens = torch.clamp(seq_lens - query_lens, min=0)
        block_tables = attn_metadata_router.block_tables
        num_reqs = max(1, (block_tables[:, 0] != 0).sum().item())  # number of non-empty requests

        xlite_attn_metadata = AttnMeta()
        xlite_attn_metadata.lens = query_lens[:num_reqs].tolist()
        xlite_attn_metadata.cached_lens = cached_lens[:num_reqs].tolist()
        xlite_attn_metadata.block_tables_cpu = block_tables[:num_reqs].tolist()
        xlite_attn_metadata.positions = torch.atleast_2d(positions)[:, :num_actual_tokens].contiguous()

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        stream = torch.npu.current_stream().npu_stream
        input_ids = cast(torch.Tensor, input_ids)
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            self.xlite_model.forward(
                rt=self.xlite_rt,
                input=input_ids[:num_actual_tokens],
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_actual_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            self.xlite_model.forward_with_inputs_embeds(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if xlite_deepstack_input_embeds and hasattr(self.npu_runnable, "_clear_deepstack_input_embeds"):
                self.npu_runnable._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return _ensure_output(h[:num_actual_tokens], "xlite forward")
