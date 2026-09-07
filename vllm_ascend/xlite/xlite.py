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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import AttnDSA, AttnMeta, AttnMHA, AttnMLA, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
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
        xlite_config (XModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (XModel): Native xlite model container populated by subclasses.
        cossin_cache (torch.Tensor): Precomputed RoPE frequency cache tensor for rotary positional embeddings.
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
            npu_runnable (nn.Module): The fallback model instance used by the :data:`npu_runner` (vLLM-ascend native).
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner = npu_runner
        self.npu_runnable = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()
        self._build_model_config()
        self._build_model()
        self.xlite_model.init(self.xlite_config, torch.distributed.get_rank())
        self.cossin_cache = self._precompute_freqs_cis()

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
        xlite_config.rope_head_dim = xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = hf_config.intermediate_size
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
        xlite_config.max_m = (
            math.ceil(vllm_config.scheduler_config.max_num_batched_tokens / tp_size) * tp_size
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
        original_max_len = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)
        if xlite_config.max_seq_len > original_max_len and "mscale" in rope_params and "factor" in rope_params:
            mscale: float = 1.0 + 0.1 * rope_params["mscale"] * math.log(rope_params["factor"])  # type: ignore[operator,arg-type]
            xlite_config.softmax_scale *= mscale**2

        # MoE configuration (from Glm4MoeXliteModel, adapted for DeepseekV3)
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model = self.xlite_model
        layers, _ = self._get_layers_and_model_prefix()

        # MLA attention weights
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight")
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight")
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV")
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T")

        if not self.quantization:
            return

        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(wt_lst[0])
        self.xlite_config.quant_attn_weight_transpose = True
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mla_q_norm_bias = get_layer_weights(layers, "self_attn.q_a_layernorm.bias")
            xlite_model.mla_kv_norm_bias = get_layer_weights(layers, "self_attn.kv_a_layernorm.bias")

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute Yarn-style RoPE frequency cache for DeepseekV3 MLA attention.

        Returns complex exponential tensor for rotary positional embeddings.
        Format: [max_seq_len, rope_head_dim//2] complex tensor (torch.polar).
        """
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # Extract Yarn parameters from rope_parameters
        rope_params = getattr(hf_config, "rope_parameters", {})
        base = rope_params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))
        factor = rope_params.get("factor", 1.0)
        original_seq_len = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)
        beta_fast = rope_params.get("beta_fast", 32)
        beta_slow = rope_params.get("beta_slow", 1)

        dim = xlite_config.rope_head_dim  # qk_rope_head_dim (64 for DeepseekV3)
        seqlen = xlite_config.max_seq_len

        # Helper functions for Yarn frequency correction
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
            return torch.clamp(linear_func, 0, 1)

        # Compute base frequencies on CPU
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))

        # Apply Yarn scaling if sequence length exceeds original
        if seqlen > original_seq_len:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        # Create position indices and compute outer product
        t = torch.arange(seqlen, dtype=torch.float32, device="cpu")
        freqs = torch.outer(t, freqs)

        # Return complex exponential format (as expected by xlite MLA forward)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis.to(device="npu")


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
        # NOTE: MTP layers are not supported in the current implementation
        indexer_caches: list[tuple[torch.Tensor, ...]] = []
        mla_caches: list[tuple[torch.Tensor, ...]] = []

        idx = 0
        index_mask = self.xlite_config.index_full_mask or [True] * self.xlite_config.n_layers
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
        xlite_config.gate_captured = False


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
    """A graph-based wrapper that dispatches between xlite and runnable paths."""

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
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
        self.xlite_model = self.adapter_xlite_model.xlite_model  # xlite C++ model instance
        self.freq_cis = self.adapter_xlite_model.cossin_cache
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
        self.hidden_states = torch.empty(
            xlite_config.max_m, xlite_config.hidden_size, device=self.device, dtype=self.adapter_xlite_model.dtype
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
        except Exception:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from None

    def register_kv_caches(self, kv_caches: list[tuple[torch.Tensor, ...]]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[tuple[torch.Tensor, ...]]): Runtime KV cache handles or tensors.
        """
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)

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
            intermediate_tensors (IntermediateTensors | None): Optional intermediate tensors from pipeline stages.
            inputs_embeds (torch.Tensor | None): Optional external input embeddings (e.g. multimodal/deepstack
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
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata: Any = forward_context.attn_metadata
        if attn_metadata is None:
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        attn_metadata = attn_metadata.get("model.layers.0.self_attn.attn", next(iter(attn_metadata.values()), None))
        if not isinstance(attn_metadata, self.adapter_xlite_model._attn_metadata_type):
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

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
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        seq_lens = attn_metadata_router.seq_lens
        cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :]
        query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
        cached_lens = torch.clamp(seq_lens - query_lens, min=0)

        num_actual_tokens = attn_metadata_router.num_actual_tokens
        num_tokens = forward_context.max_tokens_across_dp

        xlite_attn_metadata = AttnMeta()
        xlite_attn_metadata.lens = query_lens.tolist()
        xlite_attn_metadata.cached_lens = cached_lens.tolist()
        xlite_attn_metadata.block_tables_cpu = attn_metadata_router.block_tables.tolist()
        if positions.ndim == 2:
            xlite_attn_metadata.positions = positions[:, :num_actual_tokens].contiguous()
            positions = positions[0]
        else:
            xlite_attn_metadata.positions = positions

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        stream = torch.npu.current_stream().npu_stream
        if inputs_embeds is None:
            self.xlite_model.forward(
                self.xlite_rt, input_ids, xlite_attn_metadata, self.kv_caches, self.freq_cis, h, stream
            )
        else:
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
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
            if xlite_deepstack_input_embeds and hasattr(self.npu_runnable, "_clear_deepstack_input_embeds"):
                self.npu_runnable._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return h[:num_actual_tokens]
