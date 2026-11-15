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

import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
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
from xlite._C import AttnDSA, AttnMetaV2, AttnMHA, AttnMLA, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.xlite.utils import (
    _DUMMY_TENSOR,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    get_dotted_attr,
    get_layer_weights,
    is_dummy_tensor,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerV2

    XliteModelRunner: TypeAlias = XliteModelRunnerV2

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]

_architecture_strategy_map: dict[str, type[XliteModelBase]] = {}
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

    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map."""
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

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            npu_runner (XliteModelRunner): The NPU model runner instance.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = self.npu_runner.device
        self.dtype = vllm_config.model_config.dtype
        # full mode: xlite handle all mixed and decode batches; otherwise, xlite only handle decode batches (roughly)
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
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

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]

    @abstractmethod
    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Extract xlite-compatible KV cache from the vLLM-ascend KV cache.

        Args:
            kv_caches (list[torch.Tensor]): The KV cache from the upstream vLLM/vLLM-ascend.

        Returns:
            list[tuple[torch.Tensor, ...]]: The KV cache in the format expected by xlite model implementations.
        """

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
        format = torch_npu.get_npu_format(t)  # type: ignore[attr-defined]
        return format == torch_npu.Format.FRACTAL_NZ  # type: ignore[attr-defined]

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
        if deq_scale.dtype == torch.int64:
            return deq_scale
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
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        # configure runtime token budgets between xlite full/decode-only modes with respect to speculative decoding
        # NOTE: for profiling runs (e.g., graph capturing), the following configurations might need further adjustments
        scheduler_config, spec_config = vllm_config.scheduler_config, vllm_config.speculative_config
        max_num_active_seqs = getattr(scheduler_config, "max_num_active_seqs", None) or scheduler_config.max_num_seqs
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.num_spec_tokens = num_spec_tokens = spec_config.num_speculative_tokens if spec_config else 0
        draft_slots = spec_config.max_num_new_slots_for_drafting if spec_config else 0
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens or max_num_batched_tokens
        max_num_scheduled_tokens = min(max_num_scheduled_tokens, max_num_batched_tokens)
        if get_ascend_config().xlite_graph_config.full_mode:
            max_m = max_num_batched_tokens  # all request must confine to this budget per forward pass
        else:
            # xlite only handles decode batches - drafter's additional slots consume the same scheduler input budget
            max_m = max(
                min(n * (num_spec_tokens + 1), max_num_scheduled_tokens, max_num_batched_tokens - n * draft_slots)
                for n in range(1, max_num_active_seqs + 1)
            )
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

    def _build_model(self) -> None:
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
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

        xlite_model.re_up_gate_scale_bias = get_layer_weights(layers, f"{re_prefix}.w13_scale_bias", **re_kwargs)
        xlite_model.re_down_scale_bias = get_layer_weights(layers, f"{re_prefix}.w2_scale_bias", **re_kwargs)

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
        # DSA's indexer has shared layers (no own weights), so `mask` pads them with `_DUMMY_TENSOR`; the list is
        # thus never empty and a naive `len() > 0` check would misclassify dynamic-quant matmuls as static. Static
        # quant is when `aclnn_input_scale_reciprocal` exists and at least one entry is a real (non-dummy) tensor.
        input_scale = get_layer_weights(layers, f"{model_prefix}.aclnn_input_scale_reciprocal", mask=mask)
        is_static = any(not is_dummy_tensor(t) for t in input_scale)
        if is_static:
            setattr(xlite_model, f"{xlite_prefix}_input_scale", input_scale)
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
            deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
        else:
            weight_scale = get_layer_weights(layers, f"{model_prefix}.weight_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove the type conversion once `xlite` accepts and defaults the cache in FP32 format
        return super()._precompute_rope_cache().to(dtype=self.dtype, device=self.device)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        kv_caches = kv_caches[: 2 * self.xlite_config.n_layers]
        if wrong_dims := set(cache.shape[-1] for cache in kv_caches) - set([self.xlite_config.head_dim]):
            raise ValueError(f"xlite: kv_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        return [(k_cache, v_cache) for k_cache, v_cache in zip(kv_caches[0::2], kv_caches[1::2])]


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

    _supported_architectures = ["DeepseekV3ForCausalLM", "Glm4MoeLiteForCausalLM"]

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

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        if self.xlite_config.kv_lora_rank == self.xlite_config.rope_head_dim:
            raise RuntimeError("xlite does not support `kv_lora_rank == rope_head_dim` for MLA attention.")
        n_layers = self.xlite_config.n_layers
        kv_nope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.kv_lora_rank, kv_caches))[:n_layers]
        k_rope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.rope_head_dim, kv_caches))[:n_layers]
        if (n_nope := len(kv_nope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA KV nope caches ({n_nope} vs. {n_layers})")
        if (n_rope := len(k_rope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA K rope caches ({n_rope} vs. {n_layers})")
        return [(k_nope, k_rope) for k_nope, k_rope in zip(kv_nope_caches, k_rope_caches)]


class DeepseekV32XliteModel(DeepseekV3XliteModel):
    """xlite adapter for Deepseek-V3.2/GLM-5.x architectures with Deepseek sparse attention (DSA)."""

    _supported_architectures = ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnDSA
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        index_types: list[str] = list(map(lambda x: str(x).lower(), getattr(hf_config, "indexer_types", []) or []))
        xlite_config.index_full_mask = list(map(lambda x: x.startswith("full"), index_types))[: xlite_config.n_layers]

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

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        # For DSA, `kv_caches` is passed as [<index_k_cache (optional)>, kv_nope_cache, k_pe_cache, ..., <mtp_layer>].
        # Under MRV2, all `index_k_cache` are passed first; under V1, they are interleaved with the kv caches.
        # For xlite, the `kv_caches` is expected as [(kv_nope_cache, k_pe_cache, index_k_cache), ...] for each layer.
        # TODO: consider the compatibility with `enable_sparse_sfa_c8` and `enable_sparse_li_c8`
        xcfg = self.xlite_config
        if xcfg.index_head_dim == xcfg.kv_lora_rank or xcfg.index_head_dim == xcfg.rope_head_dim:
            raise RuntimeError("xlite does not support `index_head_dim == kv_lora_rank or rope_head_dim` for DSA.")
        index_mask: list[bool] = (getattr(xcfg, "index_full_mask", None) or [True] * xcfg.n_layers)[: xcfg.n_layers]

        mla_kv_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] != xcfg.index_head_dim, kv_caches))
        mla_caches: list[tuple[torch.Tensor, ...]] = super().extract_kv_cache(mla_kv_caches)
        index_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] == xcfg.index_head_dim, kv_caches))
        if (n_index_layers := sum(index_mask)) != len(index_caches := index_caches[:n_index_layers]):
            raise ValueError(f"xlite: index_k_cache count mismatch ({len(index_caches)} vs. {n_index_layers})")
        if wrong_dims := set(cache.shape[-1] for cache in index_caches) - set([xcfg.index_head_dim]):
            raise ValueError(f"xlite: index_k_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        for i, mask in enumerate(index_mask):
            if not mask:
                index_caches.insert(i, _DUMMY_TENSOR)
        return [(mla_cache[0], mla_cache[1], i_cache) for mla_cache, i_cache in zip(mla_caches, index_caches)]


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


def get_adapter_xlite_model(npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> XliteModelBase:
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

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            npu_runner (XliteModelRunner): The native NPU model runner.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.device = device
        # `full_mode` only sizes the token budget (see the comment in `StandardXliteModel._build_config`).
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size
        self._is_dummy_batch: bool = False

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
        # Persistent one-token batch for DP dummy runs (`__call__`): constant, allocated once.
        self._dummy_attn_metadata = self._init_dummy_metadata(xlite_config)

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if torch.distributed.get_rank() == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]
        self.adapter_xlite_model.npu_runner = None  # type: ignore[assignment]
        self.adapter_xlite_model.xlite_model = None  # type: ignore[assignment]

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

    def register_kv_caches(self, kv_caches: list[torch.Tensor]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[torch.Tensor]): Runtime KV cache handles or tensors.
        """
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)

    @contextmanager
    def dummy_batch(self) -> Generator[None, None, None]:
        """Mark the enclosed run as a DP dummy batch.

        Idle DP ranks run ``NPUWorker.execute_dummy_batch`` through the xlite forward to keep the DP collectives
        aligned. xlite only requires such a rank to contribute a minimum of one token for one request, so while
        the flag is set the wrapper clamps every request-derived quantity to exactly that. The DP-agreed output
        buffer size is unaffected (see `XliteWrapper.__call__`).
        """
        self._is_dummy_batch = True
        try:
            yield
        finally:
            self._is_dummy_batch = False

    def _init_dummy_metadata(self, xlite_config: Any) -> AttnMetaV2:
        """Build the constant one-token batch used for DP dummy runs (`dummy_batch()`).

        Idle DP ranks run ``NPUWorker.execute_dummy_batch`` through the xlite forward to keep the DP
        collectives aligned; xlite only requires such a rank to contribute one token for one request.
        Its KV write lands on block 0 / slot 0 — not provably free; xlite's kernels have no
        negative-slot skip yet.
        """
        dummy = AttnMetaV2()
        dummy.lens_cpu = [1]
        dummy.cached_lens_cpu = [0]
        dummy.lens = torch.ones(1, dtype=torch.int32, device=self.device)
        dummy.cached_lens = torch.zeros(1, dtype=torch.int32, device=self.device)
        dummy.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        pos_dim = 3 if xlite_config.mrope_section else 1
        dummy.positions = torch.zeros((pos_dim, 1), dtype=torch.int64, device=self.device)
        dummy.slot_mapping = [torch.zeros(1, dtype=torch.int32, device=self.device)]
        dummy.block_tables = [torch.zeros((1, 1), dtype=torch.int32, device=self.device)]
        return dummy

    def _common_attn_metadata(self) -> AscendCommonAttentionMetadata:
        """Return the group-0 common attention metadata of the step being forwarded.

        Runners with a model state retain the per-group commons in `model_state._decoder_replay_common_metadata`
        (per-group dict, group 0 = layer 0's group); other runners retain the group-0 base object directly.
        """
        if (model_state := getattr(self.npu_runner, "model_state", None)) is not None:
            return model_state._decoder_replay_common_metadata[0]
        # TODO: screen this MRV1 fallback and remove if possible
        return self.npu_runner._decoder_replay_common_metadata  # type: ignore[return-value]

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
        common = self._common_attn_metadata()
        # Route by token count, max across DP ranks (graph modes hold the bucket-padded count — safe:
        # xlite only requires DP-uniform output rows never smaller than the rank's own token count).
        num_tokens: int
        if (num_tokens := _EXTRA_CTX.max_tokens_across_dp) is None:
            if (dp_metadata := forward_context.dp_metadata) is not None:
                num_tokens = dp_metadata.num_tokens_across_dp_cpu.max().item()
            else:
                num_tokens = common.num_actual_tokens
        if num_tokens > self.max_tokens:
            if self.full_mode:
                logger.warning_once(
                    "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                    num_tokens,
                    self.max_tokens,
                    "https://atomgit.com/openeuler/GVirt",
                )
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        # `input_ids` / `inputs_embeds` length = padded token count = the output rows consumers expect.
        if input_ids is not None:
            num_padded_input_tokens = input_ids.size(0)
        elif inputs_embeds is not None:
            num_padded_input_tokens = inputs_embeds.size(0)
        else:  # defensive: the model always receives at least one of the two
            num_padded_input_tokens = common.num_actual_tokens

        num_actual_tokens = common.num_actual_tokens
        # Rows handed to xlite: the true token count, or exactly one token for a DP dummy batch.
        num_forward_tokens = 1 if self._is_dummy_batch else num_actual_tokens

        if self._is_dummy_batch:
            xlite_attn_metadata = self._dummy_attn_metadata
        else:
            num_reqs = common.num_reqs
            seq_lens_cpu = common.seq_lens_cpu
            query_lens = torch.diff(common.query_start_loc_cpu[: num_reqs + 1])
            # `lens` / `cached_lens` are the only derived tensors (no backend ships them); everything else
            # passes through as views of the runner's buffers
            query_start_loc = common.query_start_loc[: num_reqs + 1]
            lens = query_start_loc[1:] - query_start_loc[:-1]
            cached_lens = torch.clamp(common.seq_lens[:num_reqs] - lens, min=0)
            xlite_attn_metadata = AttnMetaV2()
            xlite_attn_metadata.lens_cpu = query_lens.tolist()
            xlite_attn_metadata.cached_lens_cpu = torch.clamp(seq_lens_cpu[:num_reqs] - query_lens, min=0).tolist()
            xlite_attn_metadata.lens = lens
            xlite_attn_metadata.cached_lens = cached_lens
            xlite_attn_metadata.query_start_loc = query_start_loc
            xlite_attn_metadata.positions = positions
            xlite_attn_metadata.slot_mapping = [common.slot_mapping]
            xlite_attn_metadata.block_tables = [common.block_table_tensor]

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        input_ids = cast(torch.Tensor, input_ids)
        curr_stream = torch.npu.current_stream().npu_stream  # type: ignore[attr-defined]
        if curr_stream == 0:
            logger.warning_once(
                "xlite: current stream handle is 0; xlite will fall back to blocking stream synchronization."
            )
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            self.xlite_model.forward_v2(
                rt=self.xlite_rt,
                input=input_ids[:num_forward_tokens],
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=[self.freq_cis],
                output=h,
                curr_stream=curr_stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_forward_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            # Unlike `forward_v2` (whose binding takes `std::vector<at::Tensor>`), the
            # `forward_with_inputs_embeds_v2` binding takes a single `at::Tensor` for `freqs_cis`.
            self.xlite_model.forward_with_inputs_embeds_v2(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=curr_stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if deepstack_input_embeds and callable(
                _clear_stack := getattr(self.npu_runnable, "_clear_deepstack_input_embeds", None)
            ):
                _clear_stack(inputs_embeds.size(0))
        return h[:num_padded_input_tokens]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | torch.nn.Module:
        # TODO: consider implementing a custom logits computation path for xlite to avoid falling back to the native
        return self.npu_runnable.compute_logits(hidden_states)  # type: ignore[attr-defined]
