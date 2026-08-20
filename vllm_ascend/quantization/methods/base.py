#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Abstract base classes for Ascend quantization schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.moe_utils import maybe_normalize_mxfp_scale_layout
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.tp_weight_switch import (
    TPWeightGatherPart,
    TPWeightGatherSpec,
    TPWeightRepeatPart,
    TPWeightRepeatSpec,
    TPWeightSwitchMixin,
    TPWeightSwitchState,
)

__all__ = [
    "AscendAttentionScheme",
    "AscendLinearScheme",
    "AscendMoEScheme",
    "QuantType",
    "TPWeightGatherPart",
    "TPWeightGatherSpec",
    "TPWeightRepeatPart",
    "TPWeightRepeatSpec",
    "TPWeightSwitchMixin",
    "TPWeightSwitchState",
]

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput


def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)

    return int(num_experts - global_redundant_expert_num - num_shared_experts)


class AscendLinearScheme(TPWeightSwitchMixin, ABC):
    """Base class for all linear quantization schemes.

    Subclasses must implement get_weight() and apply() methods.
    Other methods have default implementations that return empty dicts
    or do nothing.
    """

    @abstractmethod
    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return weight tensor specifications.

        Args:
            input_size: Input dimension of the linear layer.
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors with
            the correct shape and dtype.
        """
        ...

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        """Return per-tensor parameter specifications (e.g., input_scale).

        Args:
            params_dtype: Data type for parameters.
            **kwargs: Additional keyword arguments for subclass extensions

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return per-channel parameter specifications (e.g., weight_scale).

        Args:
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return per-group parameter specifications.

        Args:
            input_size: Input dimension of the linear layer.
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.
            layer_type: Type of layer (e.g., "row" for RowParallelLinear).

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    @abstractmethod
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            layer: The linear layer module.
            x: Input tensor.
            bias: Optional bias tensor.
            tp_rank: Tensor parallel rank.

        Returns:
            Output tensor after quantized linear operation.
        """
        ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing (transpose, format conversion, etc.).

        Args:
            layer: The linear layer module.
        """
        return


class AscendAttentionScheme(ABC):
    """Base class for all attention quantization schemes.

    Subclasses must implement apply() method.
    Other methods have default implementations.
    """

    def create_weights(self, layer: torch.nn.Module) -> None:
        """Create weights for attention quantization.

        Args:
            layer: The attention layer module.
        """
        return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for attention layer.

        Args:
            layer: The attention layer module.
        """
        return

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        """Forward computation for attention layer.

        Args:
            layer: The attention layer module.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_cache: KV cache.
            attn_metadata: Attention metadata.
            attn_type: Attention type.
            scale: Scale factor.
            output: Output tensor.

        Returns:
            Output tensor after attention computation.
        """
        ...


class AscendMoEScheme(ABC):
    """Base class for all MoE quantization schemes.

    Subclasses must implement get_weight(), get_dynamic_quant_param(),
    and apply() methods.

    Attributes:
        quant_type: The quantization type for this scheme. Subclasses should
                   override this class attribute to declare their quant type.
    """

    # Default quant type - subclasses should override this
    quant_type: QuantType = QuantType.NONE
    # Activation quant dtype used by the MLP gmm hooks. Subclasses override it.
    act_quant_type: torch.dtype | None = None
    # Activations that this method implements through a fused gmm1+act+quant
    # path (``apply_gmm1_act_quant``). Stored as activation string values.
    fused_activations: frozenset[str] = frozenset()

    @abstractmethod
    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return weight tensor specifications for MoE layer.

        Args:
            num_experts: Number of experts.
            intermediate_size_per_partition: Intermediate size per partition.
            hidden_sizes: Hidden dimension size.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        ...

    @abstractmethod
    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return dynamic quantization parameters for MoE layer.

        Args:
            num_experts: Number of experts.
            intermediate_size_per_partition: Intermediate size per partition.
            hidden_sizes: Hidden dimension size.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        ...

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward computation for MoE layer.

        Args:
            layer: The MoE layer module.
            x: Input hidden states.
            topk_weights: Router weights of shape (num_tokens, top_k).
            topk_ids: Selected expert ids of shape (num_tokens, top_k).

        Returns:
            Output tensor after MoE computation.
        """
        ...

    def get_eplb_weight_views(self, layer: torch.nn.Module) -> list[torch.Tensor]:
        """Return expert-first weight views consumed by upstream EPLB."""
        return []

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for MoE layer.

        Args:
            layer: The MoE layer module.
        """
        return

    def _quant_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.quant_type in [QuantType.W4A16, QuantType.W4A16MXFP]:
            # A16 quantization doesn't need to quant hidden_states
            return hidden_states, None
        # Each quant method knows its own quant type, so the kernel is called
        # directly instead of going through a device-level wrapper:
        # MXFP types always quantize with the A5 MX kernel, all others with the
        # generic dynamic quant kernel.
        use_mxfp_quant = self.quant_type in (
            QuantType.W8A8MXFP,
            QuantType.W4A4MXFP,
            QuantType.W4A8MXFP,
            QuantType.W4A16MXFP,
        )
        if dynamic_scale is None:
            # When hidden_states haven't been quanted, we need to quant hidden_states.
            if use_mxfp_quant:
                hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=self.act_quant_type
                )
                return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)
            return torch_npu.npu_dynamic_quant(hidden_states, dst_type=self.act_quant_type)
        # When hidden_states have been quanted, we don't need to quant hidden_states.
        if use_mxfp_quant:
            return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)
        return hidden_states, dynamic_scale

    def supports_fused_activation(self, activation) -> bool:
        """Whether this method provides a fused gmm1+act+quant path for
        ``activation`` (an ``MoEActivation`` member or its string value)."""
        return getattr(activation, "value", activation) in self.fused_activations

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        """gate/up projection (gmm1), returns the pre-activation output."""
        raise NotImplementedError(f"{type(self).__name__} does not implement apply_gmm1().")

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused gmm1 + activation + output quantization.

        Only called for activations listed in ``fused_activations``. Returns
        ``(hidden_states, act_out_scale)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement apply_gmm1_act_quant().")

    def apply_act_quant(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(Re)quantize the activation output. A16 paths return
        ``(hidden_states, None)``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement apply_act_quant().")

    def apply_gmm2(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        hidden_states: torch.Tensor,
        act_out_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """down projection (gmm2)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement apply_gmm2().")

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Build the normalized :class:`MoEWeights` payload from ``layer``.

        Used by the FUSED_MC2 communication path, which bypasses the MLP
        stage and needs the per-expert weight lists assembled per quant
        method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_fused_mc2_weights().")

    def get_mlp_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Build the standard MLP-layout :class:`MoEWeights` payload.

        Used by the quantized MoE LoRA backend, which needs the base-expert
        weights in the same layout as the MLP gmm hooks (w1/w2 with their
        scales) now that weights are carried by the routed-expert layer.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_mlp_weights().")
