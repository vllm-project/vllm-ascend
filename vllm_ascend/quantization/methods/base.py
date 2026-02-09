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

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch


class QuantType(Enum):
    """Quantization type enum for MoE schemes."""

    NONE = 0
    W8A8 = 1
    W4A8 = 2


class AscendLinearScheme(ABC):
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

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return per-tensor parameter specifications (e.g., input_scale).

        Args:
            params_dtype: Data type for parameters.

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
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Forward computation for MoE layer.

        Args:
            layer: The MoE layer module.
            x: Input hidden states.
            router_logits: Router logits for expert selection.
            top_k: Number of experts to select per token.
            renormalize: Whether to renormalize expert weights.
            use_grouped_topk: Whether to use grouped top-k selection.
            global_num_experts: Total number of experts globally.
            expert_map: Mapping from local to global expert indices.
            topk_group: Group size for grouped top-k.
            num_expert_group: Number of expert groups.
            custom_routing_function: Custom routing function.
            scoring_func: Scoring function name.
            routed_scaling_factor: Scaling factor for routed experts.
            e_score_correction_bias: Expert score correction bias.
            is_prefill: Whether in prefill phase.
            enable_force_load_balance: Whether to force load balancing.
            log2phy: Logical to physical expert mapping.
            global_redundant_expert_num: Number of redundant experts.
            **kwargs: Additional keyword arguments.

        Returns:
            Output tensor after MoE computation.
        """
        ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for MoE layer.

        Args:
            layer: The MoE layer module.
        """
        return
