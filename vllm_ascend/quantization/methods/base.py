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
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vllm_ascend.quantization.quant_type import QuantType


@dataclass(frozen=True)
class TPWeightGatherSpec:
    """Tensor attribute that should be all-gathered across TP ranks."""

    attr_name: str
    gather_dim: int = 0


@dataclass(frozen=True)
class TPWeightRepeatSpec:
    """Tensor attribute that should be expanded by local repeat."""

    attr_name: str
    repeat_dim: int = 0


@dataclass
class TPWeightGatherPart:
    """Runtime tensors for one all-gathered TP attribute."""

    spec: TPWeightGatherSpec
    tp_tensor: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor


@dataclass
class TPWeightRepeatPart:
    """Runtime tensors for one repeated TP attribute."""

    spec: TPWeightRepeatSpec
    tp_tensor: torch.Tensor
    full_tensor: torch.Tensor


@dataclass
class TPWeightSwitchState:
    """Reusable buffers for switching a layer between TP and full weights."""

    gather_parts: dict[str, TPWeightGatherPart] = field(default_factory=dict)
    repeat_parts: dict[str, TPWeightRepeatPart] = field(default_factory=dict)


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


class AscendLinearScheme(ABC):
    """Base class for all linear quantization schemes.

    Subclasses must implement get_weight() and apply() methods.
    Other methods have default implementations that return empty dicts
    or do nothing.
    """

    tp_weight_gather_specs: ClassVar[tuple[TPWeightGatherSpec, ...]] = ()
    tp_weight_repeat_specs: ClassVar[tuple[TPWeightRepeatSpec, ...]] = ()

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

    @staticmethod
    def split_tensor_for_tp(
        tensor: torch.Tensor,
        tp_size: int,
        tp_rank: int,
        dim: int = 0,
        contiguous: bool = True,
    ) -> torch.Tensor:
        """Slice one TP shard from a full tensor."""
        dim = dim if dim >= 0 else tensor.dim() + dim
        if tensor.shape[dim] % tp_size != 0:
            raise RuntimeError(
                "Cannot split tensor for TP because the target dimension is not divisible by TP size: "
                f"shape={tuple(tensor.shape)}, dim={dim}, tp_size={tp_size}."
            )
        shard_size = tensor.shape[dim] // tp_size
        shard = tensor.narrow(dim, tp_rank * shard_size, shard_size)
        return shard.contiguous() if contiguous else shard

    def enable_tp_weight_switch(
        self,
        layer: torch.nn.Module,
        tp_size: int,
        *,
        pool: dict[Any, torch.Tensor] | None = None,
        pool_key_prefix: Any | None = None,
        clone_tp_tensors: bool = False,
    ) -> TPWeightSwitchState:
        """Enable TP/full weight switching for one linear layer.

        This is deliberately an explicit feature entry point. It allocates the
        full-weight and repeated-parameter buffers only when a caller needs to
        switch a layer out of its normal TP layout.
        """
        state = TPWeightSwitchState()
        gather_specs = self.tp_weight_gather_specs
        repeat_specs = self.tp_weight_repeat_specs

        for spec in gather_specs:
            tensor = getattr(layer, spec.attr_name, None)
            if tensor is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares TP gather attr {spec.attr_name!r}, "
                    f"but layer {getattr(layer, 'prefix', layer)} does not have it."
                )
            dim = spec.gather_dim if spec.gather_dim >= 0 else tensor.dim() + spec.gather_dim
            if dim < 0 or dim >= tensor.dim():
                raise RuntimeError(
                    f"Invalid TP gather dim {spec.gather_dim} for attr {spec.attr_name!r} "
                    f"with shape {tuple(tensor.shape)}."
            )

            tp_tensor = tensor.clone().detach().contiguous() if clone_tp_tensors else tensor.detach()
            if clone_tp_tensors:
                with torch.no_grad():
                    tensor.set_(tp_tensor)
            full_shape_list = list(tp_tensor.shape)
            full_shape_list[dim] *= tp_size
            full_shape = tuple(full_shape_list)
            if dim == 0:
                gather_input = tp_tensor
            else:
                gather_input = torch.movedim(tp_tensor, dim, 0).contiguous()
            gather_shape = (full_shape[dim], *full_shape[:dim], *full_shape[dim + 1 :])
            pool_key = (
                pool_key_prefix,
                spec.attr_name,
                tp_tensor.device.type,
                tp_tensor.device.index,
                tp_tensor.dtype,
                dim,
                full_shape,
            )
            if pool is None:
                gather_output = torch.empty(gather_shape, dtype=tp_tensor.dtype, device=tp_tensor.device)
            else:
                gather_output = pool.get(pool_key)
                if gather_output is None:
                    gather_output = torch.empty(gather_shape, dtype=tp_tensor.dtype, device=tp_tensor.device)
                    pool[pool_key] = gather_output
            if dim == 0:
                full_tensor = gather_output
            else:
                full_tensor = torch.movedim(gather_output, 0, dim)
            state.gather_parts[spec.attr_name] = TPWeightGatherPart(
                spec=spec,
                tp_tensor=tp_tensor,
                gather_input=gather_input,
                gather_output=gather_output,
                full_tensor=full_tensor,
            )

        for spec in repeat_specs:
            tensor = getattr(layer, spec.attr_name, None)
            if tensor is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares TP repeat attr {spec.attr_name!r}, "
                    f"but layer {getattr(layer, 'prefix', layer)} does not have it."
                )
            dim = spec.repeat_dim if spec.repeat_dim >= 0 else tensor.dim() + spec.repeat_dim
            if dim < 0 or dim >= tensor.dim():
                raise RuntimeError(
                    f"Invalid TP repeat dim {spec.repeat_dim} for attr {spec.attr_name!r} "
                    f"with shape {tuple(tensor.shape)}."
                )
            repeats = [1] * tensor.dim()
            repeats[dim] = tp_size
            tp_tensor = tensor.detach()
            state.repeat_parts[spec.attr_name] = TPWeightRepeatPart(
                spec=spec,
                tp_tensor=tp_tensor,
                full_tensor=tp_tensor.repeat(*repeats),
            )

        return state

    def all_gather_tp_weight(
        self,
        state: TPWeightSwitchState,
        group: Any,
        *,
        async_op: bool = True,
    ) -> list[torch.distributed.Work | None]:
        """All-gather every TP-sharded tensor in ``state``."""
        from vllm_ascend.distributed.utils import all_gather_async

        handles: list[torch.distributed.Work | None] = []
        for part in state.gather_parts.values():
            _, handle = all_gather_async(
                part.gather_input,
                group,
                output=part.gather_output,
                async_op=async_op,
            )
            handles.append(handle)
        return handles

    @staticmethod
    def wait_tp_weight_all_gather(handles: list[torch.distributed.Work | None]) -> None:
        for handle in handles:
            if handle is not None:
                handle.wait()

    def switch_tp_weight(
        self,
        layer: torch.nn.Module,
        state: TPWeightSwitchState,
        *,
        use_full_weight: bool,
    ) -> None:
        """Switch layer tensor attributes between TP-local and full views."""
        for attr_name, part in state.gather_parts.items():
            target = part.full_tensor if use_full_weight else part.tp_tensor
            with torch.no_grad():
                getattr(layer, attr_name).set_(target)
        for attr_name, part in state.repeat_parts.items():
            target = part.full_tensor if use_full_weight else part.tp_tensor
            with torch.no_grad():
                getattr(layer, attr_name).set_(target)


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
