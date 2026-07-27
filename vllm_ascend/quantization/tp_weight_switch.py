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
"""TP full-weight switching support shared by compatible linear methods."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch


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
    """Reusable buffers and outstanding collectives for one linear layer."""

    gather_parts: dict[str, TPWeightGatherPart] = field(default_factory=dict)
    repeat_parts: dict[str, TPWeightRepeatPart] = field(default_factory=dict)
    handles: list[torch.distributed.Work] = field(default_factory=list)


class TPWeightSwitchMixin:
    """Opt-in TP/full weight switching for one linear method.

    Subclasses declare the direct layer attributes needed by a full-weight
    forward after ``process_weights_after_loading()`` has completed. When
    adding support for a quantization type, document the following in the
    subclass docstring or adjacent spec declaration:

    1. The post-processing layout and TP-sharded dimension of every declared
       attribute. ``TPWeightGatherSpec.gather_dim`` must identify that exact
       dimension, including packed or transposed weight layouts.
    2. Every attribute that must be concatenated across TP ranks in
       ``tp_weight_gather_specs``. This normally includes the weight and any
       rank-distinct scale or metadata consumed by the full-weight kernel.
    3. Every attribute that must be repeated locally in
       ``tp_weight_repeat_specs``. Use this only when the full-weight kernel
       expects the same rank-local value once per TP shard; rank-distinct data
       must be gathered instead.
    4. The supported o_proj execution paths and any excluded path. Declaring
       switch specs only guarantees weight switching; it does not make a
       quantized operator compatible with every SFA or DSA-CP kernel. Declare
       ``supports_dsa_cp_o_proj = True`` only after validating the DSA-CP
       operator.

    ``attr_name`` must name a direct tensor attribute on the layer. The mixin
    owns only reusable TP/full state and collective handles; subclasses retain
    ownership of the attribute list and its quantization-specific semantics.
    """

    tp_weight_gather_specs: ClassVar[tuple[TPWeightGatherSpec, ...]] = ()
    tp_weight_repeat_specs: ClassVar[tuple[TPWeightRepeatSpec, ...]] = ()
    supports_dsa_cp_o_proj: ClassVar[bool] = False

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
        """Allocate TP/full aliases and buffers for one compatible linear layer."""
        state = TPWeightSwitchState()

        for spec in self.tp_weight_gather_specs:
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

        for spec in self.tp_weight_repeat_specs:
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
    ) -> None:
        """All-gather every TP-sharded tensor in ``state``."""
        from vllm_ascend.distributed.utils import all_gather_async

        if state.handles:
            raise RuntimeError("TP weight all-gather is still pending; wait before launching another one.")
        for part in state.gather_parts.values():
            _, handle = all_gather_async(
                part.gather_input,
                group,
                output=part.gather_output,
                async_op=async_op,
            )
            if handle is not None:
                state.handles.append(handle)

    @staticmethod
    def wait_tp_weight_all_gather(state: TPWeightSwitchState) -> None:
        try:
            for handle in state.handles:
                handle.wait()
        finally:
            state.handles.clear()

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
