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
"""Parallel-domain-agnostic full-weight switching for linear methods."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch


@dataclass(frozen=True)
class WeightSwitchGatherSpec:
    """Tensor attribute that should be all-gathered across the selected domain."""

    attr_name: str
    gather_dim: int = 0


@dataclass(frozen=True)
class WeightSwitchRepeatSpec:
    """Tensor attribute that should be expanded by local repeat in the selected domain."""

    attr_name: str
    repeat_dim: int = 0


@dataclass(frozen=True)
class WeightSwitchConfig:
    """Parallel domain used to reconstruct a switched full weight.

    The caller chooses the domain.  DSA-CP currently supplies the TP group;
    PCP can supply its PCP group without introducing a second switching API.
    """

    group: Any
    world_size: int
    rank: int
    shard_axis: Literal["input", "output"] | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                "WeightSwitchConfig requires 0 <= rank < world_size, "
                f"got rank={self.rank}, world_size={self.world_size}."
            )

    @classmethod
    def from_group(
        cls,
        group: Any,
        *,
        shard_axis: Literal["input", "output"] | None = None,
    ) -> "WeightSwitchConfig":
        return cls(
            group=group,
            world_size=group.world_size,
            rank=group.rank_in_group,
            shard_axis=shard_axis,
        )


@dataclass(frozen=True)
class WeightLoadPartition:
    """Checkpoint partition used only while loading a sharded parameter.

    This is deliberately separate from a linear layer's forward communication
    group.  A caller can compose two domains for checkpoint sharding while
    keeping the layer's original forward TP group unchanged.
    """

    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                "WeightLoadPartition requires 0 <= rank < world_size, "
                f"got rank={self.rank}, world_size={self.world_size}."
            )

    @classmethod
    def from_nested_groups(cls, outer_group: Any, inner_group: Any) -> "WeightLoadPartition":
        """Compose contiguous checkpoint shards in ``outer`` then ``inner`` order."""
        return cls(
            world_size=outer_group.world_size * inner_group.world_size,
            rank=outer_group.rank_in_group * inner_group.world_size + inner_group.rank_in_group,
        )


@dataclass
class WeightSwitchGatherPart:
    """Runtime tensors for one all-gathered sharded attribute."""

    spec: WeightSwitchGatherSpec
    local_tensor: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor


@dataclass
class WeightSwitchRepeatPart:
    """Runtime tensors for one locally repeated attribute."""

    spec: WeightSwitchRepeatSpec
    local_tensor: torch.Tensor
    full_tensor: torch.Tensor


@dataclass
class WeightSwitchState:
    """Reusable buffers and outstanding collectives for one linear layer."""

    gather_parts: dict[str, WeightSwitchGatherPart] = field(default_factory=dict)
    repeat_parts: dict[str, WeightSwitchRepeatPart] = field(default_factory=dict)
    handles: list[torch.distributed.Work] = field(default_factory=list)


@dataclass(frozen=True)
class WeightSwitchLoadState:
    """Input partition sizes before and after a second parallel-domain shard.

    A row-parallel linear is normally constructed and loaded with its TP-local
    input width.  PCP adds a second shard inside that TP-local shard, so this
    state records both widths without changing the existing TP loader.
    """

    input_size_per_partition_before: int
    input_size_per_partition_after: int


class WeightSwitchMixin:
    """Opt-in local/full weight switching for one linear method.

    Subclasses declare the direct layer attributes needed by a full-weight
    forward after ``process_weights_after_loading()`` has completed. The
    input-sharded specs are used by row-parallel linears and the
    output-sharded specs by column-parallel linears. When adding support for a
    quantization type, document the following in the subclass docstring or
    adjacent spec declaration:

    1. The post-processing layout and sharded dimension of every declared
       attribute. ``WeightSwitchGatherSpec.gather_dim`` must identify that exact
       dimension, including packed or transposed weight layouts.
    2. Every attribute that must be concatenated across TP ranks in
       ``weight_switch_gather_specs``. This normally includes the weight and any
       rank-distinct scale or metadata consumed by the full-weight kernel.
    3. Every attribute that must be repeated locally in
       ``weight_switch_repeat_specs``. Use this only when the full-weight kernel
       expects the same rank-local value once per parallel shard; rank-distinct data
       must be gathered instead.
    4. Both input- and output-sharded layouts when the method can be used by
       row- and column-parallel linears. Include every quantization parameter
       that becomes sharded in each layout.
    5. Set ``supports_weight_switch = True`` only after validating every
       declared layout. Callers use this as the common opt-in gate; it carries
       no SFA- or DSA-CP-specific meaning.

    ``attr_name`` must name a direct tensor attribute on the layer. The mixin
    owns only reusable local/full state and collective handles; subclasses retain
    ownership of the attribute list and its quantization-specific semantics.
    """

    weight_switch_gather_specs: ClassVar[tuple[WeightSwitchGatherSpec, ...]] = ()
    weight_switch_repeat_specs: ClassVar[tuple[WeightSwitchRepeatSpec, ...]] = ()
    weight_switch_output_gather_specs: ClassVar[tuple[WeightSwitchGatherSpec, ...]] = ()
    weight_switch_output_repeat_specs: ClassVar[tuple[WeightSwitchRepeatSpec, ...]] = ()
    supports_weight_switch: ClassVar[bool] = False

    def _get_weight_switch_specs(
        self,
        layer: torch.nn.Module,
        config: WeightSwitchConfig,
    ) -> tuple[
        tuple[WeightSwitchGatherSpec, ...],
        tuple[WeightSwitchRepeatSpec, ...],
        str,
    ]:
        """Resolve the layer axis whose local/full views will be switched."""
        input_size = getattr(layer, "input_size", None)
        input_size_per_partition = getattr(layer, "input_size_per_partition", None)
        output_size = getattr(layer, "output_size", None)
        output_size_per_partition = getattr(layer, "output_size_per_partition", None)
        input_sharded = (
            input_size is not None and input_size_per_partition is not None and input_size != input_size_per_partition
        )
        output_sharded = (
            output_size is not None
            and output_size_per_partition is not None
            and output_size != output_size_per_partition
        )
        if input_sharded and output_sharded:
            raise RuntimeError(
                "Weight switching requires a linear layer with at most one pre-existing sharded axis, "
                f"got input_size={input_size}, input_size_per_partition={input_size_per_partition}, "
                f"output_size={output_size}, output_size_per_partition={output_size_per_partition}."
            )
        if input_sharded:
            shard_axis = "input"
        elif output_sharded:
            shard_axis = "output"
        elif config.shard_axis is not None:
            shard_axis = config.shard_axis
        else:
            raise RuntimeError(
                "Weight switching needs WeightSwitchConfig.shard_axis when the layer is not already sharded."
            )

        if shard_axis == "input":
            gather_specs = self.weight_switch_gather_specs
            repeat_specs = self.weight_switch_repeat_specs
        else:
            gather_specs = self.weight_switch_output_gather_specs
            repeat_specs = self.weight_switch_output_repeat_specs
        return gather_specs, repeat_specs, shard_axis

    @staticmethod
    def split_tensor_for_parallel(
        tensor: torch.Tensor,
        world_size: int,
        rank: int,
        dim: int = 0,
        contiguous: bool = True,
    ) -> torch.Tensor:
        """Slice one shard from a full tensor in a parallel domain."""
        dim = dim if dim >= 0 else tensor.dim() + dim
        if tensor.shape[dim] % world_size != 0:
            raise RuntimeError(
                "Cannot split tensor because the target dimension is not divisible by the parallel world size: "
                f"shape={tuple(tensor.shape)}, dim={dim}, world_size={world_size}."
            )
        shard_size = tensor.shape[dim] // world_size
        shard = tensor.narrow(dim, rank * shard_size, shard_size)
        return shard.contiguous() if contiguous else shard

    def prepare_layer_for_parallel_weight_load(
        self,
        layer: torch.nn.Module,
        config: WeightSwitchConfig,
        load_partition: WeightLoadPartition,
    ) -> WeightSwitchLoadState:
        """Install direct checkpoint loading into a nested local shard.

        The input-sharded parameters are resized before ``model.load_weights``.
        Their replacement loader takes a single contiguous checkpoint slice
        selected by ``load_partition``; no TP-local checkpoint tensor is loaded
        before selecting the PCP-local shard.  The layer's forward TP metadata
        is intentionally not modified.
        """
        if not self.supports_weight_switch:
            raise RuntimeError(f"{type(self).__name__} does not support weight switching.")

        original_width = getattr(layer, "input_size_per_partition", None)
        if not isinstance(original_width, int) or original_width <= 0:
            raise RuntimeError(
                "Loader-time weight sharding requires a row-parallel layer with "
                f"a positive input_size_per_partition, got {original_width!r}."
            )
        if original_width % config.world_size != 0:
            raise RuntimeError(
                "Cannot split the TP-local input width across the requested parallel domain: "
                f"input_size_per_partition={original_width}, world_size={config.world_size}."
            )
        if load_partition.world_size % config.world_size != 0:
            raise RuntimeError(
                "The checkpoint load partition must contain an integral number "
                "of the switched parallel-domain shards: "
                f"load_world_size={load_partition.world_size}, switch_world_size={config.world_size}."
            )
        if hasattr(layer, "_weight_switch_load_state"):
            raise RuntimeError(
                f"Loader-time weight sharding has already been installed for {getattr(layer, 'prefix', layer)}."
            )

        local_width = original_width // config.world_size
        wrapped_params = 0
        for _, param in layer.named_parameters(recurse=False):
            input_dim = getattr(param, "input_dim", None)
            original_loader = getattr(param, "weight_loader", None)
            if input_dim is None:
                continue
            if original_loader is None:
                raise RuntimeError(
                    "Loader-time weight sharding requires weight_loader on every input-sharded parameter: "
                    f"layer={getattr(layer, 'prefix', layer)}, parameter={param}."
                )

            physical_shape = list(param.shape)
            if physical_shape[input_dim] % config.world_size != 0:
                raise RuntimeError(
                    "Cannot resize input-sharded parameter for direct checkpoint loading: "
                    f"layer={getattr(layer, 'prefix', layer)}, shape={tuple(param.shape)}, "
                    f"input_dim={input_dim}, world_size={config.world_size}."
                )
            physical_shape[input_dim] //= config.world_size
            with torch.no_grad():
                param.set_(torch.empty(tuple(physical_shape), dtype=param.dtype, device=param.device))

            is_v2_parameter = hasattr(param, "load_row_parallel_weight") and hasattr(param, "tp_rank")

            def weight_loader(
                target_param: torch.nn.Parameter,
                loaded_weight: torch.Tensor,
                *args: Any,
                _original_loader=original_loader,
                _is_v2_parameter=is_v2_parameter,
                **kwargs: Any,
            ) -> Any:
                target_input_dim = getattr(target_param, "input_dim", None)
                if target_input_dim is None:
                    return _original_loader(target_param, loaded_weight, *args, **kwargs)

                if _is_v2_parameter:
                    # v2 parameters perform their own row-parallel narrow. Give
                    # that implementation the composed checkpoint rank only for
                    # this load; it has no bearing on forward TP communication.
                    old_rank = target_param.tp_rank
                    old_size = target_param.tp_size
                    target_param.tp_rank = load_partition.rank
                    target_param.tp_size = load_partition.world_size
                    try:
                        return _original_loader(target_param, loaded_weight, *args, **kwargs)
                    finally:
                        target_param.tp_rank = old_rank
                        target_param.tp_size = old_size

                is_sharded_weight = getattr(target_param, "is_sharded_weight", False)
                use_bitsandbytes_4bit = getattr(target_param, "use_bitsandbytes_4bit", False)
                is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
                param_data = target_param.data
                if not is_sharded_weight:
                    shard_size = param_data.shape[target_input_dim]
                    start_idx = load_partition.rank * shard_size
                    loaded_weight = loaded_weight.narrow(target_input_dim, start_idx, shard_size)
                elif loaded_weight.shape != param_data.shape:
                    # A pre-sharded checkpoint may already be outer-domain local.
                    # In that case select only the switched-domain shard.
                    shard_size = param_data.shape[target_input_dim]
                    if loaded_weight.shape[target_input_dim] == shard_size * config.world_size:
                        start_idx = config.rank * shard_size
                        loaded_weight = loaded_weight.narrow(target_input_dim, start_idx, shard_size)

                if len(loaded_weight.shape) == 0:
                    loaded_weight = loaded_weight.reshape(1)
                if param_data.shape != loaded_weight.shape:
                    raise RuntimeError(
                        "Direct checkpoint loader produced an unexpected parameter shape: "
                        f"target={tuple(param_data.shape)}, loaded={tuple(loaded_weight.shape)}, "
                        f"input_dim={target_input_dim}, load_rank={load_partition.rank}, "
                        f"load_world_size={load_partition.world_size}."
                    )
                param_data.copy_(loaded_weight)
                return None

            param.weight_loader = weight_loader
            wrapped_params += 1

        if wrapped_params == 0:
            raise RuntimeError(
                "Loader-time weight sharding found no input-sharded parameters on "
                f"{getattr(layer, 'prefix', layer)}."
            )

        load_state = WeightSwitchLoadState(
            input_size_per_partition_before=original_width,
            input_size_per_partition_after=local_width,
        )
        layer.input_size_per_partition = local_width
        layer._weight_switch_load_state = load_state
        return load_state

    def enable_weight_switch(
        self,
        layer: torch.nn.Module,
        config: WeightSwitchConfig,
        *,
        pool: dict[Any, torch.Tensor] | None = None,
        pool_key_prefix: Any | None = None,
        clone_local_tensors: bool = False,
    ) -> WeightSwitchState:
        """Allocate local/full aliases and buffers for one compatible linear layer."""
        if not self.supports_weight_switch:
            raise RuntimeError(f"{type(self).__name__} does not support weight switching.")

        gather_specs, repeat_specs, shard_axis = self._get_weight_switch_specs(layer, config)
        if not gather_specs and not repeat_specs:
            raise RuntimeError(
                f"{type(self).__name__} does not declare weight-switch specs for a {shard_axis}-sharded layer."
            )

        state = WeightSwitchState()

        for gather_spec in gather_specs:
            tensor = getattr(layer, gather_spec.attr_name, None)
            if tensor is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares gather attr {gather_spec.attr_name!r}, "
                    f"but layer {getattr(layer, 'prefix', layer)} does not have it."
                )
            dim = gather_spec.gather_dim if gather_spec.gather_dim >= 0 else tensor.dim() + gather_spec.gather_dim
            if dim < 0 or dim >= tensor.dim():
                raise RuntimeError(
                    f"Invalid gather dim {gather_spec.gather_dim} for attr {gather_spec.attr_name!r} "
                    f"with shape {tuple(tensor.shape)}."
                )

            local_tensor = tensor.clone().detach().contiguous() if clone_local_tensors else tensor.detach()
            if clone_local_tensors:
                with torch.no_grad():
                    tensor.set_(local_tensor)
            full_shape_list = list(local_tensor.shape)
            full_shape_list[dim] *= config.world_size
            full_shape = tuple(full_shape_list)
            if dim == 0:
                gather_input = local_tensor
            else:
                gather_input = torch.movedim(local_tensor, dim, 0).contiguous()
            gather_shape = (full_shape[dim], *full_shape[:dim], *full_shape[dim + 1 :])
            pool_key = (
                pool_key_prefix,
                gather_spec.attr_name,
                local_tensor.device.type,
                local_tensor.device.index,
                local_tensor.dtype,
                dim,
                full_shape,
            )
            if pool is None:
                gather_output = torch.empty(gather_shape, dtype=local_tensor.dtype, device=local_tensor.device)
            else:
                gather_output = pool.get(pool_key)
                if gather_output is None:
                    gather_output = torch.empty(gather_shape, dtype=local_tensor.dtype, device=local_tensor.device)
                    pool[pool_key] = gather_output
            if dim == 0:
                full_tensor = gather_output
            else:
                full_tensor = torch.movedim(gather_output, 0, dim)
            state.gather_parts[gather_spec.attr_name] = WeightSwitchGatherPart(
                spec=gather_spec,
                local_tensor=local_tensor,
                gather_input=gather_input,
                gather_output=gather_output,
                full_tensor=full_tensor,
            )

        for repeat_spec in repeat_specs:
            tensor = getattr(layer, repeat_spec.attr_name, None)
            if tensor is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares repeat attr {repeat_spec.attr_name!r}, "
                    f"but layer {getattr(layer, 'prefix', layer)} does not have it."
                )
            dim = repeat_spec.repeat_dim if repeat_spec.repeat_dim >= 0 else tensor.dim() + repeat_spec.repeat_dim
            if dim < 0 or dim >= tensor.dim():
                raise RuntimeError(
                    f"Invalid repeat dim {repeat_spec.repeat_dim} for attr {repeat_spec.attr_name!r} "
                    f"with shape {tuple(tensor.shape)}."
                )
            repeats = [1] * tensor.dim()
            repeats[dim] = config.world_size
            local_tensor = tensor.detach()
            state.repeat_parts[repeat_spec.attr_name] = WeightSwitchRepeatPart(
                spec=repeat_spec,
                local_tensor=local_tensor,
                full_tensor=local_tensor.repeat(*repeats),
            )

        return state

    def all_gather_weight(
        self,
        state: WeightSwitchState,
        config: WeightSwitchConfig,
        *,
        async_op: bool = True,
    ) -> None:
        """All-gather every sharded tensor in ``state`` across ``config.group``."""
        from vllm_ascend.distributed.utils import all_gather_async

        if state.handles:
            raise RuntimeError("Weight all-gather is still pending; wait before launching another one.")
        for part in state.gather_parts.values():
            _, handle = all_gather_async(
                part.gather_input,
                config.group,
                output=part.gather_output,
                async_op=async_op,
            )
            if handle is not None:
                state.handles.append(handle)

    @staticmethod
    def wait_weight_all_gather(state: WeightSwitchState) -> None:
        try:
            for handle in state.handles:
                handle.wait()
        finally:
            state.handles.clear()

    def switch_weight(
        self,
        layer: torch.nn.Module,
        state: WeightSwitchState,
        *,
        use_full_weight: bool,
    ) -> None:
        """Switch layer tensor attributes between local and full views."""
        for attr_name, gather_part in state.gather_parts.items():
            target = gather_part.full_tensor if use_full_weight else gather_part.local_tensor
            with torch.no_grad():
                getattr(layer, attr_name).set_(target)
        for attr_name, repeat_part in state.repeat_parts.items():
            target = repeat_part.full_tensor if use_full_weight else repeat_part.local_tensor
            with torch.no_grad():
                getattr(layer, attr_name).set_(target)
