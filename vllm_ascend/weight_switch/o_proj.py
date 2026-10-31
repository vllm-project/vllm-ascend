# SPDX-License-Identifier: Apache-2.0
"""Parallel O-projection execution independent of the attention backend."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from .linear import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchMixin,
    WeightSwitchState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class OProjectionMode(Enum):
    FULL_WEIGHT = auto()
    LOCAL_WEIGHT = auto()
    EXCHANGE_TOKENS = auto()


class ParallelOProjection:
    """Own O-projection loading, temporary weights, communication and bias.

    A single linear projection uses the default adapter. A grouped projection
    additionally supplies its first factor and its grouped matmul kernel.
    Callers retain only metadata translation and weight-gather launch points.
    """

    def __init__(
        self,
        input_size: int,
        output_layer: torch.nn.Module,
        *,
        config: WeightSwitchConfig,
        tp_group: Any,
        pool: dict[Any, torch.Tensor],
        pool_key: str,
        project_full: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decode_sharded: bool,
        input_layer: torch.nn.Module | None = None,
        input_projection: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_input_groups: int | None = None,
        clone_local_tensors: bool = False,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_layer.output_size_per_partition
        self.o_proj = output_layer
        self.input_layer = input_layer
        self.input_projection = input_projection
        self.num_input_groups = num_input_groups
        self.tp_group = tp_group
        self.config = config
        self.decode_sharded = decode_sharded
        self.o_proj_full_pools = pool
        self.o_proj_weight_switch_pool_key = pool_key
        self.project_full = project_full
        self.clone_local_tensors = clone_local_tensors
        self.enabled = False
        self.mode = OProjectionMode.LOCAL_WEIGHT
        self.token_capacity = 0
        self.all_gather_input: torch.Tensor | None = None
        self.all_gather_output: torch.Tensor | None = None
        self.reduce_scatter_input: torch.Tensor | None = None
        self.reduce_scatter_output: torch.Tensor | None = None
        self._weight_switch_states: tuple[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState], ...] | None = (
            None
        )
        self._weight_switch_enabled = False
        self._load_prepared = False

    @staticmethod
    def _resolve_o_proj_weight_switch_method(layer: torch.nn.Module) -> WeightSwitchMixin:
        quant_method = getattr(layer, "quant_method", None)
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if not isinstance(linear_method, WeightSwitchMixin) or not linear_method.supports_weight_switch:
            raise RuntimeError(
                f"O-proj weight switching requires a weight-switch capable method, got {type(linear_method).__name__}."
            )
        return linear_method

    def _layers(self) -> tuple[tuple[str, torch.nn.Module, WeightSwitchMixin], ...]:
        layers: tuple[tuple[str, torch.nn.Module], ...] = (("output_projection", self.o_proj),)
        if self.input_layer is not None:
            layers = (("input_projection", self.input_layer), *layers)
        return tuple((name, layer, self._resolve_o_proj_weight_switch_method(layer)) for name, layer in layers)

    def prepare_weight_load(self, vllm_config: "VllmConfig") -> None:
        groups = self.num_input_groups
        if groups is not None and (groups <= 0 or groups % self.config.world_size != 0):
            raise ValueError(f"O-proj groups={groups} cannot be split across world_size={self.config.world_size}.")
        if self.input_layer is not None and (groups is None or self.input_projection is None):
            raise ValueError("A grouped O projection requires its group count and projection kernel.")
        if not self._load_prepared:
            partition = WeightLoadPartition.from_nested_groups(self.tp_group, self.config.group)
            for name, layer, method in self._layers():
                if name == "output_projection":
                    method.prepare_layer_for_parallel_weight_load(layer, self.config, partition)
                else:
                    method.prepare_layer_for_parallel_output_weight_load(layer, self.config, partition)
            self._load_prepared = True
        if self.input_layer is not None:
            assert groups is not None
            local_groups = groups // self.config.world_size
            self.input_layer.n_local_groups = local_groups
            method = self._resolve_o_proj_weight_switch_method(self.input_layer)
            if hasattr(method, "n_local_groups"):
                method.n_local_groups = local_groups
        if self.decode_sharded:
            self._prepare_exchange_buffers(vllm_config)
        self.enabled = True

    def enable(self) -> None:
        if not self.enabled or self._weight_switch_enabled:
            return
        self._weight_switch_states = tuple(
            (
                layer,
                method,
                method.enable_weight_switch(
                    layer,
                    self.config,
                    pool=self.o_proj_full_pools,
                    pool_key_prefix=(type(method).__qualname__, name, self.o_proj_weight_switch_pool_key),
                    clone_local_tensors=self.clone_local_tensors,
                ),
            )
            for name, layer, method in self._layers()
        )
        self._weight_switch_enabled = True

    @property
    def uses_full_weight(self) -> bool:
        return self.mode is OProjectionMode.FULL_WEIGHT

    @property
    def exchanges_tokens(self) -> bool:
        return self.mode is OProjectionMode.EXCHANGE_TOKENS

    def set_mode(self, *, use_full_weight: bool, exchange_tokens: bool) -> None:
        if not self._weight_switch_enabled:
            self.mode = OProjectionMode.LOCAL_WEIGHT
        elif use_full_weight:
            self.mode = OProjectionMode.FULL_WEIGHT
        elif exchange_tokens:
            self.mode = OProjectionMode.EXCHANGE_TOKENS
        else:
            self.mode = OProjectionMode.LOCAL_WEIGHT

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        self.mode = OProjectionMode.LOCAL_WEIGHT
        try:
            yield
        except BaseException:
            with suppress(Exception):
                self.wait_for_weights()
            raise
        else:
            self.wait_for_weights()
        finally:
            self.mode = OProjectionMode.LOCAL_WEIGHT

    @contextmanager
    def batch(self, *, needs_global_metadata: bool, has_prefill: bool) -> Iterator[None]:
        with self.forward_context():
            self.enable()
            exchange_tokens = needs_global_metadata and self.decode_sharded and not has_prefill
            self.set_mode(
                use_full_weight=needs_global_metadata and not exchange_tokens, exchange_tokens=exchange_tokens
            )
            yield

    def _require_weight_switches(self) -> tuple[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState], ...]:
        if self._weight_switch_states is None:
            raise RuntimeError("O-proj weight switching has not been prepared.")
        return self._weight_switch_states

    def all_gather_weights(self) -> None:
        if not self.uses_full_weight:
            return
        switches = self._require_weight_switches()
        try:
            for _, method, state in switches:
                method.all_gather_weight(state, self.config)
        except Exception:
            with suppress(Exception):
                self.wait_for_weights()
            raise

    def wait_for_weights(self) -> None:
        first_error: Exception | None = None
        for _, method, state in self._weight_switch_states or ():
            if not state.handles:
                continue
            try:
                method.wait_weight_all_gather(state)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @contextmanager
    def use_full_weights(self) -> Iterator[None]:
        switched: list[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState]] = []
        try:
            for layer, method, state in self._require_weight_switches():
                method.wait_weight_all_gather(state)
                switched.append((layer, method, state))
                method.switch_weight(layer, state, use_full_weight=True)
            yield
        finally:
            for layer, method, state in reversed(switched):
                method.switch_weight(layer, state, use_full_weight=False)

    def _prepare_exchange_buffers(self, vllm_config: "VllmConfig") -> None:
        """Allocate separate, graph-stable activation buffers before capture.

        Callers may share a pool only across projections executed in sequence.
        Weight buffers remain separate from these activation buffers.
        """
        config = self.config
        dtype = vllm_config.model_config.dtype
        device = self.o_proj.weight.device
        compilation = vllm_config.compilation_config
        capacity = max(
            vllm_config.scheduler_config.max_num_seqs * (1 + vllm_config.num_speculative_tokens),
            compilation.max_cudagraph_capture_size or 0,
            max(compilation.cudagraph_capture_sizes or [0]),
        )
        self.token_capacity = capacity
        shapes = {
            "all_gather_input": (capacity, self.input_size),
            "all_gather_output": (config.world_size * capacity, self.input_size),
            "reduce_scatter_input": (config.world_size * capacity, self.output_size),
            "reduce_scatter_output": (capacity, self.output_size),
        }
        for name, shape in shapes.items():
            key = (
                self.o_proj_weight_switch_pool_key,
                "activation",
                getattr(config.group, "unique_name", None),
                name,
                device,
                dtype,
                shape,
            )
            buffer = self.o_proj_full_pools.get(key)
            if buffer is None:
                buffer = torch.empty(shape, dtype=dtype, device=device)
                self.o_proj_full_pools[key] = buffer
            setattr(self, name, buffer)

    def _gather_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        config = self.config
        num_tokens = inputs.shape[0]
        if num_tokens > self.token_capacity:
            raise ValueError("Parallel O-proj rows exceed the configured token/graph capacity.")
        if self.all_gather_input is None or self.all_gather_output is None:
            raise RuntimeError("Parallel O-proj exchange buffers must be prepared before forward.")
        local_rows = self.all_gather_input[:num_tokens]
        local_rows.copy_(inputs)
        global_rows = self.all_gather_output[: config.world_size * num_tokens]
        # Rank-major padded rows preserve the matching ReduceScatter owners.
        dist.all_gather_into_tensor(global_rows, local_rows, group=config.group.device_group)
        return global_rows

    def _scatter_tokens(self, partial_output: torch.Tensor, num_tokens: int) -> torch.Tensor:
        config = self.config
        if self.reduce_scatter_input is None or self.reduce_scatter_output is None:
            raise RuntimeError("Parallel O-proj exchange buffers must be prepared before forward.")
        scatter_input = self.reduce_scatter_input[: config.world_size * num_tokens]
        scatter_input.copy_(partial_output)
        local_output = self.reduce_scatter_output[:num_tokens]
        dist.reduce_scatter_tensor(local_output, scatter_input, group=config.group.device_group)
        return local_output

    def _finalize_output(self, partial_output: torch.Tensor) -> torch.Tensor:
        layer = self.o_proj
        if layer.reduce_results and self.tp_group.world_size > 1:
            partial_output = self.tp_group.all_reduce(partial_output)
        # A deferred TP reduction must receive bias from just one TP rank.
        # Read the layer's current bias to support online weight replacement.
        if (
            not layer.skip_bias_add
            and layer.bias is not None
            and (layer.reduce_results or self.tp_group.rank_in_group == 0)
        ):
            partial_output = partial_output + layer.bias
        return partial_output

    def _project_local(self, inputs: torch.Tensor) -> torch.Tensor:
        local_input = WeightSwitchMixin.split_tensor_for_parallel(
            inputs, self.config.world_size, self.config.rank, dim=-1
        )
        if self.input_projection is not None:
            assert self.num_input_groups is not None
            local_groups = self.num_input_groups // self.config.world_size
            group_width = self.input_size // self.num_input_groups
            grouped = local_input.reshape(local_input.shape[0], local_groups, group_width)
            local_input = self.input_projection(grouped).reshape(grouped.shape[0], self.o_proj.input_size_per_partition)
        method = self._resolve_o_proj_weight_switch_method(self.o_proj)
        bias_rank = 0 if self.tp_group.rank_in_group == 0 and self.config.rank == 0 else 1
        return method.apply_weight_shard(self.o_proj, local_input, bias_rank=bias_rank)

    def forward(self, inputs: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not self._weight_switch_enabled:
            return self.project_full(inputs, output)
        if self.uses_full_weight:
            with self.use_full_weights():
                return self.project_full(inputs, output)

        num_tokens = inputs.shape[0]
        inputs = inputs.reshape(num_tokens, self.input_size)
        if self.exchanges_tokens:
            inputs = self._gather_tokens(inputs)
            partial_output = self._project_local(inputs)
            partial_output = self._scatter_tokens(partial_output, num_tokens)
        else:
            partial_output = self._project_local(inputs)
            partial_output = self.config.group.all_reduce(partial_output)
        output.copy_(self._finalize_output(partial_output))
        return output
