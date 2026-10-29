# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Common Model Runner V2 pipeline-parallel utilities."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.utils import should_reuse_topk

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_PP_TRANSPORT_PREFIX = "pp_transport"


class _PPTransportModel(Protocol):
    config: "PretrainedConfig"
    start_layer: int
    end_layer: int
    aux_hidden_state_layers: tuple[int, ...]
    topk_indices_buffer: torch.Tensor | None
    receive_pp_topk_indices: bool
    send_pp_topk_indices: bool


@dataclass(frozen=True)
class SpecPPSupport:
    """Capabilities for one speculative decoding method under PP."""

    architectures: frozenset[str] | None = None
    needs_aux_hidden_states: bool = False


_SPEC_PP_SUPPORT_BY_METHOD: Mapping[str, SpecPPSupport] = MappingProxyType(
    {
        "mtp": SpecPPSupport(),
        "eagle3": SpecPPSupport(
            architectures=frozenset(
                {
                    "MiniMaxM3SparseForCausalLM",
                    "MiniMaxM3SparseForConditionalGeneration",
                }
            ),
            needs_aux_hidden_states=True,
        ),
        "dspark": SpecPPSupport(
            architectures=frozenset(
                {
                    "DeepseekV4ForCausalLM",
                    "DeepseekV41ForCausalLM",
                    "GlmMoeDsaForCausalLM",
                    "KimiLinearForCausalLM",
                    "KimiK3ForCausalLM",
                    "KimiK3ForConditionalGeneration",
                }
            ),
            needs_aux_hidden_states=True,
        ),
    }
)


def resolve_spec_pp_support(vllm_config: VllmConfig) -> SpecPPSupport | None:
    """Return the registered Spec+PP capabilities for this configuration."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or vllm_config.parallel_config.pipeline_parallel_size <= 1:
        return None

    support = _SPEC_PP_SUPPORT_BY_METHOD.get(speculative_config.method)
    if support is None:
        return None

    model_config = vllm_config.model_config
    if support.architectures is not None and (
        model_config is None or model_config.architecture not in support.architectures
    ):
        return None
    return support


class PPTransportDataType(str, Enum):
    """Data types carried between PP ranks via ``IntermediateTensors``."""

    AUX_HIDDEN_STATES = "aux_hidden_states"
    TOPK_INDICES = "topk_indices"


_PPTransportBufferFactory = Callable[
    [_PPTransportModel, IntermediateTensors, int, torch.dtype, torch.device, bool],
    IntermediateTensors,
]


def configure_pp_topk_transport(
    model: _PPTransportModel,
    data_types: tuple[PPTransportDataType, ...],
) -> None:
    """Initialize model state required by selected PP transports."""
    if PPTransportDataType.TOPK_INDICES not in data_types:
        return

    topk_indices_buffer = model.topk_indices_buffer
    model.receive_pp_topk_indices = topk_indices_buffer is not None and pp_stage_requires_topk_indices(
        model.config, model.start_layer
    )
    model.send_pp_topk_indices = topk_indices_buffer is not None and pp_stage_requires_topk_indices(
        model.config, model.end_layer
    )


def _add_aux_hidden_state_buffers(
    model: _PPTransportModel,
    intermediate_tensors: IntermediateTensors,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    include_start_layer: bool,
) -> IntermediateTensors:
    num_incoming_aux_layers = sum(
        layer_idx < model.start_layer or (include_start_layer and layer_idx == model.start_layer)
        for layer_idx in model.aux_hidden_state_layers
    )
    return add_pp_transport_buffers(
        intermediate_tensors,
        PPTransportDataType.AUX_HIDDEN_STATES,
        num_incoming_aux_layers,
        (batch_size, model.config.hidden_size),
        dtype,
        device,
    )


def _add_topk_indices_buffer(
    model: _PPTransportModel,
    intermediate_tensors: IntermediateTensors,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    include_start_layer: bool,
) -> IntermediateTensors:
    del dtype, device, include_start_layer
    if not model.receive_pp_topk_indices:
        return intermediate_tensors

    topk_indices_buffer = model.topk_indices_buffer
    assert topk_indices_buffer is not None
    if batch_size > topk_indices_buffer.shape[0]:
        raise ValueError(
            "PP Top-K receive buffer exceeds the model buffer capacity: "
            f"requested {batch_size} tokens, capacity {topk_indices_buffer.shape[0]}."
        )
    return add_pp_transport_tensors(
        intermediate_tensors,
        PPTransportDataType.TOPK_INDICES,
        [topk_indices_buffer[:batch_size]],
    )


_PP_TRANSPORT_BUFFER_FACTORIES: Mapping[PPTransportDataType, _PPTransportBufferFactory] = MappingProxyType(
    {
        PPTransportDataType.AUX_HIDDEN_STATES: _add_aux_hidden_state_buffers,
        PPTransportDataType.TOPK_INDICES: _add_topk_indices_buffer,
    }
)


def make_empty_intermediate_tensors(
    model: _PPTransportModel,
    tensor_factory: Callable[[int, torch.dtype, torch.device], IntermediateTensors],
    data_types: tuple[PPTransportDataType, ...] = (PPTransportDataType.AUX_HIDDEN_STATES,),
    *,
    include_start_layer: bool = True,
) -> Callable[[int, torch.dtype, torch.device], IntermediateTensors]:
    """Wrap a model's PP tensor factory with typed receive buffers."""
    if len(set(data_types)) != len(data_types):
        raise ValueError(f"Duplicate PP transport data types: {data_types}.")
    unsupported_data_types = set(data_types) - _PP_TRANSPORT_BUFFER_FACTORIES.keys()
    if unsupported_data_types:
        raise ValueError(f"Unsupported PP transport data types: {unsupported_data_types}.")

    def wrapped_tensor_factory(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        intermediate_tensors = tensor_factory(batch_size, dtype, device)
        for data_type in data_types:
            intermediate_tensors = _PP_TRANSPORT_BUFFER_FACTORIES[data_type](
                model,
                intermediate_tensors,
                batch_size,
                dtype,
                device,
                include_start_layer,
            )
        return intermediate_tensors

    return wrapped_tensor_factory


def _get_transport_key_prefix(data_type: PPTransportDataType) -> str:
    return f"{_PP_TRANSPORT_PREFIX}_{data_type.value}_"


def get_pp_transport_tensors(
    intermediate_tensors: IntermediateTensors | None,
    data_type: PPTransportDataType,
) -> list[torch.Tensor]:
    """Return tensors of one transport type in their original order."""
    if intermediate_tensors is None:
        return []

    key_prefix = _get_transport_key_prefix(data_type)
    indexed_tensors = [
        (int(key.removeprefix(key_prefix)), tensor)
        for key, tensor in intermediate_tensors.tensors.items()
        if key.startswith(key_prefix)
    ]
    indexed_tensors.sort(key=lambda item: item[0])
    return [tensor for _, tensor in indexed_tensors]


def add_pp_transport_tensors(
    intermediate_tensors: IntermediateTensors,
    data_type: PPTransportDataType,
    tensors: Sequence[torch.Tensor],
) -> IntermediateTensors:
    """Add tensors of one transport type to a PP payload."""
    key_prefix = _get_transport_key_prefix(data_type)
    for index, tensor in enumerate(tensors):
        intermediate_tensors.tensors[f"{key_prefix}{index}"] = tensor
    return intermediate_tensors


def add_pp_transport_buffers(
    intermediate_tensors: IntermediateTensors,
    data_type: PPTransportDataType,
    count: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    """Add empty receive buffers for one PP transport data type."""
    tensors = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(count)]
    return add_pp_transport_tensors(intermediate_tensors, data_type, tensors)


def pp_stage_requires_topk_indices(config: object, start_layer: int) -> bool:
    """Return whether a PP stage needs Top-K indices from its predecessor."""
    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    if start_layer <= 0 or start_layer >= num_hidden_layers:
        return False

    indexer_types = getattr(config, "indexer_types", None)
    uses_shared_indexer = (
        indexer_types is not None
        and start_layer < len(indexer_types)
        and indexer_types[start_layer].lower() == "shared"
    )
    uses_index_cache = bool(getattr(config, "use_index_cache", False)) and should_reuse_topk(config, start_layer)
    return uses_shared_indexer or uses_index_cache


@dataclass(frozen=True)
class PPTransportTensorSpec:
    """A token-major wire buffer; the token count is supplied per forward."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def allocate(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_tokens, *self.shape), dtype=self.dtype, device=device)


def gather_pp_cache_rows(cache: torch.Tensor, slots: torch.Tensor, output: torch.Tensor) -> None:
    """Gather one native cache plane directly into its PP wire buffer."""
    # Load Triton helpers only when a worker actually exports cache rows.
    from vllm.triton_utils import triton

    from vllm_ascend.ops.triton.pcp_kv_cache import _copy_pcp_kv_cache_kernel, _get_pcp_kv_cache_rows
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

    if cache.ndim == 3:
        cache = cache.unsqueeze(2)
    assert cache.ndim == 4 and cache.shape[2] == 1
    assert output.shape == (slots.numel(), cache.shape[-1])
    assert output.dtype == cache.dtype
    slots = slots.contiguous()
    if not slots.numel():
        return

    init_device_properties_triton()
    num_cores = get_vectorcore_num()
    block_cols = triton.next_power_of_2(cache.shape[-1])
    rows = _get_pcp_kv_cache_rows(slots.numel(), num_cores, block_cols, cache.element_size(), 1)
    _copy_pcp_kv_cache_kernel[(min(triton.cdiv(slots.numel(), rows), num_cores),)](
        cache,
        cache,  # Unused RoPE pointer: this transfer copies one complete plane.
        slots,
        output,
        slots.numel(),
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        cache.shape[-1],
        0,
        block_cols,
        rows,
    )


def pp_cache_write_metadata(metadata):
    """PCP queries are local, but replicated cache writes use global slots."""
    global_metadata = getattr(metadata, "global_metadata", None)
    return global_metadata if global_metadata is not None else metadata


def export_pp_cache_rows(cache_planes: Sequence[tuple[str, int | None]], rows: Sequence[torch.Tensor]) -> None:
    """Gather model-declared cache planes using the current forward's slots."""
    context = get_forward_context()
    for (prefix, plane), output in zip(cache_planes, rows):
        cache = context.no_compile_layers[prefix].kv_cache[0]
        if plane is not None:
            cache = cache[plane]
        output = output.view(-1, output.shape[-1])
        slots = pp_cache_write_metadata(context.attn_metadata[prefix]).flat_slot_mapping
        assert slots.shape[0] <= output.shape[0], "PP cache payload cannot hold all cache writes"
        gather_pp_cache_rows(cache, slots, output[: slots.shape[0]])


def import_pp_cache_rows(
    cache_planes: Sequence[tuple[str, int | None]],
    rows: Sequence[torch.Tensor],
    scatter_rows: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
) -> None:
    """Restore native cache planes while preserving the model's scatter ABI."""
    context = get_forward_context()
    for (prefix, plane), values in zip(cache_planes, rows):
        cache = context.no_compile_layers[prefix].kv_cache[0]
        if plane is not None:
            cache = cache[plane]
        values = values.view(-1, values.shape[-1])
        slots = pp_cache_write_metadata(context.attn_metadata[prefix]).slot_mapping
        assert slots.shape[0] <= values.shape[0], "PP cache payload cannot hold all cache writes"
        scatter_rows(cache, slots, values[: slots.shape[0]])


def _export_dependencies(
    hidden_states: torch.Tensor,
    cache_rows: list[torch.Tensor],
    state_inputs: list[torch.Tensor],
    state_outputs: list[torch.Tensor],
    transport_name: str,
) -> None:
    # Dummy/profile traces have no metadata. Resolve the actual cache and
    # slots at execution time, so compilation cannot erase the transfer.
    context = get_forward_context() if is_forward_context_available() else None
    if cache_rows and context is not None and context.attn_metadata is not None:
        context.no_compile_layers[transport_name].cache_handler.export_rows(cache_rows)
    n = hidden_states.shape[0]
    for source, output in zip(state_inputs, state_outputs):
        output.copy_(source[:n])


def _import_dependencies(
    hidden_states: torch.Tensor,
    cache_rows: list[torch.Tensor],
    state_inputs: list[torch.Tensor],
    state_outputs: list[torch.Tensor],
    transport_name: str,
) -> None:
    context = get_forward_context() if is_forward_context_available() else None
    if cache_rows and context is not None and context.attn_metadata is not None:
        context.no_compile_layers[transport_name].cache_handler.import_rows(cache_rows)
    n = hidden_states.shape[0]
    for source, output in zip(state_inputs, state_outputs):
        output[:n].copy_(source)


def _dependency_fake(*args, **kwargs) -> None:
    return None


# hidden_states is an explicit producer/consumer ordering edge. Cache writes
# live behind the operator, as attention's own cache updates do. Marking the
# incoming activation as mutated orders downstream attention after import;
# its values are unchanged. Tensor lists carry the exact boundary manifest.
direct_register_custom_op(
    op_name="pp_dependency_export",
    op_func=eager_break_during_capture(_export_dependencies),
    mutates_args=["cache_rows", "state_outputs"],
    fake_impl=_dependency_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="pp_dependency_import",
    op_func=eager_break_during_capture(_import_dependencies),
    mutates_args=["hidden_states", "state_outputs"],
    fake_impl=_dependency_fake,
    dispatch_key="PrivateUse1",
)


class PPDependencyTransport(torch.nn.Module):
    """Transport a boundary's complete cache and per-forward dependencies.

    The model supplies its cache ABI and incoming/outgoing manifests. The
    common PP path owns receive allocation, validation, snapshots and runtime
    ordering, and carries all payloads in normal ``IntermediateTensors``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        name: str,
        cache_handler: torch.nn.Module,
        incoming_caches: tuple[PPTransportTensorSpec, ...],
        outgoing_caches: tuple[PPTransportTensorSpec, ...],
        incoming_states: Mapping[str, torch.Tensor],
        outgoing_states: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.name = name
        self.cache_handler = cache_handler
        self.incoming_caches = incoming_caches
        self.outgoing_caches = outgoing_caches
        self.incoming_states = dict(incoming_states)
        self.outgoing_states = dict(outgoing_states)
        self.incoming_specs = incoming_caches + self._state_specs(incoming_states)
        self.outgoing_specs = outgoing_caches + self._state_specs(outgoing_states)
        for specs in (self.incoming_specs, self.outgoing_specs):
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate PP dependency tensor names: {names}")
        vllm_config.compilation_config.static_forward_context[name] = self

    @staticmethod
    def _state_specs(states: Mapping[str, torch.Tensor]) -> tuple[PPTransportTensorSpec, ...]:
        return tuple(
            PPTransportTensorSpec(name, tuple(tensor.shape[1:]), tensor.dtype) for name, tensor in states.items()
        )

    def add_receive_buffers(self, intermediate: IntermediateTensors, num_tokens: int, device: torch.device):
        for spec in self.incoming_specs:
            intermediate.tensors[spec.name] = spec.allocate(num_tokens, device)
        return intermediate

    def receive(self, intermediate: IntermediateTensors, num_tokens: int) -> None:
        specs = self.incoming_specs
        if not specs:
            return
        for spec in specs:
            value = intermediate[spec.name]
            if value.shape != (num_tokens, *spec.shape) or value.dtype != spec.dtype:
                raise ValueError(f"Invalid PP dependency tensor {spec.name}: {value.shape}, {value.dtype}")
        torch.ops.vllm.pp_dependency_import(
            intermediate["hidden_states"],
            [intermediate[spec.name] for spec in self.incoming_caches],
            [intermediate[name] for name in self.incoming_states],
            list(self.incoming_states.values()),
            self.name,
        )

    def send(self, intermediate: IntermediateTensors, num_tokens: int):
        specs = self.outgoing_specs
        if not specs:
            return intermediate
        device = intermediate["hidden_states"].device
        payload = {spec.name: spec.allocate(num_tokens, device) for spec in specs}
        torch.ops.vllm.pp_dependency_export(
            intermediate["hidden_states"],
            [payload[spec.name] for spec in self.outgoing_caches],
            list(self.outgoing_states.values()),
            [payload[name] for name in self.outgoing_states],
            self.name,
        )
        intermediate.tensors.update(payload)
        return intermediate
