# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Common Model Runner V2 pipeline-parallel utilities."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

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
