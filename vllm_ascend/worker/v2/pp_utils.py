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

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_PP_TRANSPORT_PREFIX = "pp_transport"


class _PPAuxHiddenStateModel(Protocol):
    config: "PretrainedConfig"
    start_layer: int
    aux_hidden_state_layers: tuple[int, ...]


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
            architectures=frozenset({"DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM"}),
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


def make_empty_intermediate_tensors(
    model: _PPAuxHiddenStateModel,
    tensor_factory: Callable[[int, torch.dtype, torch.device], IntermediateTensors],
) -> Callable[[int, torch.dtype, torch.device], IntermediateTensors]:
    """Wrap a model's PP tensor factory with auxiliary receive buffers."""

    def wrapped_tensor_factory(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        intermediate_tensors = tensor_factory(batch_size, dtype, device)
        num_incoming_aux_layers = sum(layer_idx <= model.start_layer for layer_idx in model.aux_hidden_state_layers)
        return add_pp_transport_buffers(
            intermediate_tensors,
            PPTransportDataType.AUX_HIDDEN_STATES,
            num_incoming_aux_layers,
            (batch_size, model.config.hidden_size),
            dtype,
            device,
        )

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
