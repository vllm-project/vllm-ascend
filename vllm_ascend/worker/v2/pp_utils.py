# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Common Model Runner V2 pipeline-parallel utilities."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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
    bypass_upstream_pp_guard: bool = False
    unsupported_feature: str | None = None


_SPEC_PP_SUPPORT_BY_METHOD: Mapping[str, SpecPPSupport] = MappingProxyType({
    "mtp": SpecPPSupport(),
    "eagle3": SpecPPSupport(
        architectures=frozenset(
            {
                "MiniMaxM3SparseForCausalLM",
                "MiniMaxM3SparseForConditionalGeneration",
            }
        ),
        needs_aux_hidden_states=True,
        bypass_upstream_pp_guard=True,
        unsupported_feature="EAGLE3 with pipeline parallelism",
    ),
    "dspark": SpecPPSupport(
        architectures=frozenset({"DeepseekV4ForCausalLM"}),
        needs_aux_hidden_states=True,
        bypass_upstream_pp_guard=True,
    ),
})


def resolve_spec_pp_support(vllm_config: VllmConfig) -> SpecPPSupport | None:
    """Return the registered Spec+PP capabilities for this configuration."""
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or vllm_config.parallel_config.pipeline_parallel_size <= 1
    ):
        return None

    support = _SPEC_PP_SUPPORT_BY_METHOD.get(speculative_config.method)
    if support is None:
        return None

    model_config = vllm_config.model_config
    if support.architectures is not None and (
        model_config is None
        or model_config.architecture not in support.architectures
    ):
        return None
    return support


@contextmanager
def bypass_upstream_spec_pp_guard(
    vllm_config: VllmConfig,
    support: SpecPPSupport | None,
) -> Iterator[bool]:
    """Initialize the upstream runner as PP=1 to bypass its Spec+PP guard."""
    bypass_guard = (
        support.bypass_upstream_pp_guard if support is not None else False
    )
    if not bypass_guard:
        yield False
        return

    parallel_config = vllm_config.parallel_config
    original_pp_size = parallel_config.pipeline_parallel_size
    parallel_config.pipeline_parallel_size = 1
    try:
        yield True
    finally:
        parallel_config.pipeline_parallel_size = original_pp_size


def restore_pp_after_upstream_init(
    model_runner: "GPUModelRunner",
    vllm_config: VllmConfig,
) -> None:
    """Restore PP state skipped while the upstream runner initialized as PP=1."""
    from vllm.v1.worker.gpu.buffer_utils import set_default_max_concurrency
    from vllm.v1.worker.gpu.pp_utils import PPHandler

    model_runner.use_pp = vllm_config.parallel_config.pipeline_parallel_size > 1
    assert model_runner.use_pp and model_runner.pp_handler is None

    # The parent sizes UVA pools before constructing request state and the
    # speculator. Ascend rebuilds both after this helper returns.
    set_default_max_concurrency(vllm_config.max_concurrent_batches)
    model_runner.pp_handler = PPHandler(
        max_num_reqs=model_runner.max_num_reqs,
        num_speculative_steps=model_runner.num_speculative_steps,
        device=model_runner.device,
    )


class PPTransportDataType(str, Enum):
    """Data types carried between PP ranks via ``IntermediateTensors``."""

    AUX_HIDDEN_STATES = "aux_hidden_states"


def make_empty_intermediate_tensors(
    model: _PPAuxHiddenStateModel,
    tensor_factory: Callable[
        [int, torch.dtype, torch.device], IntermediateTensors
    ],
) -> Callable[[int, torch.dtype, torch.device], IntermediateTensors]:
    """Wrap a model's PP tensor factory with auxiliary receive buffers."""

    def wrapped_tensor_factory(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        intermediate_tensors = tensor_factory(batch_size, dtype, device)
        num_incoming_aux_layers = sum(
            layer_idx <= model.start_layer
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
