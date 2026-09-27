# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction and commit adapters for Ascend."""

from collections.abc import Sequence
from functools import wraps
from inspect import signature

from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import routed_experts as _routed_experts

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.distributed.eplb.state import (
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    EXPERT_MAPPING_EP_SIZE,
    refresh_model_routing_tables,
)

_PATCH_MARKER = "_vllm_ascend_eplb_patch"


class _DeferredConsumedEvent:
    """Delay the worker acknowledgement until Ascend commit hooks finish."""

    def __init__(self, consumed_event) -> None:
        self._consumed_event = consumed_event
        self._recorded = False
        self._stream = None

    def record(self, stream=None) -> None:
        if self._recorded:
            raise RuntimeError("EPLB result consumption was acknowledged more than once.")
        self._recorded = True
        self._stream = stream

    def flush(self) -> None:
        if not self._recorded:
            raise RuntimeError("Upstream EPLB workspace move did not acknowledge the pending result.")
        self._consumed_event.record(self._stream)


class _CudaAlikeEplbPlatformProxy:
    """Delegate platform operations while exposing EPLB validation capability."""

    def __init__(self, platform) -> None:
        self._platform = platform

    def is_cuda_alike(self) -> bool:
        return _is_npu_platform(self._platform) or self._platform.is_cuda_alike()

    def __getattr__(self, name):
        return getattr(self._platform, name)


def _is_npu_platform(platform) -> bool:
    return getattr(platform, "device_type", None) == "npu"


def _patch_parallel_config() -> None:
    platform = _parallel_config.current_platform
    if not isinstance(platform, _CudaAlikeEplbPlatformProxy):
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)


def _wrap_communicator_factory(original_factory):
    factory_signature = signature(original_factory)
    if "group_coordinator" not in factory_signature.parameters:
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory has no group_coordinator parameter.")

    @wraps(original_factory)
    def _create_eplb_communicator(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        return AscendGlooEplbCommunicator(
            cpu_group=bound.arguments["group_coordinator"].cpu_group,
        )

    setattr(_create_eplb_communicator, _PATCH_MARKER, True)
    return _create_eplb_communicator


def _patch_communicator_factory() -> None:
    original_factory = _eplb_communicator.create_eplb_communicator
    if getattr(original_factory, _PATCH_MARKER, False):
        return
    wrapped_factory = _wrap_communicator_factory(original_factory)
    _eplb_communicator.create_eplb_communicator = wrapped_factory
    _eplb_state.create_eplb_communicator = wrapped_factory


def _build_distributed_initial_expert_map(
    num_routed_experts: int,
    num_redundant_experts: int,
    ep_size: int | None = None,
) -> Sequence[int]:
    """Spread initial redundant experts across EP ranks."""
    if ep_size is None:
        ep_size = EXPERT_MAPPING_EP_SIZE.get()
    if num_redundant_experts == 0:
        return list(range(num_routed_experts))
    num_physical_experts = num_routed_experts + num_redundant_experts
    if num_routed_experts < 1 or ep_size < 1 or num_physical_experts % ep_size:
        raise ValueError("Physical experts must be divisible by a positive EP size")

    slots_per_rank = num_physical_experts // ep_size
    result: list[int] = []
    primary_begin = 0
    for rank in range(ep_size):
        redundant_count = num_redundant_experts // ep_size + (rank < num_redundant_experts % ep_size)
        primary_end = primary_begin + slots_per_rank - redundant_count
        result.extend(range(primary_begin, primary_end))
        result.extend((primary_end + index) % num_routed_experts for index in range(redundant_count))
        primary_begin = primary_end
    return result


def _with_expert_mapping_ep_size(original, ep_size_getter):
    @wraps(original)
    def wrapped(*args, **kwargs):
        token = EXPERT_MAPPING_EP_SIZE.set(ep_size_getter(*args, **kwargs))
        try:
            return original(*args, **kwargs)
        finally:
            EXPERT_MAPPING_EP_SIZE.reset(token)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch_initial_expert_layout() -> None:
    build_map = _eplb_state.EplbState.build_initial_global_physical_to_logical_map
    if "ep_size" in signature(build_map).parameters:
        return
    _eplb_state.EplbState.build_initial_global_physical_to_logical_map = staticmethod(
        _build_distributed_initial_expert_map
    )
    routed_experts = _routed_experts.RoutedExperts

    original_get = routed_experts.get_expert_mapping
    if not getattr(original_get, _PATCH_MARKER, False):
        routed_experts.get_expert_mapping = _with_expert_mapping_ep_size(
            original_get,
            lambda self, *_args, **_kwargs: self.moe_config.ep_size
            if getattr(self, "_use_v2_model_runner", False)
            else 1,
        )

    original_make = routed_experts.make_expert_params_mapping
    if not getattr(original_make, _PATCH_MARKER, False):
        make_signature = signature(original_make)

        def model_ep_size(*args, **kwargs):
            bound = make_signature.bind(*args, **kwargs)
            if not bound.arguments.get("num_redundant_experts", 0):
                return 1
            model = bound.arguments["model"]
            ep_sizes = {
                module.moe_config.ep_size
                for module in model.modules()
                if isinstance(module, routed_experts) and getattr(module, "_use_v2_model_runner", False)
            }
            if len(ep_sizes) > 1:
                raise RuntimeError("MRV2 redundant expert loading requires one EP size")
            return ep_sizes.pop() if ep_sizes else 1

        routed_experts.make_expert_params_mapping = staticmethod(
            _with_expert_mapping_ep_size(original_make, model_ep_size)
        )


def _wrap_move_to_workspace(original_move):
    move_signature = signature(original_move)
    if not {"model_state", "ep_rank"}.issubset(move_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: async workspace move signature changed.")

    @wraps(original_move)
    def _move_to_workspace(*args, **kwargs):
        bound = move_signature.bind(*args, **kwargs)
        model_state = bound.arguments["model_state"]
        pending_result = model_state.pending_result
        layer_idx = pending_result.layer_idx if pending_result is not None else None

        deferred_event = None
        consumed_event = None
        if pending_result is not None:
            consumed_event = pending_result.consumed_event
            deferred_event = _DeferredConsumedEvent(consumed_event)
            pending_result.consumed_event = deferred_event
        try:
            result = original_move(*bound.args, **bound.kwargs)
            if layer_idx is not None:
                refresh_model_routing_tables(model_state, layer_idx)
                if bound.arguments["ep_rank"] == 0 and layer_idx == model_state.model.num_moe_layers - 1:
                    logger.info(
                        "%s: model=%s",
                        ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                        model_state.model_name,
                    )
        finally:
            if pending_result is not None and consumed_event is not None:
                pending_result.consumed_event = consumed_event
        if deferred_event is not None:
            deferred_event.flush()
        return result

    setattr(_move_to_workspace, _PATCH_MARKER, True)
    return _move_to_workspace


def _patch_async_move_to_workspace() -> None:
    original_move = _eplb_state._move_to_workspace
    if not getattr(original_move, _PATCH_MARKER, False):
        _eplb_state._move_to_workspace = _wrap_move_to_workspace(original_move)


_patch_parallel_config()
_patch_initial_expert_layout()
_patch_communicator_factory()
_patch_async_move_to_workspace()
