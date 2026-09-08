# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction and commit adapters for Ascend."""

from functools import wraps
from inspect import signature

from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import async_worker as _async_worker
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.distributed.eplb.state import (
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    refresh_model_routing_tables,
)
from vllm_ascend.distributed.eplb.stair_worker import run_stair_planner, transfer_stair_layer

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


def _wrap_worker_planner(original):
    @wraps(original)
    def _run(model_state, eplb_state, physical_to_logical_map_cpu, cuda_stream):
        if getattr(eplb_state, "_stair_config", None) is None:
            return original(model_state, eplb_state, physical_to_logical_map_cpu, cuda_stream)
        return run_stair_planner(model_state, eplb_state, physical_to_logical_map_cpu, cuda_stream)

    setattr(_run, _PATCH_MARKER, True)
    return _run


def _wrap_worker_transfer(original):
    @wraps(original)
    def _transfer(*args, **kwargs):
        bound = signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        communicator = bound.arguments["communicator"]
        if hasattr(communicator, "_stair_source_rank") and not bound.arguments["is_profile"]:
            return transfer_stair_layer(*bound.args, **bound.kwargs)
        return original(*bound.args, **bound.kwargs)

    setattr(_transfer, _PATCH_MARKER, True)
    return _transfer


def _patch_async_worker() -> None:
    if not getattr(_async_worker.run_rebalance_experts, _PATCH_MARKER, False):
        _async_worker.run_rebalance_experts = _wrap_worker_planner(_async_worker.run_rebalance_experts)
    if not getattr(_async_worker.transfer_layer, _PATCH_MARKER, False):
        _async_worker.transfer_layer = _wrap_worker_transfer(_async_worker.transfer_layer)


_patch_parallel_config()
_patch_communicator_factory()
_patch_async_move_to_workspace()
_patch_async_worker()
