# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction and commit adapters for Ascend."""

from functools import wraps
from inspect import signature

from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state

from vllm_ascend.distributed.eplb.async_worker import NO_TRANSFER_CYCLE_COMPLETE
from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.distributed.eplb.state import refresh_model_routing_tables

_PATCH_MARKER = "_vllm_ascend_eplb_patch"


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
    required_parameters = {"group_coordinator", "backend", "expert_weights", "expert_buffer"}
    if not required_parameters.issubset(factory_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory signature changed.")

    @wraps(original_factory)
    def _create_eplb_communicator(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if _is_npu_platform(_parallel_config.current_platform) and bound.arguments["backend"] == "torch_gloo":
            return AscendGlooEplbCommunicator(cpu_group=bound.arguments["group_coordinator"].cpu_group)
        return original_factory(*args, **kwargs)

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
        if pending_result is not None and pending_result.transfer_metadata is NO_TRANSFER_CYCLE_COMPLETE:
            model_state.rebalanced = False
            model_state.pending_result = None
            pending_result.consumed_event.record()
            return None

        result = original_move(*bound.args, **bound.kwargs)
        if layer_idx is not None:
            refresh_model_routing_tables(model_state, layer_idx)
            state = getattr(model_state, "_ascend_eplb_state", None)
            if state is not None:
                state.commit_policy_layer(model_state, layer_idx)
        return result

    setattr(_move_to_workspace, _PATCH_MARKER, True)
    return _move_to_workspace


def _patch_async_move_to_workspace() -> None:
    original_move = _eplb_state._move_to_workspace
    if not getattr(original_move, _PATCH_MARKER, False):
        _eplb_state._move_to_workspace = _wrap_move_to_workspace(original_move)


_patch_parallel_config()
_patch_communicator_factory()
_patch_async_move_to_workspace()
