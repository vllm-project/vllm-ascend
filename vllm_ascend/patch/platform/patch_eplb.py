# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Patch the remaining vLLM EPLB construction points for Ascend."""

from functools import wraps
from inspect import signature

from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import eplb_state as _eplb_state

from vllm_ascend.distributed.eplb_state import refresh_model_lookups

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
        # This module-local reference is read when ParallelConfig validates
        # EPLB. Communicator selection remains an NPUPlatform responsibility.
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)


def _wrap_move_to_workspace(original_move):
    move_signature = signature(original_move)
    required_parameters = {"model_state", "ep_rank"}
    if not required_parameters.issubset(move_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: async workspace move signature changed.")

    @wraps(original_move)
    def _move_to_workspace(*args, **kwargs):
        bound = move_signature.bind(*args, **kwargs)
        model_state = bound.arguments["model_state"]
        pending_result = model_state.pending_result
        layer_idx = pending_result.layer_idx if pending_result is not None else None
        result = original_move(*bound.args, **bound.kwargs)
        if layer_idx is not None:
            refresh_model_lookups(model_state, layer_idx)
        return result

    setattr(_move_to_workspace, _PATCH_MARKER, True)
    return _move_to_workspace


def _patch_async_move_to_workspace() -> None:
    original_move = _eplb_state._move_to_workspace
    if not getattr(original_move, _PATCH_MARKER, False):
        _eplb_state._move_to_workspace = _wrap_move_to_workspace(original_move)


_patch_parallel_config()
_patch_async_move_to_workspace()
