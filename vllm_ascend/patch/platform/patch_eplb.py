# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction, execution, and commit adapters for Ascend."""

from functools import wraps
from inspect import signature

import torch
from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import async_worker as _async_worker
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer
from vllm_ascend.distributed.eplb.state import (
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    refresh_model_routing_tables,
)

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
# Old async APIs pass one target layer at a time. Preserve the augmented full
# target on its per-model communicator until the last workspace commit.
_EXPLICIT_TRANSFER_TARGET_ATTR = "_vllm_ascend_explicit_transfer_target"


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


def _has_explicit_sources(target) -> bool:
    return hasattr(target, "source_rank_ids") and hasattr(target, "source_slot_ids")


def _clear_transfer_target(communicator, target=None) -> None:
    if target is None or getattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, None) is target:
        communicator.__dict__.pop(_EXPLICIT_TRANSFER_TARGET_ATTR, None)


def _wrap_async_rebalance(original_rebalance):
    rebalance_signature = signature(original_rebalance)
    if "model_state" not in rebalance_signature.parameters:
        raise RuntimeError("Unsupported vLLM EPLB contract: async rebalance has no model_state parameter.")

    @wraps(original_rebalance)
    def _async_rebalance(*args, **kwargs):
        bound = rebalance_signature.bind(*args, **kwargs)
        communicator = bound.arguments["model_state"].communicator
        _clear_transfer_target(communicator)
        model_state = bound.arguments["model_state"]
        stats = model_state.eplb_stats
        original_load_window = stats.global_expert_load_window
        # The legacy async runner accepts only [layers, experts]. Temporarily
        # aggregate our [bins, layers, experts] sums, then restore the snapshot.
        if original_load_window.ndim == 3:
            with _async_worker.device_stream(bound.arguments.get("stream")):
                stats.global_expert_load_window = original_load_window.sum(dim=0)
        try:
            target = original_rebalance(*bound.args, **bound.kwargs)
        finally:
            stats.global_expert_load_window = original_load_window
        if _has_explicit_sources(target):
            setattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, target)
        return target

    setattr(_async_rebalance, _PATCH_MARKER, True)
    return _async_rebalance


def _wrap_async_transfer(original_transfer):
    transfer_signature = signature(original_transfer)
    required = {"old_layer_indices", "new_layer_indices", "expert_weights", "expert_weights_buffer"}
    required.update({"ep_group", "communicator", "is_profile", "stream", "rank_mapping", "layer_idx"})
    if not required.issubset(transfer_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: asynchronous transfer signature changed.")

    @wraps(original_transfer)
    def _async_transfer(*args, **kwargs):
        bound = transfer_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        communicator = values["communicator"]
        full_target = getattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, None)
        if not _has_explicit_sources(full_target):
            return original_transfer(*bound.args, **bound.kwargs)
        layer_idx = values["layer_idx"]
        try:
            if values["is_profile"] or values["rank_mapping"] is not None:
                return original_transfer(*bound.args, **bound.kwargs)
            layer_target = full_target[layer_idx]
            if not torch.equal(layer_target, values["new_layer_indices"]):
                raise RuntimeError("EPLB explicit transfer target does not match the current layer")
            return stage_explicit_layer_transfer(
                old_layer_indices=values["old_layer_indices"],
                new_layer_indices=layer_target,
                source_rank_ids=full_target.source_rank_ids[layer_idx],
                source_slot_ids=full_target.source_slot_ids[layer_idx],
                expert_weights=values["expert_weights"],
                expert_weight_buffers=values["expert_weights_buffer"],
                ep_group=values["ep_group"],
                communicator=communicator,
                stream=values["stream"],
                layer_idx=layer_idx,
            )
        except Exception:
            _clear_transfer_target(communicator, full_target)
            raise

    setattr(_async_transfer, _PATCH_MARKER, True)
    return _async_transfer


def _patch_explicit_transfer_execution() -> None:
    original_rebalance = _async_worker.run_rebalance_experts
    if not getattr(original_rebalance, _PATCH_MARKER, False):
        _async_worker.run_rebalance_experts = _wrap_async_rebalance(original_rebalance)
    original_async_transfer = _async_worker.transfer_layer
    if not getattr(original_async_transfer, _PATCH_MARKER, False):
        _async_worker.transfer_layer = _wrap_async_transfer(original_async_transfer)


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
                is_last_layer = layer_idx == model_state.model.num_moe_layers - 1
                if is_last_layer:
                    _clear_transfer_target(model_state.communicator)
                if bound.arguments["ep_rank"] == 0 and is_last_layer:
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
_patch_communicator_factory()
_patch_explicit_transfer_execution()
_patch_async_move_to_workspace()
