# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction and commit adapters for Ascend."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any

import torch
from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import async_worker as _async_worker
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.distributed.eplb import rebalance_execute as _rebalance_execute
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import routed_experts as _routed_experts

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.distributed.eplb.state import (
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    EXPERT_MAPPING_EP_SIZE,
    refresh_model_routing_tables,
)

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
# Older vLLM releases do not pass EP size through every weight-mapping API.


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
    """Backport vLLM's EP-rank-aware initial redundant-expert layout."""
    if ep_size is None:
        ep_size = EXPERT_MAPPING_EP_SIZE.get()
    if num_redundant_experts == 0:
        return list(range(num_routed_experts))

    num_physical_experts = num_routed_experts + num_redundant_experts
    if ep_size <= 0 or num_physical_experts % ep_size != 0:
        raise ValueError(
            f"The number of physical experts must be divisible by ep_size. Got {num_physical_experts=} and {ep_size=}."
        )

    num_local_experts = num_physical_experts // ep_size
    redundant_counts = [
        num_redundant_experts // ep_size + (rank < num_redundant_experts % ep_size) for rank in range(ep_size)
    ]
    primary_counts = [num_local_experts - redundant_count for redundant_count in redundant_counts]

    result: list[int] = []
    primary_begin = 0
    for primary_count, redundant_count in zip(primary_counts, redundant_counts):
        primary_end = primary_begin + primary_count
        local_experts = list(range(primary_begin, primary_end))
        local_experts.extend((primary_end + index) % num_routed_experts for index in range(redundant_count))
        result.extend(local_experts)
        primary_begin = primary_end
    return result


def _with_expert_mapping_ep_size(original, ep_size_getter):
    @wraps(original)
    def _wrapped(*args, **kwargs):
        token = EXPERT_MAPPING_EP_SIZE.set(ep_size_getter(*args, **kwargs))
        try:
            return original(*args, **kwargs)
        finally:
            EXPERT_MAPPING_EP_SIZE.reset(token)

    setattr(_wrapped, _PATCH_MARKER, True)
    return _wrapped


def _patch_initial_expert_layout() -> None:
    build_map = _eplb_state.EplbState.build_initial_global_physical_to_logical_map
    if "ep_size" in signature(build_map).parameters:
        return

    _eplb_state.EplbState.build_initial_global_physical_to_logical_map = staticmethod(
        _build_distributed_initial_expert_map
    )
    routed_experts = _routed_experts.RoutedExperts

    original_build = routed_experts.build_expert_params_mapping
    if not getattr(original_build, _PATCH_MARKER, False):

        def _build_expert_params_mapping(
            *args,
            ep_size: int | None = None,
            **kwargs,
        ):
            selected_ep_size = EXPERT_MAPPING_EP_SIZE.get() if ep_size is None else ep_size
            token = EXPERT_MAPPING_EP_SIZE.set(selected_ep_size)
            try:
                return original_build(*args, **kwargs)
            finally:
                EXPERT_MAPPING_EP_SIZE.reset(token)

        setattr(_build_expert_params_mapping, _PATCH_MARKER, True)
        routed_experts.build_expert_params_mapping = staticmethod(_build_expert_params_mapping)

    original_get = routed_experts.get_expert_mapping
    if not getattr(original_get, _PATCH_MARKER, False):
        routed_experts.get_expert_mapping = _with_expert_mapping_ep_size(
            original_get,
            lambda self, *_args, **_kwargs: self.moe_config.ep_size,
        )

    original_make = routed_experts.make_expert_params_mapping
    if not getattr(original_make, _PATCH_MARKER, False):
        make_signature = signature(original_make)

        def _model_ep_size(*args, **kwargs):
            bound = make_signature.bind(*args, **kwargs)
            model = bound.arguments["model"]
            ep_sizes = {module.moe_config.ep_size for module in model.modules() if isinstance(module, routed_experts)}
            num_redundant_experts = bound.arguments["num_redundant_experts"]
            if num_redundant_experts > 0 and len(ep_sizes) != 1:
                raise RuntimeError(
                    "Exactly one expert-parallel size must be available when "
                    "loading redundant expert weights, but found "
                    f"{sorted(ep_sizes)}."
                )
            return ep_sizes.pop() if ep_sizes else 1

        routed_experts.make_expert_params_mapping = staticmethod(
            _with_expert_mapping_ep_size(original_make, _model_ep_size)
        )


@dataclass
class _BackportedAsyncEplbLayerResult:
    layer_idx: int | None
    new_physical_to_logical_map: torch.Tensor | None
    transfer_metadata: Any | None
    consumed_event: Any
    is_last_result: bool


def _get_changed_layer_indices(
    old_mapping: torch.Tensor,
    new_mapping: torch.Tensor,
) -> list[int]:
    if old_mapping.shape != new_mapping.shape:
        raise ValueError("Old and new EPLB mappings must have the same shape")
    return torch.nonzero(torch.any(old_mapping != new_mapping, dim=1), as_tuple=False).flatten().tolist()


def _backported_transfer_run_periodically(
    state,
    cuda_stream,
    is_profile: bool = False,
) -> None:
    while True:
        state.rearrange_event.wait(stream=cuda_stream)
        eplb_group = _async_worker.get_eplb_group().device_group
        eplb_cpu_group = _async_worker.get_eplb_group().cpu_group
        ep_rank = eplb_group.rank()

        assert state.is_async
        for model_state in state.model_states.values():
            model_state.communicator.set_stream(cuda_stream)
            with torch.cuda.stream(cuda_stream):
                old_mapping = model_state.physical_to_logical_map.cpu()
            new_mapping = _async_worker.run_rebalance_experts(model_state, state, old_mapping, cuda_stream)
            changed_layers = _get_changed_layer_indices(old_mapping, new_mapping)
            if ep_rank == 0:
                _async_worker.logger.info(
                    "async EPLB worker: transferring %d changed layers: %s",
                    len(changed_layers),
                    changed_layers,
                )

            if not changed_layers:
                consumed_event = _async_worker.CpuGpuEvent()
                model_state.pending_result = _BackportedAsyncEplbLayerResult(
                    layer_idx=None,
                    new_physical_to_logical_map=None,
                    transfer_metadata=None,
                    consumed_event=consumed_event,
                    is_last_result=True,
                )
                consumed_event.wait(stream=cuda_stream)
                assert model_state.pending_result is None
                continue

            for changed_idx, layer_idx in enumerate(changed_layers):
                flag = torch.tensor(
                    [int(model_state.rebalanced)],
                    dtype=torch.int32,
                    device="cpu",
                )
                torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                if int(flag.item()) != eplb_cpu_group.size():
                    _async_worker.logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop (flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        int(flag.item()),
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    break

                metadata = _async_worker.transfer_layer(
                    old_layer_indices=old_mapping[layer_idx],
                    new_layer_indices=new_mapping[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                    layer_idx=layer_idx,
                )
                cuda_stream.synchronize()
                consumed_event = _async_worker.CpuGpuEvent()
                model_state.pending_result = _BackportedAsyncEplbLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_mapping[layer_idx],
                    transfer_metadata=metadata,
                    consumed_event=consumed_event,
                    is_last_result=changed_idx == len(changed_layers) - 1,
                )
                consumed_event.wait(stream=cuda_stream)
                assert model_state.pending_result is None


def _backported_move_to_workspace(model_state, ep_rank: int) -> None:
    result = model_state.pending_result
    assert result is not None
    if result.layer_idx is not None:
        assert result.transfer_metadata is not None
        assert result.new_physical_to_logical_map is not None
        _rebalance_execute.move_from_buffer(
            expert_weights=model_state.model.expert_weights[result.layer_idx],
            expert_weights_buffers=model_state.expert_buffer,
            transfer_metadata=result.transfer_metadata,
            new_indices=result.new_physical_to_logical_map.numpy(),
            ep_rank=ep_rank,
        )
        _eplb_state._commit_eplb_maps_for_layer(
            model_state,
            new_physical_to_logical_map=result.new_physical_to_logical_map,
            layer=result.layer_idx,
        )
    if result.is_last_result:
        model_state.rebalanced = False
        owner = getattr(model_state, "_ascend_eplb_owner", None)
        if owner is not None and not any(state.rebalanced for state in owner.model_states.values()):
            owner._async_cycle_in_progress = False
            owner.expert_rearrangement_step = 0
    model_state.pending_result = None
    result.consumed_event.record()


def _backported_drain_async(self) -> None:
    if not self.is_async:
        return
    for model_key, model_state in self.model_states.items():
        needs_drain = model_state.rebalanced
        if needs_drain:
            _eplb_state.logger.info("Draining async EPLB worker for model %s", model_key)
        while model_state.rebalanced:
            if self._all_ranks_result_ready(model_state):
                result = model_state.pending_result
                assert result is not None
                if result.is_last_result:
                    model_state.rebalanced = False
                model_state.pending_result = None
                result.consumed_event.record()
            else:
                _eplb_state.time.sleep(0.001)
        if needs_drain:
            _eplb_state.logger.info("Async EPLB worker drained for model %s", model_key)
    self._async_cycle_in_progress = False
    self.expert_rearrangement_step = 0


def _patch_async_noop_cycle() -> None:
    result_fields = signature(_rebalance_execute.AsyncEplbLayerResult).parameters
    if "is_last_result" in result_fields:
        return
    _rebalance_execute.AsyncEplbLayerResult = _BackportedAsyncEplbLayerResult
    _async_worker.AsyncEplbLayerResult = _BackportedAsyncEplbLayerResult
    _eplb_state.AsyncEplbLayerResult = _BackportedAsyncEplbLayerResult
    _async_worker._get_changed_layer_indices = _get_changed_layer_indices
    _async_worker.transfer_run_periodically = _backported_transfer_run_periodically
    _eplb_state._move_to_workspace = _backported_move_to_workspace
    _eplb_state.EplbState.drain_async = _backported_drain_async


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
        is_last_result = (
            getattr(
                pending_result,
                "is_last_result",
                layer_idx == model_state.model.num_moe_layers - 1,
            )
            if pending_result is not None
            else False
        )

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
            if bound.arguments["ep_rank"] == 0 and is_last_result:
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
_patch_async_noop_cycle()
_patch_async_move_to_workspace()
