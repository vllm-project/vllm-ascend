# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction, execution, and commit adapters for Ascend."""

from collections.abc import Sequence
from contextlib import nullcontext
from functools import wraps
from inspect import signature

import numpy as np
import torch
from vllm.distributed.eplb import async_worker as _async_worker
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import routed_experts as _routed_experts
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_ascend.distributed.eplb.eplb_communicator import (
    AscendHixlEplbCommunicator,
)
from vllm_ascend.distributed.eplb.eplb_state import (
    _EXPLICIT_TRANSFER_TARGET_ATTR,
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    EXPERT_MAPPING_EP_SIZE,
    AscendEplbState,
    _AscendAsyncLayerResult,
)
from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
# Old async APIs pass one target layer at a time. Preserve the augmented full
# target on its per-model communicator until the last workspace commit.


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
            lambda self, *_args, **_kwargs: (
                self.moe_config.ep_size if getattr(self, "_use_v2_model_runner", False) else 1
            ),
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


def _has_explicit_sources(target) -> bool:
    return hasattr(target, "source_rank_ids") and hasattr(target, "source_slot_ids")


def _clear_transfer_target(communicator, target=None) -> None:
    if target is None or getattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, None) is target:
        communicator.__dict__.pop(_EXPLICIT_TRANSFER_TARGET_ATTR, None)


def _wrap_async_transfer(original_transfer):
    transfer_signature = signature(original_transfer)
    stream_parameter = "stream" if "stream" in transfer_signature.parameters else "cuda_stream"
    required = {"old_layer_indices", "new_layer_indices", "expert_weights", "expert_weights_buffer"}
    required.update({"ep_group", "communicator", "is_profile", stream_parameter, "rank_mapping", "layer_idx"})
    if not required.issubset(transfer_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: asynchronous transfer signature changed.")

    @wraps(original_transfer)
    def _async_transfer(*args, **kwargs):
        bound = transfer_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        communicator = values["communicator"]
        full_target = getattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, None)
        if full_target is None or not _has_explicit_sources(full_target):
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
                stream=values[stream_parameter],
                layer_idx=layer_idx,
            )
        except Exception:
            _clear_transfer_target(communicator, full_target)
            raise

    setattr(_async_transfer, _PATCH_MARKER, True)
    return _async_transfer


def _patch_explicit_transfer_execution() -> None:
    original_async_transfer = _async_worker.transfer_layer
    if not getattr(original_async_transfer, _PATCH_MARKER, False):
        _async_worker.transfer_layer = _wrap_async_transfer(original_async_transfer)


def _wrap_async_worker(original_worker):
    worker_signature = signature(original_worker)
    worker_stream_parameter = "stream" if "stream" in worker_signature.parameters else "cuda_stream"
    transfer_signature = signature(_async_worker.transfer_layer)
    transfer_stream_parameter = "stream" if "stream" in transfer_signature.parameters else "cuda_stream"

    def _run(state, stream, is_profile):
        rank = _async_worker.get_eplb_group().device_group.rank()
        while state.wait_for_rearrangement(stream):
            eplb_group = _async_worker.get_eplb_group().device_group
            eplb_cpu_group = _async_worker.get_eplb_group().cpu_group
            for model_state in state.model_states.values():
                model_state.communicator.set_stream(stream)
                _clear_transfer_target(model_state.communicator)
                previous_consumed_event = None
                with stream if stream is not None else nullcontext():
                    old_mapping = model_state.physical_to_logical_map.cpu()
                new_mapping = _async_worker.run_rebalance_experts(model_state, state, old_mapping, stream)
                if new_mapping.device.type != "cpu" or new_mapping.dtype not in (torch.int32, torch.int64):
                    raise ValueError("EPLB policy target must be a CPU int32 or int64 tensor")
                if old_mapping.shape != new_mapping.shape:
                    raise ValueError("EPLB planner changed the mapping shape")
                changed_layers = torch.nonzero((old_mapping != new_mapping).any(dim=1)).flatten().tolist()
                new_mapping.changed_layer_count = len(changed_layers)
                if _has_explicit_sources(new_mapping):
                    setattr(model_state.communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, new_mapping)
                if not changed_layers:
                    flag = torch.tensor([int(model_state.rebalanced)], dtype=torch.int32, device="cpu")
                    torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                    if int(flag.item()) != eplb_cpu_group.size():
                        model_state.rebalanced = False
                        continue
                    consumed_event = _async_worker.CpuGpuEvent()
                    model_state.pending_result = _AscendAsyncLayerResult(
                        layer_idx=None,
                        new_physical_to_logical_map=None,
                        new_logical_to_physical_map=None,
                        new_logical_replica_count=None,
                        transfer_metadata=None,
                        consumed_event=consumed_event,
                        is_last_result=True,
                    )
                    consumed_event.wait(stream=stream)
                    state._last_consumed_event = consumed_event._event
                    assert model_state.pending_result is None
                    continue

                # Pinning may replace the tensor, and the foreground clears the
                # communicator's target before acknowledging the last result.
                cycle_target = new_mapping
                new_mapping, new_logical_map, new_replica_count = _prepare_commit_maps(
                    model_state,
                    new_mapping,
                )

                for index, layer_idx in enumerate(changed_layers):
                    flag = torch.tensor([int(model_state.rebalanced)], dtype=torch.int32, device="cpu")
                    torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                    if int(flag.item()) != eplb_cpu_group.size():
                        model_state.rebalanced = False
                        break
                    transfer_kwargs = {
                        "old_layer_indices": old_mapping[layer_idx],
                        "new_layer_indices": new_mapping[layer_idx],
                        "expert_weights": model_state.model.expert_weights[layer_idx],
                        "expert_weights_buffer": model_state.expert_buffer,
                        "communicator": model_state.communicator,
                        "ep_group": eplb_group,
                        "is_profile": is_profile,
                        "layer_idx": layer_idx,
                        transfer_stream_parameter: stream,
                    }
                    communicator = model_state.communicator
                    if isinstance(communicator, AscendHixlEplbCommunicator):
                        communicator.set_consumed_event(previous_consumed_event)
                    try:
                        metadata = _async_worker.transfer_layer(**transfer_kwargs)
                    finally:
                        if isinstance(communicator, AscendHixlEplbCommunicator):
                            communicator.set_consumed_event(None)
                    with gpu_sync_allowed():
                        stream.synchronize()
                    consumed_event = _async_worker.CpuGpuEvent()
                    model_state.pending_result = _AscendAsyncLayerResult(
                        layer_idx=layer_idx,
                        new_physical_to_logical_map=new_mapping[layer_idx],
                        new_logical_to_physical_map=new_logical_map[layer_idx],
                        new_logical_replica_count=new_replica_count[layer_idx],
                        transfer_metadata=metadata,
                        consumed_event=consumed_event,
                        is_last_result=index == len(changed_layers) - 1,
                    )
                    consumed_event.wait(stream=stream)
                    state._last_consumed_event = consumed_event._event
                    assert model_state.pending_result is None
                    if isinstance(communicator, AscendHixlEplbCommunicator):
                        # record() replaces the underlying event. Capture it only
                        # after wait() has observed the foreground acknowledgement.
                        previous_consumed_event = consumed_event._event
                    if index == len(changed_layers) - 1:
                        # On both supported lanes, the last workspace move may
                        # use upstream's helper instead of the wrapped commit
                        # path, so report completion from this worker.
                        if rank == 0:
                            if not _has_explicit_sources(cycle_target):
                                logger.info(
                                    "%s: model=%s",
                                    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                                    model_state.model_name,
                                )
                            else:
                                source_ranks = np.asarray(cycle_target.source_rank_ids)
                                destination_ranks = np.arange(source_ranks.shape[-2])[None, :, None]
                                rank_transfers = np.count_nonzero(source_ranks != destination_ranks)
                                imbalance = getattr(cycle_target, "predicted_imbalance_summary", None)
                                rank_node_ids = getattr(cycle_target, "rank_node_ids", None)
                                if imbalance is None or rank_node_ids is None:
                                    logger.info(
                                        "%s: model=%s rank_transfers=%d",
                                        ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                                        model_state.model_name,
                                        rank_transfers,
                                    )
                                else:
                                    rank_node_ids = np.asarray(rank_node_ids)
                                    cross_node_transfers = np.count_nonzero(
                                        rank_node_ids[source_ranks] != rank_node_ids[destination_ranks]
                                    )
                                    mean_before, p95_before, mean_after, p95_after = imbalance
                                    logger.info(
                                        "%s: model=%s mean=%.4f->%.4f p95=%.4f->%.4f changed_layers=%d "
                                        "rank_transfers=%d cross_node_transfers=%d",
                                        ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                                        model_state.model_name,
                                        mean_before,
                                        mean_after,
                                        p95_before,
                                        p95_after,
                                        len(changed_layers),
                                        rank_transfers,
                                        cross_node_transfers,
                                    )

    @wraps(original_worker)
    def _transfer_run_periodically(*args, **kwargs):
        bound = worker_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        state = bound.arguments["state"]
        stream = bound.arguments[worker_stream_parameter]
        is_profile = bound.arguments["is_profile"]
        if not isinstance(state, AscendEplbState) or is_profile:
            return original_worker(*bound.args, **bound.kwargs)
        try:
            return _run(state, stream, is_profile)
        except Exception as error:
            if state._hixl_session is not None:
                # Preserve ownership when migration or context restoration fails.
                session = state._hixl_session
                if session._error is None:
                    session._error = error
                session._usable = False
            raise

    setattr(_transfer_run_periodically, _PATCH_MARKER, True)
    return _transfer_run_periodically


def _pin_host_map(src: torch.Tensor) -> torch.Tensor:
    if PIN_MEMORY and src.is_cpu and not src.is_pinned():
        return src.pin_memory()
    return src


def _prepare_commit_maps(
    model_state,
    physical_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logical_map, replica_count = _eplb_state.compute_logical_maps(
        physical_map,
        model_state.logical_to_physical_map.shape[1],
    )
    padding = model_state.logical_to_physical_map.shape[-1] - logical_map.shape[-1]
    if padding < 0:
        raise ValueError("Async EPLB produced more replicas than the configured capacity")
    if padding:
        logical_map = torch.nn.functional.pad(logical_map, (0, padding), value=-1)
    return (
        _pin_host_map(physical_map),
        _pin_host_map(logical_map),
        _pin_host_map(replica_count),
    )


def _patch_changed_layer_transfer() -> None:
    original_worker = _async_worker.transfer_run_periodically
    if not getattr(original_worker, _PATCH_MARKER, False):
        _async_worker.transfer_run_periodically = _wrap_async_worker(original_worker)


_patch_initial_expert_layout()
_patch_explicit_transfer_execution()
_patch_changed_layer_transfer()
