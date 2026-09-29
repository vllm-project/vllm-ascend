# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Narrow vLLM EPLB construction, execution, and commit adapters for Ascend."""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, Literal, get_args

import numpy as np
import torch
from pydantic.dataclasses import rebuild_dataclass
from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import async_worker as _async_worker
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import routed_experts as _routed_experts
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_ascend.distributed.eplb.eplb_communicator import (
    AscendGlooEplbCommunicator,
    AscendHixlEplbCommunicator,
)
from vllm_ascend.distributed.eplb.eplb_state import (
    ASYNC_EPLB_CYCLE_COMMITTED_LOG,
    EXPERT_MAPPING_EP_SIZE,
    AscendEplbState,
    refresh_model_routing_tables,
)
from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
# Old async APIs pass one target layer at a time. Preserve the augmented full
# target on its per-model communicator until the last workspace commit.
_EXPLICIT_TRANSFER_TARGET_ATTR = "_vllm_ascend_explicit_transfer_target"
_ASCEND_EPLB_POLICIES = ("default", "stair")


@dataclass
class _AscendAsyncLayerResult:
    layer_idx: int | None
    new_physical_to_logical_map: torch.Tensor | None
    new_logical_to_physical_map: torch.Tensor | None
    new_logical_replica_count: torch.Tensor | None
    transfer_metadata: Any
    consumed_event: Any
    is_last_result: bool


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


def _probe_hixl_binding() -> str | None:
    """Return None when HIXL is usable, otherwise the reason it is not.

    Two bindings qualify: the official ``hixl`` Python package shipped with
    newer CANN distributions, and the vllm_ascend ctypes fallback that drives
    ``libcann_hixl.so`` directly when the package is absent.
    """
    try:
        import hixl

        if hasattr(hixl, "Hixl"):
            return None
        package_reason = "the hixl package has no Hixl binding"
    except ImportError as error:
        package_reason = str(error) or "the hixl package is missing"

    from vllm_ascend.distributed.eplb import hixl_compat

    try:
        hixl_compat.ensure_available()
    except Exception as error:
        return f"{package_reason}; ctypes fallback: {error}"
    return None


def _patch_parallel_config() -> None:
    platform = _parallel_config.current_platform
    if not isinstance(platform, _CudaAlikeEplbPlatformProxy):
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)

    original_post_init = _parallel_config.ParallelConfig.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def _post_init(config):
        if (
            _is_npu_platform(_parallel_config.current_platform)
            and config.enable_eplb
            and config.eplb_config.communicator is None
        ):
            unavailable = _probe_hixl_binding()
            if unavailable is None:
                config.eplb_config.communicator = "hixl"
                logger.info("Ascend EPLB selected hixl: a HIXL binding is available.")
            else:
                config.eplb_config.communicator = "torch_gloo"
                logger.info("Ascend EPLB selected torch_gloo: HIXL is unavailable (%s).", unavailable)
        original_post_init(config)

    setattr(_post_init, _PATCH_MARKER, True)
    _parallel_config.ParallelConfig.__post_init__ = _post_init
    rebuild_dataclass(_parallel_config.ParallelConfig, force=True)


def _patch_eplb_policy_config() -> None:
    """Extend the upstream policy field while preserving its validation."""
    config_cls = _parallel_config.EPLBConfig
    policy_field = getattr(config_cls, "__dataclass_fields__", {}).get("policy")
    if policy_field is None:
        raise RuntimeError("Unsupported vLLM EPLB contract: policy field is missing.")
    policy_type = Literal["default", "stair"]
    if get_args(policy_field.type) == _ASCEND_EPLB_POLICIES and policy_field.default == "stair":
        return

    decorators = getattr(config_cls, "__pydantic_decorators__", None)
    validator = None if decorators is None else decorators.model_validators.get("_validate_eplb_config")
    if validator is None:
        raise RuntimeError("Unsupported vLLM EPLB contract: policy validator is missing.")
    original_validator = validator.func

    @wraps(original_validator)
    def _validate_with_stair(config):
        if config.policy != "stair":
            return original_validator(config)
        config.policy = "default"
        try:
            validated = original_validator(config)
        finally:
            config.policy = "stair"
        return validated

    _parallel_config.EPLBPolicyOption = policy_type  # type: ignore[misc]
    config_cls.__annotations__["policy"] = policy_type
    policy_field.type = policy_type
    policy_field.default = "stair"
    validator.func = _validate_with_stair


def _patch_eplb_communicator_config() -> None:
    """Add HIXL to the upstream selector while retaining automatic selection."""
    config_cls = _parallel_config.EPLBConfig
    communicator_field = getattr(config_cls, "__dataclass_fields__", {}).get("communicator")
    if communicator_field is None:
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator field is missing.")
    communicator_type = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl", "hixl"] | None
    if communicator_field.type == communicator_type and communicator_field.default is None:
        return

    if communicator_field.type != communicator_type:
        _parallel_config.EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl", "hixl"]
        config_cls.__annotations__["communicator"] = communicator_type
        communicator_field.type = communicator_type
    communicator_field.default = None
    rebuild_dataclass(config_cls, force=True)
    rebuild_dataclass(_parallel_config.ParallelConfig, force=True)


def _wrap_communicator_factory(original_factory):
    factory_signature = signature(original_factory)
    if "group_coordinator" not in factory_signature.parameters:
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory has no group_coordinator parameter.")

    @wraps(original_factory)
    def _create_eplb_communicator(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        backend = bound.arguments["backend"]
        if backend == "torch_gloo":
            return AscendGlooEplbCommunicator(
                cpu_group=bound.arguments["group_coordinator"].cpu_group,
            )
        if backend == "hixl":
            return AscendHixlEplbCommunicator(
                cpu_group=bound.arguments["group_coordinator"].cpu_group,
                all_expert_weights=bound.arguments["expert_weights"],
                expert_buffer=bound.arguments["expert_buffer"],
            )
        return original_factory(*bound.args, **bound.kwargs)

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


def _wrap_async_rebalance(original_rebalance):
    rebalance_signature = signature(original_rebalance)
    stream_parameter = "stream" if "stream" in rebalance_signature.parameters else "cuda_stream"
    required = {
        "model_state",
        "eplb_state",
        "physical_to_logical_map_cpu",
        stream_parameter,
    }
    if not required.issubset(rebalance_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: asynchronous rebalance signature changed.")

    @wraps(original_rebalance)
    def _async_rebalance(*args, **kwargs):
        bound = rebalance_signature.bind(*args, **kwargs)
        model_state = bound.arguments["model_state"]
        eplb_state = bound.arguments["eplb_state"]
        communicator = model_state.communicator
        _clear_transfer_target(communicator)
        prepared_stats = getattr(model_state, "_policy_load_stats", None)
        eplb_stats = model_state.eplb_stats
        if (
            prepared_stats is None
            or eplb_stats is None
            or prepared_stats.values is not eplb_stats.global_expert_load_window
        ):
            target = original_rebalance(*bound.args, **bound.kwargs)
        else:
            stream = bound.arguments[stream_parameter]
            with stream if stream is not None else nullcontext():
                cpu_stats = PreparedLoadStats(
                    prepared_stats.values.cpu(),
                    prepared_stats.sample_counts,
                )
            current_mapping = bound.arguments["physical_to_logical_map_cpu"]
            rank_node_ids = eplb_state.get_rank_node_ids()
            target = eplb_state.policy.rebalance_experts(
                cpu_stats,
                eplb_stats.num_replicas,
                eplb_stats.num_groups,
                eplb_stats.num_nodes,
                eplb_stats.num_gpus,
                current_mapping,
                last_committed_mean_ratios=model_state._last_committed_mean_ratios,
                rank_node_ids=rank_node_ids,
            )
            if target.device.type != "cpu":
                raise RuntimeError("EPLB policy returned a non-CPU expert mapping")
            target.changed_layer_count = int((target != current_mapping).any(dim=1).sum().item())
            target.rank_node_ids = rank_node_ids
        if _has_explicit_sources(target):
            setattr(communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, target)
        return target

    setattr(_async_rebalance, _PATCH_MARKER, True)
    return _async_rebalance


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
    original_rebalance = _async_worker.run_rebalance_experts
    if not getattr(original_rebalance, _PATCH_MARKER, False):
        _async_worker.run_rebalance_experts = _wrap_async_rebalance(original_rebalance)
    original_async_transfer = _async_worker.transfer_layer
    if not getattr(original_async_transfer, _PATCH_MARKER, False):
        _async_worker.transfer_layer = _wrap_async_transfer(original_async_transfer)


def _wrap_async_worker(original_worker):
    worker_signature = signature(original_worker)
    worker_stream_parameter = "stream" if "stream" in worker_signature.parameters else "cuda_stream"
    transfer_signature = signature(_async_worker.transfer_layer)
    transfer_stream_parameter = "stream" if "stream" in transfer_signature.parameters else "cuda_stream"

    @wraps(original_worker)
    def _transfer_run_periodically(*args, **kwargs):
        bound = worker_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        state = bound.arguments["state"]
        stream = bound.arguments[worker_stream_parameter]
        is_profile = bound.arguments["is_profile"]
        if not isinstance(state, AscendEplbState) or is_profile:
            return original_worker(*bound.args, **bound.kwargs)
        while True:
            state.rearrange_event.wait(stream=stream)
            eplb_group = _async_worker.get_eplb_group().device_group
            eplb_cpu_group = _async_worker.get_eplb_group().cpu_group
            for model_state in state.model_states.values():
                model_state.communicator.set_stream(stream)
                with stream if stream is not None else nullcontext():
                    old_mapping = model_state.physical_to_logical_map.cpu()
                new_mapping = _async_worker.run_rebalance_experts(model_state, state, old_mapping, stream)
                if old_mapping.shape != new_mapping.shape:
                    raise ValueError("EPLB planner changed the mapping shape")
                changed_layers = torch.nonzero((old_mapping != new_mapping).any(dim=1)).flatten().tolist()
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
                    assert model_state.pending_result is None
                    continue

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
                    metadata = _async_worker.transfer_layer(**transfer_kwargs)
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
                    assert model_state.pending_result is None

    setattr(_transfer_run_periodically, _PATCH_MARKER, True)
    return _transfer_run_periodically


def _move_changed_layer_to_workspace(model_state, ep_rank: int) -> None:
    result = model_state.pending_result
    assert result is not None
    if result.layer_idx is not None:
        assert result.new_physical_to_logical_map is not None
        assert result.new_logical_to_physical_map is not None
        assert result.new_logical_replica_count is not None
        _eplb_state.move_from_buffer(
            expert_weights=model_state.model.expert_weights[result.layer_idx],
            expert_weights_buffers=model_state.expert_buffer,
            transfer_metadata=result.transfer_metadata,
            new_indices=result.new_physical_to_logical_map.numpy(),
            ep_rank=ep_rank,
        )
        _commit_ascend_maps_for_layer(
            model_state,
            result.new_physical_to_logical_map,
            result.new_logical_to_physical_map,
            result.new_logical_replica_count,
            result.layer_idx,
        )
    if result.is_last_result:
        model_state.rebalanced = False
    model_state.pending_result = None
    result.consumed_event.record()


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
    return tuple(_pin_host_map(value) for value in (physical_map, logical_map, replica_count))


def _copy_host_map(src: torch.Tensor, dst: torch.Tensor) -> None:
    dst.copy_(src, non_blocking=True)


def _commit_ascend_maps_for_layer(
    model_state,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    layer_idx: int,
) -> None:
    physical_dst = model_state.physical_to_logical_map[layer_idx]
    if physical_to_logical_map.shape != physical_dst.shape:
        raise ValueError("Async EPLB cannot change the number of physical experts")
    _copy_host_map(physical_to_logical_map, physical_dst)

    logical_dst = model_state.logical_to_physical_map[layer_idx]
    if logical_to_physical_map.shape != logical_dst.shape:
        raise ValueError("Prepared logical map does not match the configured capacity")
    _copy_host_map(logical_to_physical_map, logical_dst)
    _copy_host_map(
        logical_replica_count,
        model_state.logical_replica_count[layer_idx],
    )


def _patch_changed_layer_transfer() -> None:
    original_worker = _async_worker.transfer_run_periodically
    if not getattr(original_worker, _PATCH_MARKER, False):
        _async_worker.transfer_run_periodically = _wrap_async_worker(original_worker)


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
        full_target = getattr(model_state.communicator, _EXPLICIT_TRANSFER_TARGET_ATTR, None)

        deferred_event = None
        consumed_event = None
        if pending_result is not None:
            consumed_event = pending_result.consumed_event
            deferred_event = _DeferredConsumedEvent(consumed_event)
            pending_result.consumed_event = deferred_event
        result = None
        try:
            if isinstance(pending_result, _AscendAsyncLayerResult):
                _move_changed_layer_to_workspace(
                    model_state,
                    bound.arguments["ep_rank"],
                )
            else:
                result = original_move(*bound.args, **bound.kwargs)
            if layer_idx is not None:
                refresh_model_routing_tables(model_state, layer_idx)
                if full_target is not None and hasattr(full_target, "predicted_mean_ratios"):
                    predicted_ratio = full_target.predicted_mean_ratios[layer_idx]
                    if np.isfinite(predicted_ratio):
                        model_state._last_committed_mean_ratios[layer_idx] = predicted_ratio
            if is_last_result:
                _clear_transfer_target(model_state.communicator)
                if bound.arguments["ep_rank"] == 0:
                    if full_target is None:
                        logger.info(
                            "%s: model=%s",
                            ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                            model_state.model_name,
                        )
                    else:
                        source_ranks = np.asarray(full_target.source_rank_ids)
                        destination_ranks = np.arange(source_ranks.shape[-2])[None, :, None]
                        rank_transfers = np.count_nonzero(source_ranks != destination_ranks)
                        imbalance = getattr(full_target, "predicted_imbalance_summary", None)
                        rank_node_ids = getattr(full_target, "rank_node_ids", None)
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
                                full_target.changed_layer_count,
                                rank_transfers,
                                cross_node_transfers,
                            )
                hixl_timings = getattr(model_state.communicator, "_eplb_hixl_phase_timings", [])
                if hixl_timings and bound.arguments["ep_rank"] == 0:
                    logger.info(
                        "HIXL EPLB transfer: model=%s rank=%d launch_ms=%.3f background_transfer_ms=%.3f "
                        "confirmation_ms=%.3f foreground_wait_ms=%.3f requests=%d bytes=%d layers=%d "
                        "migration_span_steps=%d migration_deferred_steps=%d",
                        model_state.model_name,
                        bound.arguments["ep_rank"],
                        sum(timing.launch_ms for timing in hixl_timings),
                        sum(timing.transfer_ms for timing in hixl_timings),
                        sum(timing.confirmation_ms for timing in hixl_timings),
                        getattr(model_state, "_eplb_foreground_wait_ms", 0.0),
                        sum(timing.request_count for timing in hixl_timings),
                        sum(timing.transfer_bytes for timing in hixl_timings),
                        len(hixl_timings),
                        getattr(model_state, "_eplb_migration_span_steps", 0),
                        getattr(model_state, "_eplb_migration_deferred_steps", 0),
                    )
                model_state.communicator.__dict__.pop("_eplb_hixl_phase_timings", None)
                model_state.__dict__.pop("_eplb_migration_span_steps", None)
                model_state.__dict__.pop("_eplb_migration_deferred_steps", None)
                model_state.__dict__.pop("_eplb_foreground_wait_ms", None)
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


_patch_eplb_policy_config()
_patch_eplb_communicator_config()
_patch_parallel_config()
_patch_initial_expert_layout()
_patch_communicator_factory()
_patch_explicit_transfer_execution()
_patch_changed_layer_transfer()
_patch_async_move_to_workspace()
