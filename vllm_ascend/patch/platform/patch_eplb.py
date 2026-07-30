# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Patch vLLM EPLB construction points for the Ascend backend."""

from functools import wraps
from inspect import signature

import torch
import torch_npu
from pydantic.dataclasses import rebuild_dataclass
from vllm.config import ParallelConfig
from vllm.config import parallel as _parallel_config
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

from vllm_ascend.distributed.eplb_communicator import HcclEplbCommunicator
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
_NPU_FORMAT_ND = 2
logger = init_logger(__name__)


def _replace_nz_expert_buffers_with_nd(model_state) -> None:
    """Use logical ND tensors as the transfer workspace for NZ expert weights."""
    if getattr(model_state, "_ascend_uses_nd_expert_buffers", False):
        return
    replacement_formats = []
    for weight_idx, weight_view in enumerate(model_state.model.expert_weights[0]):
        if not isinstance(weight_view, list):
            continue
        formats = [int(torch_npu.get_npu_format(weight)) for weight in weight_view]
        if not any(weight_format != _NPU_FORMAT_ND for weight_format in formats):
            continue
        buffer_type = type(weight_view)
        model_state.expert_buffer[weight_idx] = buffer_type(
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            for weight in weight_view
        )
        replacement_formats.append(
            (
                weight_idx,
                formats[0],
                int(torch_npu.get_npu_format(model_state.expert_buffer[weight_idx][0])),
            )
        )
    model_state._ascend_uses_nd_expert_buffers = True
    if replacement_formats and get_ep_group().rank_in_group == 0:
        logger.info(
            "Ascend EPLB uses ND transfer buffers for NZ expert weights: %s",
            replacement_formats,
        )


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


def _wrap_parallel_config_post_init(original_post_init):
    @wraps(original_post_init)
    def _post_init(self, *args, **kwargs):
        platform = _parallel_config.current_platform
        if (
            self.enable_eplb
            and _is_npu_platform(platform)
            and not self.eplb_config.use_async
            and self.eplb_config.communicator is None
        ):
            # torch_nccl means torch.distributed on the device process
            # group. The communicator factory maps it to HCCL on NPU.
            self.eplb_config.communicator = "torch_nccl"
        return original_post_init(self, *args, **kwargs)

    setattr(_post_init, _PATCH_MARKER, True)
    return _post_init


def _patch_parallel_config() -> None:
    platform = _parallel_config.current_platform
    if not isinstance(platform, _CudaAlikeEplbPlatformProxy):
        # ParallelConfig is embedded in VllmConfig's Pydantic schema. Replacing
        # the module-local platform reference keeps both schemas on the original
        # validator while changing only the NPU EPLB capability result.
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)

    original_post_init = ParallelConfig.__post_init__
    if not getattr(original_post_init, _PATCH_MARKER, False):
        ParallelConfig.__post_init__ = _wrap_parallel_config_post_init(original_post_init)

    rebuild_dataclass(ParallelConfig, force=True)


def _wrap_communicator_factory(original_factory):
    factory_signature = signature(original_factory)
    required_parameters = {"group_coordinator", "backend", "expert_weights", "expert_buffer"}
    if not required_parameters.issubset(factory_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory signature changed.")

    @wraps(original_factory)
    def _create_eplb_communicator(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        group_coordinator = bound.arguments["group_coordinator"]
        backend = bound.arguments["backend"]
        if backend == "torch_nccl" and _is_npu_platform(_parallel_config.current_platform):
            return HcclEplbCommunicator(group_coordinator.device_group)
        return original_factory(*args, **kwargs)

    setattr(_create_eplb_communicator, _PATCH_MARKER, True)
    return _create_eplb_communicator


def _patch_communicator_factory() -> None:
    original_factory = _eplb_communicator.create_eplb_communicator
    if getattr(original_factory, _PATCH_MARKER, False):
        return
    _create_eplb_communicator = _wrap_communicator_factory(original_factory)
    _eplb_communicator.create_eplb_communicator = _create_eplb_communicator
    # eplb_state imports the factory by name, so update its retained binding too.
    _eplb_state.create_eplb_communicator = _create_eplb_communicator


def _patch_router() -> None:
    original_apply = BaseRouter._apply_eplb_mapping
    if getattr(original_apply, _PATCH_MARKER, False):
        return
    if tuple(signature(original_apply).parameters) != ("self", "topk_ids"):
        raise RuntimeError("Unsupported vLLM EPLB contract: BaseRouter._apply_eplb_mapping signature changed.")

    @wraps(original_apply)
    def _apply_eplb_mapping(self, topk_ids):
        eplb_state = self.eplb_state
        if eplb_state is None:
            return topk_ids
        self._validate_eplb_state()
        physical_id_lookup = getattr(eplb_state, "physical_id_lookup", None)
        if physical_id_lookup is None:
            raise RuntimeError("Ascend EPLB physical ID lookup is not initialized.")
        return torch.ops.vllm.ascend_eplb_map_to_physical(
            topk_ids,
            physical_id_lookup,
        )

    setattr(_apply_eplb_mapping, _PATCH_MARKER, True)
    BaseRouter._apply_eplb_mapping = _apply_eplb_mapping


def _refresh_layer_lookup(layer_state) -> None:
    logical_to_physical_map = layer_state.logical_to_physical_map
    logical_replica_count = layer_state.logical_replica_count
    if logical_to_physical_map is None or logical_replica_count is None:
        raise RuntimeError("Cannot build Ascend EPLB lookup before layer state is initialized.")
    new_lookup = _eplb_ops.build_physical_id_lookup(
        logical_to_physical_map,
        logical_replica_count,
        get_ep_group().rank_in_group,
    )
    physical_id_lookup = getattr(layer_state, "physical_id_lookup", None)
    if physical_id_lookup is not None and physical_id_lookup.shape == new_lookup.shape:
        physical_id_lookup.copy_(new_lookup, non_blocking=True)
    else:
        layer_state.physical_id_lookup = new_lookup


def _refresh_model_lookups(model_state, layer_idx: int | None = None) -> None:
    layers = list(model_state.model.moe_layers)
    selected_layers = enumerate(layers) if layer_idx is None else ((layer_idx, layers[layer_idx]),)
    for _, layer in selected_layers:
        layer_state = layer.eplb_state
        if layer_state is not None:
            _refresh_layer_lookup(layer_state)


def _wrap_set_layer_state(original_set_layer_state):
    state_signature = signature(original_set_layer_state)
    required_parameters = {
        "self",
        "moe_layer_idx",
        "expert_load_view",
        "logical_to_physical_map",
        "logical_replica_count",
    }
    if not required_parameters.issubset(state_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbLayerState.set_layer_state signature changed.")

    @wraps(original_set_layer_state)
    def _set_layer_state(*args, **kwargs):
        bound = state_signature.bind(*args, **kwargs)
        result = original_set_layer_state(*bound.args, **bound.kwargs)
        _refresh_layer_lookup(bound.arguments["self"])
        return result

    setattr(_set_layer_state, _PATCH_MARKER, True)
    return _set_layer_state


def _wrap_commit_eplb_maps(original_commit, *, per_layer: bool):
    commit_signature = signature(original_commit)
    required_parameters = {"model_state", "new_physical_to_logical_map"}
    if per_layer:
        required_parameters.add("layer")
    if not required_parameters.issubset(commit_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: map commit signature changed.")

    @wraps(original_commit)
    def _commit(*args, **kwargs):
        bound = commit_signature.bind(*args, **kwargs)
        result = original_commit(*bound.args, **bound.kwargs)
        layer_idx = bound.arguments["layer"] if per_layer else None
        _refresh_model_lookups(bound.arguments["model_state"], layer_idx)
        return result

    setattr(_commit, _PATCH_MARKER, True)
    return _commit


def _wrap_from_mapping(original_from_mapping):
    from_mapping_signature = signature(original_from_mapping)
    required_parameters = {
        "cls",
        "model",
        "model_config",
        "device",
        "parallel_config",
        "expanded_physical_to_logical",
        "num_valid_physical_experts",
    }
    if not required_parameters.issubset(from_mapping_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.from_mapping signature changed.")

    @wraps(original_from_mapping)
    def _from_mapping(*args, **kwargs):
        state = original_from_mapping(*args, **kwargs)
        for model_state in state.model_states.values():
            _replace_nz_expert_buffers_with_nd(model_state)
            _refresh_model_lookups(model_state)
        return state

    setattr(_from_mapping, _PATCH_MARKER, True)
    return _from_mapping


def _wrap_add_model(original_add_model):
    add_model_signature = signature(original_add_model)
    required_parameters = {"self", "model", "model_config"}
    if not required_parameters.issubset(add_model_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.add_model signature changed.")

    @wraps(original_add_model)
    def _add_model(*args, **kwargs):
        bound = add_model_signature.bind(*args, **kwargs)
        result = original_add_model(*bound.args, **bound.kwargs)
        model = bound.arguments["model"]
        for model_state in bound.arguments["self"].model_states.values():
            if model_state.model is model:
                _replace_nz_expert_buffers_with_nd(model_state)
                break
        return result

    setattr(_add_model, _PATCH_MARKER, True)
    return _add_model


def _wrap_eplb_state_step(original_step):
    step_signature = signature(original_step)
    required_parameters = {"self", "is_dummy", "is_profile", "log_stats"}
    if not required_parameters.issubset(step_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.step signature changed.")

    @wraps(original_step)
    def _step(self, *args, **kwargs):
        bound = step_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if (
            not getattr(self, "_ascend_scope_matched", True)
            and not bound.arguments["is_dummy"]
            and not bound.arguments["is_profile"]
        ):
            bound.arguments["is_dummy"] = True
            bound.arguments["log_stats"] = False
        elif (
            not bound.arguments["is_dummy"]
            and not bound.arguments["is_profile"]
            and not self._should_record_current_step(log_stats=bound.arguments["log_stats"])
        ):
            # V2 records local GMM counts with one add per layer. Clear the
            # whole pass here when the upstream collection window is closed,
            # avoiding a device-side record predicate in every MoE layer.
            for model_state in self.model_states.values():
                model_state.expert_load_pass.zero_()
        return original_step(*bound.args, **bound.kwargs)

    setattr(_step, _PATCH_MARKER, True)
    return _step


def _patch_eplb_state() -> None:
    original_add_model = _eplb_state.EplbState.add_model
    if not getattr(original_add_model, _PATCH_MARKER, False):
        _eplb_state.EplbState.add_model = _wrap_add_model(original_add_model)

    original_step = _eplb_state.EplbState.step
    if not getattr(original_step, _PATCH_MARKER, False):
        _eplb_state.EplbState.step = _wrap_eplb_state_step(original_step)

    original_set_layer_state = _eplb_state.EplbLayerState.set_layer_state
    if not getattr(original_set_layer_state, _PATCH_MARKER, False):
        _eplb_state.EplbLayerState.set_layer_state = _wrap_set_layer_state(original_set_layer_state)

    original_commit = _eplb_state._commit_eplb_maps
    if not getattr(original_commit, _PATCH_MARKER, False):
        _eplb_state._commit_eplb_maps = _wrap_commit_eplb_maps(original_commit, per_layer=False)

    original_commit_layer = _eplb_state._commit_eplb_maps_for_layer
    if not getattr(original_commit_layer, _PATCH_MARKER, False):
        _eplb_state._commit_eplb_maps_for_layer = _wrap_commit_eplb_maps(
            original_commit_layer,
            per_layer=True,
        )

    original_from_mapping = _eplb_state.EplbState.__dict__["from_mapping"].__func__
    if not getattr(original_from_mapping, _PATCH_MARKER, False):
        _eplb_state.EplbState.from_mapping = classmethod(_wrap_from_mapping(original_from_mapping))


_patch_parallel_config()
_patch_communicator_factory()
_patch_router()
_patch_eplb_state()
