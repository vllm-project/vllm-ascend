# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Patch vLLM EPLB construction points for the Ascend backend."""

from functools import wraps
from inspect import signature

import torch
from pydantic.dataclasses import rebuild_dataclass
from vllm.config import ParallelConfig
from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_ascend.distributed.eplb_communicator import HcclEplbCommunicator
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops  # noqa: F401

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
        # ParallelConfig is embedded in VllmConfig's Pydantic schema. Replacing
        # the module-local platform reference keeps both schemas on the original
        # validator while changing only the NPU EPLB capability result.
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)

    original_post_init = ParallelConfig.__post_init__
    if not getattr(original_post_init, _PATCH_MARKER, False):

        @wraps(original_post_init)
        def _post_init(self):
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
            return original_post_init(self)

        setattr(_post_init, _PATCH_MARKER, True)
        ParallelConfig.__post_init__ = _post_init

    rebuild_dataclass(ParallelConfig, force=True)


def _patch_communicator_factory() -> None:
    original_factory = _eplb_communicator.create_eplb_communicator
    if getattr(original_factory, _PATCH_MARKER, False):
        return
    required_parameters = {"group_coordinator", "backend", "expert_weights", "expert_buffer"}
    if not required_parameters.issubset(signature(original_factory).parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory signature changed.")

    @wraps(original_factory)
    def _create_eplb_communicator(group_coordinator, backend, expert_weights, expert_buffer):
        if backend == "torch_nccl" and _is_npu_platform(_parallel_config.current_platform):
            return HcclEplbCommunicator(group_coordinator.device_group)
        return original_factory(group_coordinator, backend, expert_weights, expert_buffer)

    setattr(_create_eplb_communicator, _PATCH_MARKER, True)
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
        return torch.ops.vllm.ascend_eplb_map_and_record(
            topk_ids,
            eplb_state.logical_to_physical_map,
            eplb_state.logical_replica_count,
            eplb_state.expert_load_view,
            eplb_state.should_record_tensor,
            eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()],
        )

    setattr(_apply_eplb_mapping, _PATCH_MARKER, True)
    BaseRouter._apply_eplb_mapping = _apply_eplb_mapping


def _patch_eplb_state() -> None:
    original_step = _eplb_state.EplbState.step
    if getattr(original_step, _PATCH_MARKER, False):
        return
    required_parameters = {"self", "is_dummy", "is_profile", "log_stats"}
    if not required_parameters.issubset(signature(original_step).parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.step signature changed.")

    @wraps(original_step)
    def _step(self, is_dummy=False, is_profile=False, log_stats=False):
        if not is_dummy and not is_profile and not getattr(self, "_ascend_scope_matched", True):
            is_dummy = True
            log_stats = False
        return original_step(self, is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    setattr(_step, _PATCH_MARKER, True)
    _eplb_state.EplbState.step = _step


_patch_parallel_config()
_patch_communicator_factory()
_patch_router()
_patch_eplb_state()
