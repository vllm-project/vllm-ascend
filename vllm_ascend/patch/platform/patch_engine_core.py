#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextvars import ContextVar

import vllm.v1.engine.core as _engine_core_mod
from vllm.config import VllmConfig
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    get_layerwise_offload_config,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    AscendLayerwiseKVCacheConfig,
)
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc, _balance_scheduling_enabled
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc, _get_dyntra_lb_config
from vllm_ascend.patch.platform.patch_dyntra_lb_core import _print_rank_0 as dyntra_print_rank_0
from vllm_ascend.patch.platform.patch_pp_mtp import _patch_engine_core as pp_mtp_patch_post_step

_PATCHED = False

_OriginalRunEngineCore = EngineCoreProc.run_engine_core
_original_initialize_kv_caches = EngineCore._initialize_kv_caches
_original_generate_scheduler_kv_cache_config = _engine_core_mod.generate_scheduler_kv_cache_config
_scheduler_planning_config: ContextVar[VllmConfig | None] = ContextVar("scheduler_planning_config", default=None)


def _initialize_kv_caches(self: EngineCore, vllm_config: VllmConfig) -> KVCacheConfig:
    if get_layerwise_offload_config(vllm_config.kv_transfer_config) is None:
        return _original_initialize_kv_caches(self, vllm_config)
    # vLLM's scheduler-config hook receives the rank configs without VllmConfig.
    token = _scheduler_planning_config.set(vllm_config)
    try:
        return _original_initialize_kv_caches(self, vllm_config)
    finally:
        _scheduler_planning_config.reset(token)


def _generate_scheduler_kv_cache_config(kv_cache_configs: list[KVCacheConfig]) -> KVCacheConfig:
    vllm_config = _scheduler_planning_config.get()
    reuse_layout = None
    if vllm_config is not None:
        # Plan before vLLM collapses each group's per-entry specs.
        reuse_layout = plan_layerwise_cache_reuse(kv_cache_configs[0], vllm_config)
    scheduler_config = _original_generate_scheduler_kv_cache_config(kv_cache_configs)
    if reuse_layout is not None:
        return AscendLayerwiseKVCacheConfig.from_config(scheduler_config, reuse_layout)
    return scheduler_config


def _patch_dp_engine_core_proc(vllm_config, dp_rank: int):
    dyntra_lb_config = _get_dyntra_lb_config(vllm_config)
    if dyntra_lb_config.enabled:
        dyntra_print_rank_0(
            "Enable DyntraLB DP load balancing.",
            dp_rank,
            dyntra_lb_config.enable_diagnostics,
        )
        _engine_core_mod.DPEngineCoreProc = DyntraLBDPEngineCoreProc
    elif _balance_scheduling_enabled(vllm_config):
        _engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc


def _run_engine_core_patch_func(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    init_ascend_config(vllm_config)

    _patch_dp_engine_core_proc(vllm_config, dp_rank)

    return _OriginalRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Compute scheduler reuse before vLLM collapses the per-entry specs.
    EngineCore._initialize_kv_caches = _initialize_kv_caches
    _engine_core_mod.generate_scheduler_kv_cache_config = _generate_scheduler_kv_cache_config

    # Patch EngineCore.post_step
    pp_mtp_patch_post_step()
    # Patch EngineCoreProc.run_engine_core
    # And in _run_engine_core_patch_func:
    # 1. Patch EngineCore.__init__ by _apply_profiling_patches
    # 2. Patch class DPEngineCoreProc by _patch_dp_engine_core_proc
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)


_apply_patch()
