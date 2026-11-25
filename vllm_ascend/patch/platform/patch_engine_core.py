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

import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.engine.core import EngineCoreProc

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc, _balance_scheduling_enabled
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc, _get_dyntra_lb_config
from vllm_ascend.patch.platform.patch_dyntra_lb_core import _print_rank_0 as dyntra_print_rank_0
from vllm_ascend.patch.platform.patch_pp_mtp import _patch_engine_core as pp_mtp_patch_post_step

_PATCHED = False

_OriginalRunEngineCore = EngineCoreProc.run_engine_core


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


def _apply_nova_scheduler_patch():
    engine_core = _engine_core_mod.EngineCore
    if getattr(engine_core, "_nova_scheduler_patched", False):
        return
    original_init = engine_core.__init__

    def init_with_nova_scheduler(self, vllm_config, *args, **kwargs):
        options = (vllm_config.additional_config or {}).get("nova_kv") or {}
        if not options.get("artifact_path"):
            return original_init(self, vllm_config, *args, **kwargs)
        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.async_scheduling or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("NOVA-KV recovery requires synchronous scheduling and pipeline parallel size 1")
        if scheduler_config.scheduler_cls is not None:
            raise ValueError("NOVA-KV recovery cannot be combined with a custom scheduler")
        # Keep the config serializable when the executor starts its workers.
        scheduler_config.scheduler_cls = "vllm_ascend.core.nova_scheduler.NovaScheduler"
        try:
            original_init(self, vllm_config, *args, **kwargs)
            self.scheduler.nova_executor = self.model_executor
        finally:
            scheduler_config.scheduler_cls = None

    engine_core.__init__ = init_with_nova_scheduler
    engine_core._nova_scheduler_patched = True


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _apply_nova_scheduler_patch()

    # Patch EngineCore.post_step
    pp_mtp_patch_post_step()
    # Patch EngineCoreProc.run_engine_core
    # And in _run_engine_core_patch_func:
    # Select the DPEngineCoreProc implementation in the child process.
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)


_apply_patch()
