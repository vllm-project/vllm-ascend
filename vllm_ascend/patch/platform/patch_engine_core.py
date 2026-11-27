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

from functools import wraps

import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.engine.core import EngineCore, EngineCoreProc

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc, _balance_scheduling_enabled
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc, _get_dyntra_lb_config
from vllm_ascend.patch.platform.patch_dyntra_lb_core import _print_rank_0 as dyntra_print_rank_0

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


def _apply_engram_ssd_startup_patch() -> None:
    # Activate after the engine has finished worker warmup and graph capture.
    original_init = EngineCore.__init__
    if getattr(original_init, "_engram_ssd_startup", False):
        return

    @wraps(original_init)
    def initialize_and_activate(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm_ascend.engram_ssd_activation import activate_engine_protocol

        activate_engine_protocol(self)

    initialize_and_activate._engram_ssd_startup = True
    EngineCore.__init__ = initialize_and_activate


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _apply_engram_ssd_startup_patch()
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)


_apply_patch()
