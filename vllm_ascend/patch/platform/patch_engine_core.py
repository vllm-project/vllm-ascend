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

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import vllm.v1.engine.core as _engine_core_mod
from vllm.logger import logger
from vllm.v1.engine.core import EngineCoreProc

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc, _balance_scheduling_enabled
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc, _get_dyntra_lb_config
from vllm_ascend.patch.platform.patch_dyntra_lb_core import _print_rank_0 as dyntra_print_rank_0

_PATCHED = False

_OriginalRunEngineCore = EngineCoreProc.run_engine_core
_OriginalEngineCoreShutdown = _engine_core_mod.EngineCore.shutdown
_OriginalEngineCoreProcHandleShutdown = _engine_core_mod.EngineCoreProc._handle_shutdown


def _call_export_api(port: int, path: str, workload: str) -> dict[str, Any]:
    body = json.dumps({"path": path, "workload": workload}, separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/eplb/export_placement",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode())


def _export_eplb_dp_aware(ports: list[int], path: str, workload: str) -> list[Any]:
    """Export through both P DP API servers concurrently."""
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        return list(pool.map(lambda port: _call_export_api(port, path, workload), ports))


def _export_eplb_from_config(self):
    if getattr(self, "_eplb_export_attempted", False):
        return
    self._eplb_export_attempted = True
    additional_config = getattr(self.vllm_config, "additional_config", None) or {}
    eplb_config = additional_config.get("eplb_config") or {}
    if isinstance(eplb_config, dict):
        export_path = eplb_config.get("export_path")
        if export_path and self.vllm_config.parallel_config.enable_eplb:
            workload = eplb_config.get("export_workload") or "MRv2 dynamic EPLB deployment"
            try:
                parallel_config = self.vllm_config.parallel_config
                if getattr(parallel_config, "data_parallel_size", 1) > 1:
                    return
                result = self.model_executor.collective_rpc(
                    "export_eplb_placement",
                    args=(export_path, workload),
                    timeout=1800,
                )
                logger.info("Exported EPLB deployment on EngineCore shutdown: %s", result)
            except Exception:
                logger.exception("Failed to export EPLB deployment during EngineCore shutdown")


def _shutdown_with_eplb_export(self):
    _export_eplb_from_config(self)
    _OriginalEngineCoreShutdown(self)


def _handle_shutdown_with_eplb_export(self):
    if getattr(self, "shutdown_state", None) == _engine_core_mod.EngineShutdownState.REQUESTED:
        _export_eplb_from_config(self)
    return _OriginalEngineCoreProcHandleShutdown(self)


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

    # Patch EngineCoreProc.run_engine_core
    # And in _run_engine_core_patch_func:
    # 1. Patch EngineCore.__init__ by _apply_profiling_patches
    # 2. Patch class DPEngineCoreProc by _patch_dp_engine_core_proc
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)
    _engine_core_mod.EngineCore.shutdown = _shutdown_with_eplb_export
    _engine_core_mod.EngineCoreProc._handle_shutdown = _handle_shutdown_with_eplb_export


_apply_patch()
