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


def _patch_sync_draft_d2h_skip() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(EngineCore.post_step, "_vllm_ascend_sync_draft_skip_patched", False):
        orig_post_step = EngineCore.post_step

        @wraps(orig_post_step)
        def _patched_post_step(self, model_executed: bool) -> None:
            vllm_config = getattr(self, "vllm_config", None)
            use_v2 = bool(
                getattr(
                    vllm_config,
                    "use_v2_model_runner",
                    getattr(getattr(self, "scheduler", None), "use_v2_model_runner", False),
                )
            )
            model_cfg = getattr(vllm_config, "model_config", None)
            is_diffusion = bool(getattr(model_cfg, "is_diffusion", False))
            if (
                use_v2
                and not is_diffusion
                and getattr(self, "check_for_draft_tokens", False)
                and not getattr(self, "async_scheduling", False)
                and model_executed
                and not any(r.use_structured_output for r in self.scheduler.running)
            ):
                return None
            return orig_post_step(self, model_executed)

        _patched_post_step._vllm_ascend_sync_draft_skip_patched = True  # type: ignore[attr-defined]
        EngineCore.post_step = _patched_post_step  # type: ignore[method-assign]

    if not getattr(Scheduler.update_from_output, "_vllm_ascend_sync_draft_skip_patched", False):
        orig_update_from_output = Scheduler.update_from_output

        @wraps(orig_update_from_output)
        def _patched_update_from_output(self, scheduler_output, model_runner_output):
            result = orig_update_from_output(self, scheduler_output, model_runner_output)
            model_cfg = getattr(self.vllm_config, "model_config", None)
            if (
                not self.scheduler_config.async_scheduling
                and getattr(self, "use_v2_model_runner", False)
                and not getattr(model_cfg, "is_diffusion", False)
                and self.num_spec_tokens > 0
                and not getattr(scheduler_output, "skip_drafts", False)
            ):
                num_spec = getattr(scheduler_output, "num_spec_tokens_to_schedule", 0)
                if num_spec > 0:
                    req_id_to_index = model_runner_output.req_id_to_index
                    sampled_token_ids = model_runner_output.sampled_token_ids
                    for req_id in model_runner_output.req_ids:
                        request = self.requests.get(req_id)
                        if (
                            request is None
                            or request.is_finished()
                            or request.is_prefill_chunk
                            or request.use_structured_output
                        ):
                            continue
                        req_idx = req_id_to_index.get(req_id)
                        if req_idx is not None and sampled_token_ids[req_idx]:
                            request.spec_token_ids = [-1] * num_spec
            return result

        _patched_update_from_output._vllm_ascend_sync_draft_skip_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = _patched_update_from_output  # type: ignore[method-assign]


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Patch EngineCore.post_step
    pp_mtp_patch_post_step()
    # Skip redundant synchronous take_draft_token_ids() D2H RPC under MRV2.
    _patch_sync_draft_d2h_skip()
    # Patch EngineCoreProc.run_engine_core
    # And in _run_engine_core_patch_func:
    # 1. Patch EngineCore.__init__ by _apply_profiling_patches
    # 2. Patch class DPEngineCoreProc by _patch_dp_engine_core_proc
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)


_apply_patch()
