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
#

import datetime
import itertools

import numpy as np
import torch
import torch.distributed as dist
import vllm.v1.engine.core as _engine_core_mod
from vllm.config import ParallelConfig
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

import vllm_ascend.patch.platform.patch_balance_schedule as _balance_patch
import vllm_ascend.patch.platform.patch_nonbsp_request_status  # noqa: F401
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.nonbsp_balance_load import balance_load
from vllm.v1.request import Request, RequestStatus


def _nonbsp_enabled(vllm_config) -> bool:
    try:
        return bool(get_ascend_config().NONBSP_ENABLE)
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("NONBSP_ENABLE", 0))


def _print_rank_0(message: str, dp_rank: int) -> None:
    if dp_rank != 0:
        return
    print(f"{datetime.datetime.now()} | [nonbsp] {message}", flush=True)


def _print_requests_by_rank(requests_by_rank, dp_rank: int) -> None:
    if dp_rank != 0:
        return

    print("\n" + "=" * 85, flush=True)
    print("[nonbsp] requests_by_rank (DP Workload Status):", flush=True)
    for rank, (blocks, split_idx) in enumerate(requests_by_rank):
        running = blocks[:split_idx]
        waiting = [block for block in blocks[split_idx:] if block > 0]
        running_items = ", ".join(f"{block:3d}" for block in running)
        waiting_items = ", ".join(f"{block:3d}" for block in waiting)
        running_str = f"Run({len(running)}): [{running_items}]"
        waiting_str = f"Wait({len(waiting)}): [{waiting_items}]"
        print(
            f" DP{rank} | {running_str:<55} | {waiting_str}",
            flush=True,
        )
    print("=" * 85, flush=True)


def _print_modifications(modifications, dp_rank: int) -> None:
    if dp_rank != 0:
        return

    print("\n" + "=" * 60, flush=True)
    print("[nonbsp] modifications:", flush=True)
    for rank, modification in enumerate(modifications):
        out_items = ", ".join(
            f"{block:3d}" for block in modification.get("out_blk", [])
        )
        in_items = ", ".join(
            f"{block:3d}" for block in modification.get("in_blk", [])
        )
        out_str = f"Out: [{out_items}]"
        in_str = f"In: [{in_items}]"
        freeze_str = f"Freeze: {bool(modification.get('freeze', False))}"
        print(
            f" DP{rank} | {out_str:<15} | {in_str:<15} | {freeze_str}",
            flush=True,
        )
    print("=" * 60, flush=True)


_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS = (
    DPEngineCoreProc._has_global_unfinished_reqs
)


def _has_global_unfinished_reqs_with_step_counter(
    self, local_unfinished: bool
) -> bool:
    print(
        f"{datetime.datetime.now()} | _has_global_unfinished_reqs | "
        f"step_counter: {self.step_counter}",
        flush=True,
    )
    return _ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS(self, local_unfinished)


DPEngineCoreProc._has_global_unfinished_reqs = (
    _has_global_unfinished_reqs_with_step_counter
)


class NonBSPDPEngineCoreProc(DPEngineCoreProc):

    def _init_data_parallel(self, vllm_config):
        super()._init_data_parallel(vllm_config)

        ascend_config = init_ascend_config(vllm_config)
        self._lb_enable = int(ascend_config.NONBSP_ENABLE)
        self._lb_start_step = int(ascend_config.NONBSP_START_STEP)
        self._lb_end_step = int(ascend_config.NONBSP_END_STEP)
        self._lb_threshold = float(ascend_config.NONBSP_BUBBLE_THRESHOLD)
        self._lb_long_req_threshold = int(
            ascend_config.NONBSP_LONG_REQ_BLOCK_THRESHOLD)
        self._lb_dynamic_max_step = int(ascend_config.NONBSP_DYNAMIC_MAX_STEP)
        self._lb_dynamic_enable = False
        self._lb_dynamic_step = 0
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        _print_rank_0(f"NONBSP_ENABLE = {self._lb_enable}", self.dp_rank)
        _print_rank_0(
            f"NONBSP_START_STEP = {self._lb_start_step}", self.dp_rank
        )
        _print_rank_0(f"NONBSP_END_STEP = {self._lb_end_step}", self.dp_rank)
        _print_rank_0(
            f"NONBSP_BUBBLE_THRESHOLD = {self._lb_threshold}", self.dp_rank
        )
        _print_rank_0(
            "NONBSP_LONG_REQ_BLOCK_THRESHOLD = "
            f"{self._lb_long_req_threshold}",
            self.dp_rank,
        )
        _print_rank_0(
            f"NONBSP_DYNAMIC_MAX_STEP = {self._lb_dynamic_max_step}",
            self.dp_rank,
        )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        dp_size = dist.get_world_size(self.dp_group)
        max_slots = max_num_seqs * 2
        self._lb_dp_size_cached = dp_size
        self._lb_max_slots_cached = max_slots
        self._lb_max_num_seqs = max_num_seqs

        full_len = max_slots + 2
        self._lb_data_np = np.zeros(full_len, dtype=np.int32)
        self._lb_data_t = torch.as_tensor(self._lb_data_np)
        self._lb_all_data_np = [
            np.zeros(full_len, dtype=np.int32) for _ in range(dp_size)
        ]
        self._lb_all_data_t_buf = [
            torch.as_tensor(arr) for arr in self._lb_all_data_np
        ]
        self._lb_dynamic_flag_np = np.zeros(2, dtype=np.int32)
        self._lb_dynamic_flag_t = torch.as_tensor(self._lb_dynamic_flag_np)

    def add_request(self, request: Request, request_wave: int = 0):
        if self._lb_enable == 2:
            blk_size = self.scheduler.block_size
            blk_num = (len(request.all_token_ids) + blk_size - 1) // blk_size
            if blk_num > self._lb_long_req_threshold:
                self._lb_pending_long_req = True
                self._lb_pending_long_req_blk = max(
                    self._lb_pending_long_req_blk, blk_num)
        super().add_request(request, request_wave)

    def run_balance_load(self):
        static_lb_enable = self._lb_enable == 1
        dynamic_lb_mode = self._lb_enable == 2
        dynamic_activated_this_step = False

        if dynamic_lb_mode and not self._lb_dynamic_enable:
            self._lb_dynamic_flag_np[0] = int(self._lb_pending_long_req)
            self._lb_dynamic_flag_np[1] = self._lb_pending_long_req_blk
            dist.all_reduce(self._lb_dynamic_flag_t,
                            op=dist.ReduceOp.MAX,
                            group=self.dp_group)
            if self._lb_dynamic_flag_np[0]:
                self._lb_dynamic_enable = True
                self._lb_dynamic_step = 0
                dynamic_activated_this_step = True
                _print_rank_0(
                    "Dynamic LB enabled by long request: "
                    f"blk_num={self._lb_dynamic_flag_np[1]}, "
                    f"threshold={self._lb_long_req_threshold}, "
                    f"current_wave={self.current_wave}, "
                    f"step_counter={self.step_counter}",
                    self.dp_rank,
                )
            self._lb_pending_long_req = False
            self._lb_pending_long_req_blk = 0

        lb_active = ((static_lb_enable or self._lb_dynamic_enable)
                     and not dynamic_activated_this_step
                     and self.step_counter >= self._lb_start_step
                     and (self._lb_end_step < 0
                          or self.step_counter < self._lb_end_step))
        self.scheduler._lb_kv_prefetch_enabled = lb_active
        if lb_active:
            has_new_long_req = self._do_lb_allgather()
            if dynamic_lb_mode and self._lb_dynamic_enable:
                if has_new_long_req:
                    self._lb_dynamic_step = 0
                else:
                    self._lb_dynamic_step += 1
                    if self._lb_dynamic_step >= self._lb_dynamic_max_step:
                        self._lb_dynamic_enable = False
                        _print_rank_0(
                            "Dynamic LB disabled after "
                            f"{self._lb_dynamic_step} steps without long request: "
                            f"current_wave={self.current_wave}, "
                            f"step_counter={self.step_counter}",
                            self.dp_rank,
                        )
        else:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False

    def _process_engine_step(self) -> bool:
        # Match the original NonBSP scheduling order: process newly queued
        # requests, compute the DP load-balancing plan, then schedule/execute
        # the current step.
        self.run_balance_load()
        return super()._process_engine_step()

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        if not result:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False
            print(
                f"{datetime.datetime.now()} | [nonbsp] run_busy_loop() | "
                "No engines running, step_counter set to 0, "
                "current_wave incremented to "
                f"{self.current_wave + 1}.",
                flush=True,
            )
        return result

    def _do_lb_allgather(self) -> bool:
        max_slots = self._lb_max_slots_cached
        dp_size = self._lb_dp_size_cached
        max_num_seqs = self._lb_max_num_seqs

        blk_size = self.scheduler.block_size
        running_blks = [
            (len(req.all_token_ids) + blk_size - 1) // blk_size
            for req in self.scheduler.running
        ]
        waiting_blks = [
            (len(req.all_token_ids) + blk_size - 1) // blk_size
            for req in itertools.chain(self.scheduler.waiting,
                                       self.scheduler.skipped_waiting)
            if req.status in (RequestStatus.WAITING, RequestStatus.LB_PAUSED)
        ]

        arr = self._lb_data_np
        arr[:] = 0
        run_cnt = min(len(running_blks), max_slots)
        wait_cnt = min(len(waiting_blks), max_slots - run_cnt)
        arr[:run_cnt] = running_blks[:run_cnt]
        arr[run_cnt:run_cnt + wait_cnt] = waiting_blks[:wait_cnt]
        arr[max_slots] = run_cnt
        arr[max_slots + 1] = int(self._lb_pending_long_req)
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        dist.all_gather(self._lb_all_data_t_buf,
                        self._lb_data_t,
                        group=self.dp_group)

        requests_by_rank = []
        has_new_long_req = False
        for rank_np in self._lb_all_data_np:
            split = int(rank_np[max_slots])
            blks = rank_np[:max_slots].tolist()
            requests_by_rank.append((blks, split))
            has_new_long_req = has_new_long_req or bool(rank_np[max_slots + 1])

        modifications = balance_load(requests_by_rank,
                                     dp_size,
                                     max_num_seqs=max_num_seqs,
                                     threshold=self._lb_threshold)
        _print_requests_by_rank(requests_by_rank, self.dp_rank)
        _print_modifications(modifications, self.dp_rank)
        self.scheduler.modifications = modifications[self.dp_rank]
        return has_new_long_req


_PreviousRunEngineCore = EngineCoreProc.run_engine_core
_UpstreamRunEngineCore = _balance_patch._ORIGINAL_RUN_ENGINE_CORE
_OriginalDPEngineCoreProc = DPEngineCoreProc


def _nonbsp_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if not _nonbsp_enabled(vllm_config):
        return _PreviousRunEngineCore(
            *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

    _print_rank_0("Enable NonBSP DP load balancing.", dp_rank)
    _engine_core_mod.DPEngineCoreProc = NonBSPDPEngineCoreProc
    try:
        return _UpstreamRunEngineCore(
            *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
    finally:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc


EngineCoreProc.run_engine_core = staticmethod(_nonbsp_run_engine_core)
