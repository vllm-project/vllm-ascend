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
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
import torch
import torch.distributed as dist
import vllm.v1.engine.core as _engine_core_mod
import vllm.v1.request as _request_module
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.request import Request

import vllm_ascend.patch.platform.patch_balance_schedule as _balance_patch
from vllm_ascend.ascend_config import NonBSPConfig, get_ascend_config, init_ascend_config
from vllm_ascend.core.scheduler_diagnostics import diagnostics_enabled

RequestStatus = _request_module.RequestStatus
_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS = DPEngineCoreProc._has_global_unfinished_reqs


class _RequestItem(TypedDict):
    blk_num: int
    newly_added: bool


class _CardData(TypedDict):
    card_idx: int
    running: list[_RequestItem]
    waiting: list[_RequestItem]
    tot_blk: int


class _Modification(TypedDict):
    out_blk: list[int]
    in_blk: list[int]
    freeze: bool


def _get_nonbsp_config(vllm_config) -> NonBSPConfig:
    try:
        return get_ascend_config().scheduler_config.nonbsp_config
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    scheduler_config = additional_config.get("scheduler_config") or {}
    if not isinstance(scheduler_config, dict):
        return NonBSPConfig()
    nonbsp_config = scheduler_config.get("nonbsp_config") or {}
    if not isinstance(nonbsp_config, dict):
        return NonBSPConfig()
    return NonBSPConfig(nonbsp_config)


def _nonbsp_enabled(vllm_config) -> bool:
    return _get_nonbsp_config(vllm_config).enabled


def _print_rank_0(message: str, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
        return
    print(f"{datetime.datetime.now()} | [nonbsp] {message}", flush=True)


def _print_requests_by_rank(requests_by_rank, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
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


def _print_modifications(modifications, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
        return

    print("\n" + "=" * 60, flush=True)
    print("[nonbsp] modifications:", flush=True)
    for rank, modification in enumerate(modifications):
        out_items = ", ".join(f"{block:3d}" for block in modification.get("out_blk", []))
        in_items = ", ".join(f"{block:3d}" for block in modification.get("in_blk", []))
        out_str = f"Out: [{out_items}]"
        in_str = f"In: [{in_items}]"
        freeze_str = f"Freeze: {bool(modification.get('freeze', False))}"
        print(
            f" DP{rank} | {out_str:<15} | {in_str:<15} | {freeze_str}",
            flush=True,
        )
    print("=" * 60, flush=True)


def _has_global_unfinished_reqs_with_diagnostics(self, local_unfinished: bool) -> bool:
    if diagnostics_enabled(self.vllm_config):
        print(
            f"{datetime.datetime.now()} | _has_global_unfinished_reqs | step_counter: {self.step_counter}",
            flush=True,
        )
    return _ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS(self, local_unfinished)


DPEngineCoreProc._has_global_unfinished_reqs = _has_global_unfinished_reqs_with_diagnostics


class NonBSPDPEngineCoreProc(DPEngineCoreProc):
    @staticmethod
    def _balance_load(
        requests_by_rank: Sequence[tuple[Sequence[int], int]],
        dev_num: int,
        max_num_seqs: int | None = None,
        max_iters: int = 1000,
        threshold: float = 5,
    ) -> list[_Modification]:
        modifications: list[_Modification] = [{"out_blk": [], "in_blk": [], "freeze": False} for _ in range(dev_num)]

        def calc_bubble(cards_info: Sequence[_CardData]) -> float:
            tot_blks = [c["tot_blk"] for c in cards_info]
            max_blk = max(tot_blks)
            avg_blk = sum(tot_blks) / dev_num
            if threshold <= 1:
                return (max_blk - avg_blk) / max_blk if max_blk > 0 else 0.0
            return max_blk - avg_blk

        def perform_swap(
            run_item: _RequestItem,
            wait_item: _RequestItem,
            card: _CardData,
        ) -> None:
            card_idx = card["card_idx"]
            modifications[card_idx]["out_blk"].append(run_item["blk_num"])
            modifications[card_idx]["in_blk"].append(wait_item["blk_num"])
            card["tot_blk"] += wait_item["blk_num"] - run_item["blk_num"]
            card["running"].remove(run_item)
            card["waiting"].remove(wait_item)
            card["running"].append(wait_item)
            card["waiting"].append(run_item)

        def perform_pull_in(
            wait_item: _RequestItem,
            card: _CardData,
        ) -> None:
            card_idx = card["card_idx"]
            modifications[card_idx]["in_blk"].append(wait_item["blk_num"])
            card["tot_blk"] += wait_item["blk_num"]
            card["waiting"].remove(wait_item)
            card["running"].append(wait_item)

        def fill_running_from_waiting(card: _CardData) -> bool:
            if max_num_seqs is None:
                return False
            pulled_any = False
            while len(card["running"]) < max_num_seqs and card["waiting"]:
                item = card["waiting"][0]
                item["newly_added"] = True
                perform_pull_in(item, card)
                pulled_any = True
            return pulled_any

        def find_best_swap(
            card: _CardData,
            mode: Literal["increase", "decrease"],
        ) -> tuple[_RequestItem, _RequestItem] | None:
            target_avg = sum(c["tot_blk"] for c in cards_data) / dev_num
            best_swap: tuple[_RequestItem, _RequestItem] | None = None
            best_dist = abs(card["tot_blk"] - target_avg)
            best_newly_added = False

            for run_item in card["running"]:
                for wait_item in card["waiting"]:
                    delta = wait_item["blk_num"] - run_item["blk_num"]
                    if mode == "increase" and delta <= 0:
                        continue
                    if mode == "decrease" and delta >= 0:
                        continue
                    new_tot = card["tot_blk"] + delta
                    new_avg = target_avg + delta / dev_num
                    new_dist = abs(new_tot - new_avg)
                    run_newly_added = run_item.get("newly_added", False)
                    if new_dist < best_dist or (new_dist == best_dist and run_newly_added and not best_newly_added):
                        best_dist = new_dist
                        best_swap = (run_item, wait_item)
                        best_newly_added = run_newly_added
            return best_swap

        def try_drop_for_throughput(max_card: _CardData) -> bool:
            card_idx = max_card["card_idx"]
            total_reqs = sum(len(c["running"]) for c in cards_data)
            if total_reqs <= 1:
                return False

            max_blk = max_card["tot_blk"]
            second_max_blk = max([c["tot_blk"] for c in cards_data if c["card_idx"] != max_card["card_idx"]] + [0])
            current_latency = 1200 + 19.2 * max_blk
            current_tp = total_reqs / current_latency
            best_drop: _RequestItem | None = None
            best_tp = current_tp
            best_newly_added = False

            for run_item in max_card["running"]:
                new_max_blk = max(
                    max_blk - run_item["blk_num"],
                    second_max_blk,
                )
                new_latency = 1200 + 19.2 * new_max_blk
                new_tp = (total_reqs - 1) / new_latency
                run_newly_added = run_item.get("newly_added", False)
                if new_tp > best_tp or (new_tp == best_tp and run_newly_added and not best_newly_added):
                    best_tp = new_tp
                    best_drop = run_item
                    best_newly_added = run_newly_added

            if best_drop:
                modifications[card_idx]["out_blk"].append(best_drop["blk_num"])
                max_card["tot_blk"] -= best_drop["blk_num"]
                max_card["running"].remove(best_drop)
                max_card["waiting"].append(best_drop)
                return True
            return False

        cards_data: list[_CardData] = []
        for rank in range(dev_num):
            blks, split = requests_by_rank[rank] if rank < len(requests_by_rank) else ([], 0)
            running: list[_RequestItem] = []
            waiting: list[_RequestItem] = []
            tot_run_blk = 0
            for index, blk_num in enumerate(blks):
                if blk_num == 0:
                    break
                req_obj: _RequestItem = {
                    "blk_num": blk_num,
                    "newly_added": False,
                }
                if index < split:
                    running.append(req_obj)
                    tot_run_blk += blk_num
                else:
                    waiting.append(req_obj)
            cards_data.append(
                {
                    "card_idx": rank,
                    "running": running,
                    "waiting": waiting,
                    "tot_blk": tot_run_blk,
                }
            )

        for card in cards_data:
            fill_running_from_waiting(card)

        iteration = 0
        while iteration < max_iters:
            cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)
            if calc_bubble(cards_data) < threshold:
                break

            swapped_this_round = False
            dropped_this_round = False
            min_card = cards_data[-1]
            best_min_swap = find_best_swap(min_card, mode="increase")
            if best_min_swap:
                perform_swap(best_min_swap[0], best_min_swap[1], min_card)
                swapped_this_round = True

            if swapped_this_round:
                if calc_bubble(cards_data) < threshold:
                    break
                cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)

            max_card = cards_data[0]
            best_max_swap = find_best_swap(max_card, mode="decrease")
            if best_max_swap:
                perform_swap(best_max_swap[0], best_max_swap[1], max_card)
                swapped_this_round = True

            if not swapped_this_round and try_drop_for_throughput(max_card):
                dropped_this_round = True

            if calc_bubble(cards_data) < threshold:
                break
            if not swapped_this_round and not dropped_this_round:
                break
            iteration += 1

        for mod in modifications:
            out_blks = mod["out_blk"][:]
            in_blks = mod["in_blk"][:]
            cancelled_out = [False] * len(out_blks)
            cancelled_in = [False] * len(in_blks)
            for in_index, in_blk in enumerate(in_blks):
                for out_index, out_blk in enumerate(out_blks):
                    if not cancelled_out[out_index] and out_blk == in_blk:
                        cancelled_out[out_index] = True
                        cancelled_in[in_index] = True
                        break
            mod["out_blk"] = [block for block, cancelled in zip(mod["out_blk"], cancelled_out) if not cancelled]
            mod["in_blk"] = [block for block, cancelled in zip(mod["in_blk"], cancelled_in) if not cancelled]

        for card in cards_data:
            rank = card["card_idx"]
            modifications[rank]["freeze"] = len(modifications[rank]["in_blk"]) == 0

        return modifications

    def _init_data_parallel(self, vllm_config):
        super()._init_data_parallel(vllm_config)

        ascend_config = init_ascend_config(vllm_config)
        nonbsp_config = ascend_config.scheduler_config.nonbsp_config
        self._lb_enable_diagnostics = nonbsp_config.enable_diagnostics
        self._lb_mode = nonbsp_config.mode
        self._lb_start_step = nonbsp_config.start_step
        self._lb_end_step = nonbsp_config.end_step
        self._lb_threshold = nonbsp_config.bubble_threshold
        self._lb_long_req_threshold = nonbsp_config.long_req_block_threshold
        self._lb_dynamic_max_step = nonbsp_config.dynamic_max_step
        self._lb_dynamic_enable = False
        self._lb_dynamic_step = 0
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        _print_rank_0("nonbsp_config.enabled = True", self.dp_rank, self._lb_enable_diagnostics)
        _print_rank_0(
            f"nonbsp_config.enable_diagnostics = {self._lb_enable_diagnostics}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.mode = {self._lb_mode}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.start_step = {self._lb_start_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.end_step = {self._lb_end_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.bubble_threshold = {self._lb_threshold}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.long_req_block_threshold = {self._lb_long_req_threshold}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"nonbsp_config.dynamic_max_step = {self._lb_dynamic_max_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
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
        self._lb_all_data_np = [np.zeros(full_len, dtype=np.int32) for _ in range(dp_size)]
        self._lb_all_data_t_buf = [torch.as_tensor(arr) for arr in self._lb_all_data_np]
        self._lb_dynamic_flag_np = np.zeros(2, dtype=np.int32)
        self._lb_dynamic_flag_t = torch.as_tensor(self._lb_dynamic_flag_np)

    def add_request(self, request: Request, request_wave: int = 0):
        if self._lb_mode == "dynamic":
            blk_size = self.scheduler.block_size
            blk_num = (len(request.all_token_ids) + blk_size - 1) // blk_size
            if blk_num > self._lb_long_req_threshold:
                self._lb_pending_long_req = True
                self._lb_pending_long_req_blk = max(self._lb_pending_long_req_blk, blk_num)
        super().add_request(request, request_wave)

    def run_balance_load(self):
        static_lb_enable = self._lb_mode == "static"
        dynamic_lb_mode = self._lb_mode == "dynamic"
        dynamic_activated_this_step = False

        if dynamic_lb_mode and not self._lb_dynamic_enable:
            self._lb_dynamic_flag_np[0] = int(self._lb_pending_long_req)
            self._lb_dynamic_flag_np[1] = self._lb_pending_long_req_blk
            dist.all_reduce(self._lb_dynamic_flag_t, op=dist.ReduceOp.MAX, group=self.dp_group)
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
                    self._lb_enable_diagnostics,
                )
            self._lb_pending_long_req = False
            self._lb_pending_long_req_blk = 0

        lb_active = (
            (static_lb_enable or self._lb_dynamic_enable)
            and not dynamic_activated_this_step
            and self.step_counter >= self._lb_start_step
            and (self._lb_end_step < 0 or self.step_counter < self._lb_end_step)
        )
        self.scheduler._lb_kv_prefetch_enabled = lb_active
        if lb_active:
            admission_candidates = self.scheduler.prepare_nonbsp_step()
            has_new_long_req = self._do_lb_allgather(admission_candidates)
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
                            self._lb_enable_diagnostics,
                        )
        else:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False

    def _process_engine_step(self) -> bool:
        # Local connector maintenance is not a coordinated DP wave step and
        # must not advance this rank's NonBSP collective sequence.
        local_unfinished = self.scheduler.has_unfinished_requests()
        if self.engines_running or local_unfinished:
            self.run_balance_load()
        return super()._process_engine_step()

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        if not result:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False
            if self._lb_enable_diagnostics:
                print(
                    f"{datetime.datetime.now()} | [nonbsp] run_busy_loop() | "
                    "No engines running, step_counter set to 0, "
                    "current_wave incremented to "
                    f"{self.current_wave + 1}.",
                    flush=True,
                )
        return result

    def _do_lb_allgather(self, admission_candidates: list[Request]) -> bool:
        max_slots = self._lb_max_slots_cached
        dp_size = self._lb_dp_size_cached
        max_num_seqs = self._lb_max_num_seqs

        blk_size = self.scheduler.block_size
        running_blks = [(len(req.all_token_ids) + blk_size - 1) // blk_size for req in self.scheduler.running]
        waiting_blks = [(len(req.all_token_ids) + blk_size - 1) // blk_size for req in admission_candidates]

        arr = self._lb_data_np
        arr[:] = 0
        run_cnt = min(len(running_blks), max_slots)
        wait_cnt = min(len(waiting_blks), max_slots - run_cnt)
        arr[:run_cnt] = running_blks[:run_cnt]
        arr[run_cnt : run_cnt + wait_cnt] = waiting_blks[:wait_cnt]
        arr[max_slots] = run_cnt
        arr[max_slots + 1] = int(self._lb_pending_long_req)
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        dist.all_gather(self._lb_all_data_t_buf, self._lb_data_t, group=self.dp_group)

        requests_by_rank = []
        has_new_long_req = False
        for rank_np in self._lb_all_data_np:
            split = int(rank_np[max_slots])
            blks = rank_np[:max_slots].tolist()
            requests_by_rank.append((blks, split))
            has_new_long_req = has_new_long_req or bool(rank_np[max_slots + 1])

        modifications = self._balance_load(
            requests_by_rank,
            dp_size,
            max_num_seqs=max_num_seqs,
            threshold=self._lb_threshold,
        )
        _print_requests_by_rank(
            requests_by_rank,
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_modifications(
            modifications,
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        self.scheduler.modifications = modifications[self.dp_rank]
        return has_new_long_req


_PreviousRunEngineCore = EngineCoreProc.run_engine_core
_UpstreamRunEngineCore = _balance_patch._OriginalRunEngineCore
_OriginalDPEngineCoreProc = DPEngineCoreProc


def _nonbsp_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    nonbsp_config = _get_nonbsp_config(vllm_config)
    if not nonbsp_config.enabled:
        return _PreviousRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

    _print_rank_0(
        "Enable NonBSP DP load balancing.",
        dp_rank,
        nonbsp_config.enable_diagnostics,
    )
    _engine_core_mod.DPEngineCoreProc = NonBSPDPEngineCoreProc
    try:
        return _UpstreamRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
    finally:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc


EngineCoreProc.run_engine_core = staticmethod(_nonbsp_run_engine_core)
