#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this updator.
import time
from queue import Empty

import numpy
import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess


class EplbUpdator:
    def __init__(self, eplb_config, loader: D2DExpertWeightLoader, eplb_process: EplbProcess, process):
        self.eplb_config = eplb_config
        self.multi_stage = eplb_config.eplb_policy_type == 3
        self.global_slots = eplb_config.num_redundant_experts if eplb_config.uses_global_expert_pool else 0
        self.init_eplb(self.eplb_config.expert_map_path, process)
        self.eplb_loader = loader
        self.eplb_process = eplb_process
        self.shared_dict = self.eplb_process.shared_dict
        self.comm_group = get_dynamic_eplb_group()

    def set_adaptor(self, adaptor: VllmEplbAdaptor):
        self.pp_rank = get_pp_group().rank_in_group
        self.adaptor = adaptor
        self.num_moe_layers = self.adaptor.num_moe_layers
        local_load = self.adaptor.get_rank_expert_workload()
        self.world_size = dist.get_world_size()
        self.device = local_load.device
        self.eplb_loader.num_layers = self.adaptor.num_dense_layers + self.adaptor.num_moe_layers

    def init_eplb(self, expert_map_path, process):
        self.rank_id = dist.get_rank()
        self.num_expert_load_gather = 10
        self.periodic_load_gather = True
        self.expert_heat_collection_interval: torch.int64 = self.eplb_config.expert_heat_collection_interval
        self.expert_map_path = expert_map_path
        self.expert_map_record_path = self.eplb_config.expert_map_record_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.expert_heat_collection_interval
                self.periodic_load_gather = False
        except Exception:
            logger.debug("[eplb/updator] VLLM_ALLOW_EXPERT_LOAD_COLLECTING unavailable in current vllm version.")
            self.num_expert_load_gather = self.expert_heat_collection_interval
            self.periodic_load_gather = False

        self.reqs = []
        self.update_info_all = []
        self.global_update_info = {}
        self.update_plan_active = False
        self.awaiting_plan = False
        self.pending_update_info = None
        self.global_update_steps = []
        self.global_pending_step = None
        self.global_transfer_reqs = []
        self.global_total_steps = 0
        self.global_plan_started_at = 0.0

        self.cur_iterations: torch.int64 = 0

        self.algorithm_execution_interval: torch.int64 = self.eplb_config.algorithm_execution_interval

        self.process = process

        logger.info("[eplb/updator] Launched EPLB subprocess, pid=%s", self.process.pid)

    def update_iteration(self):
        self.cur_iterations += 1
        if self.cur_iterations == (
            self.expert_heat_collection_interval + self.algorithm_execution_interval + self.num_moe_layers
        ):
            logger.debug("[eplb/updator] Full EPLB cycle completed, clearing moe loads and resetting iteration counter")
            if self.expert_map_record_path is not None:
                self.adaptor._export_tensor_to_file(self.shared_dict["expert_maps"], self.expert_map_record_path)

            self.adaptor.clear_all_moe_loads()
            self.cur_iterations = 0
            self.update_plan_active = False
            self.awaiting_plan = False
            self.pending_update_info = None
            self.global_update_info = {}
            self.global_update_steps = []
            self.global_pending_step = None
            self.global_transfer_reqs = []
            self.global_total_steps = 0
            self.global_plan_started_at = 0.0

    def get_update_info_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval + self.algorithm_execution_interval - 1)

    def wakeup_eplb_worker_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval - 1)

    def update_expert_weight_flag(self):
        weight_update_counter = self.cur_iterations - (
            self.expert_heat_collection_interval + self.algorithm_execution_interval
        )
        return weight_update_counter >= 0 and weight_update_counter < self.num_moe_layers

    def wakeup_eplb_worker(self):
        self.eplb_process.planner_q.put(1)

    def _poll_global_update_plan(self) -> None:
        if self.update_plan_active or not (self.get_update_info_flag() or self.awaiting_plan):
            return

        planner_state = 1
        if self.pending_update_info is None:
            try:
                self.pending_update_info = self.eplb_process.block_update_q.get_nowait()
            except Empty:
                planner_state = 0 if self.process.is_alive() else -1

        readiness = torch.tensor([planner_state], dtype=torch.int32, device=self.device)
        dist.all_reduce(
            readiness,
            op=dist.ReduceOp.MIN,
            group=self.comm_group.device_group,
        )
        global_state = int(readiness.item())
        if global_state < 0:
            raise RuntimeError("An EPLB planner exited before publishing an update plan")
        if global_state == 0:
            self.awaiting_plan = True
            return

        if not isinstance(self.pending_update_info, dict):
            raise RuntimeError("EPLB policy 4 published an invalid update plan")
        self.global_update_info = self.pending_update_info
        self.pending_update_info = None
        self.awaiting_plan = False
        self.update_plan_active = bool(self.global_update_info.get("changed", False))
        if self.update_plan_active:
            self.global_update_steps = list(self.global_update_info.get("steps", []))
            if not self.global_update_steps:
                raise RuntimeError("EPLB policy 4 published a changed plan without migration steps")
            self.global_total_steps = len(self.global_update_steps)
            self.global_plan_started_at = time.perf_counter()

    def _start_global_update_step(self) -> None:
        if not self.update_plan_active or self.global_pending_step is not None:
            return
        # An eager kernel from the preceding iteration may still read a shared
        # slot. Drain it before changing maps or writing that slot via HCCL.
        torch.npu.current_stream().synchronize()
        self.global_pending_step = self.global_update_steps.pop(0)
        self.eplb_loader.apply_global_map_updates(self.global_pending_step["deactivate"])
        self.global_transfer_reqs = self.eplb_loader.start_global_slot_transfer(self.global_pending_step)

    def forward_before(self):
        if self.global_slots:
            self._poll_global_update_plan()
            self._start_global_update_step()
            return

        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.get_update_info_flag():
            self.update_info_all = self.eplb_process.block_update_q.get()
        if self.update_expert_weight_flag():
            with record_function_or_nullcontext("EPLB generate p2p task"):
                (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = (
                    self.update_info_all.pop(0)
                )
                log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
                self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
                updated_expert_map_this_rank = torch.from_numpy(numpy.array(updated_expert_map))
                self.eplb_loader.generate_expert_d2d_transfer_task(
                    expert_send_info,
                    expert_recv_info,
                    updated_expert_map_this_rank,
                    layer_id,
                )

                # set asynchronous stream for d2d expert weight update
                self.reqs = []
                self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def forward_end(self, eplb_heat_collection_status: bool = True):
        if self.wakeup_eplb_worker_flag():
            with record_function_or_nullcontext("EPLB gather moe load"):
                self.compute_and_set_moe_load()
                self.wakeup_eplb_worker()

        if self.global_slots and self.global_pending_step is not None:
            self.eplb_loader.finish_global_slot_transfer(
                self.global_transfer_reqs,
                self.global_pending_step,
            )
            self.global_pending_step = None
            self.global_transfer_reqs = []
            if not self.global_update_steps:
                self.shared_dict["expert_maps"] = torch.tensor(
                    self.global_update_info["global_maps"], dtype=torch.int32
                )
                self.update_plan_active = False
                if self.rank_id == 0:
                    logger.info(
                        "[eplb/global] Migration completed steps=%s elapsed_ms=%.3f",
                        self.global_total_steps,
                        (time.perf_counter() - self.global_plan_started_at) * 1000.0,
                    )

        if not self.global_slots and self.update_expert_weight_flag() and self.expert_map_record_path is None:
            self.eplb_loader.update_expert_map_and_weight(self.reqs)

        # One circle of eplb update includes expert_heat_collection_interval + algorithm_execution_interval
        # + num_moe_layers (for weight update). In expert_heat_collection stage, we only update the counter
        # when eplb_heat_collection_status is True. In later stages, the counter is always updated.
        # TODO(Angazenn): Decouple algorithm execution && weight update with heat collection iterations.
        if (
            not self.awaiting_plan
            and not (self.global_slots and self.update_plan_active)
            and (self.cur_iterations >= self.expert_heat_collection_interval - 1 or eplb_heat_collection_status)
        ):
            self.update_iteration()

    def compute_and_set_moe_load(self):
        local_load = self.adaptor.get_rank_expert_workload().unsqueeze(1)
        moe_load = self.comm_group.all_gather(local_load, dim=1).cpu()

        if self.multi_stage:
            moe_load = moe_load.permute(2, 0, 1, 3)

        self.shared_dict["moe_load"] = moe_load
        logger.debug("[eplb/updator] Updated shared_dict['moe_load'] shape=%s", moe_load.shape)

        return moe_load

    def warm_up_eplb(self):
        logger.info("[eplb/updator] Starting EPLB warm-up, rank=%s, world_size=%s", self.rank_id, self.world_size)
        if not self.global_slots:
            self.eplb_loader.initialize_redundant_expert_weights()
        self.shared_dict["expert_maps"] = self.adaptor.get_global_expert_map()
        self.compute_and_set_moe_load()

        src_tensor = torch.empty((1,), device=self.device)

        comm_op_list = []
        reqs = []

        for dst_rank in range(self.comm_group.world_size):
            if dst_rank == self.comm_group.rank_in_group:
                continue
            global_dst = self.comm_group.ranks[dst_rank]
            comm_op_list.append(dist.P2POp(dist.isend, src_tensor, global_dst, group=self.comm_group.device_group))

        for src_rank in range(self.comm_group.world_size):
            if src_rank == self.comm_group.rank_in_group:
                continue
            global_src = self.comm_group.ranks[src_rank]
            comm_op_list.append(dist.P2POp(dist.irecv, src_tensor, global_src, group=self.comm_group.device_group))
        if comm_op_list:
            reqs = dist.batch_isend_irecv(comm_op_list)

        for req in reqs:
            req.wait()
        logger.info("[eplb/updator] EPLB warm-up completed")

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logger.info("[eplb/updator] EPLB subprocess terminated")
