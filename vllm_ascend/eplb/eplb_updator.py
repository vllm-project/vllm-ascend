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

from multiprocessing import Manager, Queue

import numpy
import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess


class EplbUpdator:

    def __init__(self, expert_map_path):
        self.ascend_config = get_ascend_config()
        self.init_eplb(expert_map_path)

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor
        self.eplb_loader = D2DExpertWeightLoader(eplb_adaptor=self.adaptor)
        self.num_moe_layers = self.adaptor.num_moe_layers
        self.global_expert_num = self.adaptor.global_expert_num

    def init_eplb(self, expert_map_path):
        self.rank_id = dist.get_rank()
        self.num_expert_load_gather = 10
        self.periodic_load_gather = True
        self.num_iterations_eplb_update: torch.int64 = self.ascend_config.num_iterations_eplb_update
        self.expert_map_path = expert_map_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.num_iterations_eplb_update
                self.periodic_load_gather = False
        except Exception:
            self.num_expert_load_gather = self.num_iterations_eplb_update
            self.periodic_load_gather = False

        self.expert_map_initialized = False
        self.gate_eplb = self.ascend_config.gate_eplb

        self.reqs = []
        self.update_info_all = []

        self.cur_iterations: torch.int64 = 0

        self.num_wait_worker_iterations: torch.int64 = self.ascend_config.num_wait_worker_iterations

        self.planner_block_queue: Queue = Queue()
        self.block_update_queue: Queue = Queue(maxsize=1)

        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers, world_size, num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None,
        })

        self.eplb = EplbProcess(shared_dict=self.shared_dict,
                                planner_q=self.planner_block_queue,
                                block_update_q=self.block_update_queue,
                                policy_type=1,
                                enable_d2d=True)

        self.eplb_process = self.eplb._launch_process()

        logger.info(
            f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})"
        )

    def update_iteration(self):
        self.cur_iterations += 1
        if self.cur_iterations == (self.num_iterations_eplb_update + \
                                   self.num_wait_worker_iterations + self.num_moe_layers):
            self.adaptor.model.clear_all_moe_loads()
            if not self.gate_eplb:
                self.cur_iterations = 0

    def get_update_info_flag(self):
        return self.cur_iterations == (self.num_iterations_eplb_update +
                                       self.num_wait_worker_iterations - 1)

    def wakeup_eplb_worker_flag(self):
        return self.cur_iterations == (self.num_iterations_eplb_update - 1)

    def update_expert_weight_flag(self):
        weight_update_counter = self.cur_iterations - (
            self.num_iterations_eplb_update + self.num_wait_worker_iterations)
        return (weight_update_counter >= 0
                and weight_update_counter < self.num_moe_layers)

    def get_init_expert_map(self):
        try:
            if not self.expert_map_initialized:
                self.shared_dict[
                    "expert_maps"] = self.adaptor.get_init_expert_map_from_file(
                        self.num_moe_layers, self.expert_map_path)
                self.expert_map_initialized = True
        except Exception as e:
            logger.warning(f"[ModelRunner] Failed to wake EPLB process: {e}",
                           exc_info=True)

    def wakeup_eplb_worker(self):
        self.planner_block_queue.put(1)

    def forward_before(self):
        if self.update_expert_weight_flag():
            (expert_send_info, expert_recv_info, updated_expert_map,
             log2phy_map, layer_id) = self.update_info_all.pop(0)
            log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
            self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(
                numpy.array(updated_expert_map))
            # logger.info(f"check update info, layer = {layer_id}, send = {expert_send_info_this_rank}, recv = {expert_recv_info_this_rank}")
            self.eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info, expert_recv_info,
                updated_expert_map_this_rank,
                layer_id + self.adaptor.num_dense_layers)

            # set asynchronous stream for d2d expert weight update
            self.reqs = []
            self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def take_update_info_from_eplb_process(self):
        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.get_update_info_flag():
            self.update_info_all = self.block_update_queue.get()

    def forward_end(self):
        if self.wakeup_eplb_worker_flag():
            self.compute_and_set_moe_load(is_clear=True)
            self.wakeup_eplb_worker()

        if self.update_expert_weight_flag():
            self.eplb_loader.update_expert_map_and_weight(self.reqs)

        self.update_iteration()

    def compute_and_set_moe_load(self, is_clear=False):
        local_load = self.adaptor.get_rank_expert_workload()

        self._gather_buffer = None
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.device = local_load.device
            if self._gather_buffer is None:
                shape = (self.world_size, *local_load.shape)
                self._gather_buffer = torch.empty(shape,
                                                  dtype=local_load.dtype,
                                                  device=self.device)

            dist.all_gather_into_tensor(self._gather_buffer, local_load)

            moe_load = self._gather_buffer.permute(1, 0, 2)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(
                f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}"
            )
        else:
            moe_load = local_load.unsqueeze(1)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(
                f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}"
            )
        return moe_load

    def warm_up_eplb(self):

        self.get_init_expert_map()
        self.compute_and_set_moe_load()

        src_tensor = torch.empty((1, ), device=self.device)
        self_rank = dist.get_rank()

        comm_op_list = []

        for dst_rank in range(self.world_size):
            if dst_rank == self_rank:
                continue
            comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

        for src_rank in range(self.world_size):
            if src_rank == self_rank:
                continue
            comm_op_list.append(dist.P2POp(dist.irecv, src_tensor, src_rank))
        if comm_op_list:
            reqs = dist.batch_isend_irecv(comm_op_list)

        for req in reqs:
            req.wait()

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.eplb_process.is_alive():
            self.eplb_process.terminate()
            self.eplb_process.join()
            logger.info("[ModelRunner] EPLB process terminated")
