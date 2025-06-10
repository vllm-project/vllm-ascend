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
import torch
import torch.distributed as dist
from multiprocessing import Queue, Manager

from vllm.logger import logger
from vllm_ascend.eplb.core.worker.eplb_worker import EplbProcess
from vllm_ascend.eplb.core.loader.device_transfer_loader import D2DExpertWeightLoader

class EplbUpdator:

    def __init__(self):
        self.init_eplb()

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor
        self.eplb_loader = D2DExpertWeightLoader(eplb_adaptor=self.adaptor)

    def init_eplb(self):

        self.num_iterations: torch.int64 = 10

        self.num_moe_layers = 2
        self.weight_update_counter = 0
        self.expert_map_initialized = False
        self.update_in_flight = False

        self.reqs = []

        self.cur_iterations: torch.int64 = 0

        self.planner_block_queue = Queue()
        self.block_update_queue = Queue(maxsize=1)

        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers, world_size, num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None
        })

        self.eplb = EplbProcess(
            shared_dict = self.shared_dict,
            planner_q = self.planner_block_queue,
            block_update_q = self.block_update_queue,
            policy_type = 0,
            enable_d2d = True
        )

        self.eplb_process = self.eplb._launch_process()

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})")


    def get_update_iteration(self):
        self.cur_iterations = self.cur_iterations + 1
        return self.cur_iterations % self.num_iterations == 0

    def get_init_expert_map(self):
        try:
            if not self.expert_map_initialized:
                self.shared_dict["expert_maps"] = self.adaptor.get_init_expert_map(self.num_moe_layers)
                self.expert_map_initialized = True
        except Exception as e:
            logger.warning(f"[ModelRunner] Failed to wake EPLB process: {e}", exc_info=True)

    def wakeup_eplb_worker(self):
        self.planner_block_queue.put(1)

    def forward_before(self):
        if self.update_in_flight and self.weight_update_counter < self.num_moe_layers:
            (expert_send_info, expert_recv_info, updated_expert_map, layer_id) = self.block_update_queue.get()
            rank_id = torch.distributed.get_rank()
            expert_send_info_this_rank = expert_send_info[rank_id] if rank_id in expert_send_info else []
            expert_recv_info_this_rank = expert_recv_info[rank_id] if rank_id in expert_recv_info else []
            # TODO: layer_id + 3 should be replaced by configuration
            self.eplb_loader.generate_expert_d2d_transfer_task(expert_send_info_this_rank,
                expert_recv_info_this_rank, updated_expert_map[rank_id], layer_id + 3)
            self.weight_update_counter += 1
            if self.weight_update_counter == self.num_moe_layers:
                self.update_in_flight = False

        # set asynchronous stream for d2d expert weight update
        self.reqs = []
        self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def forward_end(self):

        self.get_init_expert_map()

        if not self.update_in_flight and self.get_update_iteration():
            moe_load = self.compute_and_set_moe_load()
            self.wakeup_eplb_worker()
            self.update_in_flight = True

        self.eplb_loader.update_expert_map_and_weight(self.reqs)

    def compute_and_set_moe_load(self):
        local_load = self.adaptor.get_rank_expert_workload(self.num_moe_layers)

        self._gather_buffer = None
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        if dist.is_initialized():
            device = local_load.device
            if self._gather_buffer is None:
                shape = (self.world_size, *local_load.shape)
                self._gather_buffer = torch.empty(shape,
                                                  dtype=local_load.dtype,
                                                  device=device)

            dist.all_gather_into_tensor(self._gather_buffer, local_load)

            moe_load = self._gather_buffer.permute(1, 0, 2).contiguous()
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")
        else:
            moe_load = local_load.unsqueeze(1)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")
        return moe_load

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.eplb_process.is_alive():
            self.eplb_process.terminate()
            self.eplb_process.join()
            logger.info("[ModelRunner] EPLB process terminated")
