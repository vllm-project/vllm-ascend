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
import multiprocessing

from vllm.logger import logger
from vllm_ascend.eplb.core.worker.eplb_worker import EplbProcess

class EplbUpdator:

    def __init__(self, adaptor):
        self.adaptor = adaptor
        self.init_eplb()
    
    def init_eplb(self):

        self.update_iterations: torch.int64 = 1

        self.num_moe_layers = 2
        self.expert_map_initialized = False
        self.update_in_flight = False

        self.cur_iterations: torch.int64 = 0

        ctx = multiprocessing.get_context("spawn")
        self.manager = ctx.Manager()

        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers,num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None
        })

        self.eplb = EplbProcess(
            device_id = self.device,
            shared_dict = self.shared_dict,
            policy_type = 1,
            enable_d2d = True
        )

        self.planner_block_queue, self.block_update_queue, self.eplb_process = \
            self.eplb._launch_process()

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})")


    def get_update_iteration(self): 
        self.cur_iterations = self.cur_iterations + 1
        return self.cur_iterations % self.num_iterations == 0


    def get_init_expert_map(self):
        try:
            if not self.expert_map_initialized:
                self.adaptor.get_init_expert_map()
                self.expert_map_initialized = True
        except Exception as e:
            logger.warning(f"[ModelRunner] Failed to wake EPLB process: {e}", exc_info=True)


    def wakeup_eplb_worker(self):
        self.planner_block_queue.put(1)


    def do_eplb(self):

        self.get_init_expert_map()

        # step 1. expert workload aggregation
        if not self.update_in_flight and self.get_update_iteration():
            moe_load = self.compute_and_set_moe_load()
            self.wakeup_eplb_worker()
            self.update_in_flight = True

        # step 4. update expert map and expert weight tensor
        if self.update_in_flight:
            if not self.block_update_queue.empty():
                self.block_update_queue.get()
                rank_id = dist.get_rank()
                new_expert_map = self.shared_dict["expert_maps"][:, rank_id, :]
                self.model.update_all_expert_map(new_expert_map, self.num_moe_layers)
                #加载权重
                self.update_in_flight = False


    def compute_and_set_moe_load(self):
        moe_load = self.adaptor.get_rank_expert_workload(self.num_moe_layers)

        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(moe_load, op=dist.ReduceOp.SUM)
        self.shared_dict["moe_load"] = moe_load.to(torch.device("cpu"))

        logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] = {moe_load}")

        return moe_load


    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.eplb_process.is_alive():
            self.eplb_process.terminate()
            self.eplb_process.join()
            logger.info("[ModelRunner] EPLB process terminated")
