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
from typing import Dict, List
import torch.distributed as dist
import vllm.envs as envs
from multiprocessing import Queue, Manager

from vllm.logger import logger
from vllm_ascend.eplb.core.worker.eplb_worker import EplbProcess
from vllm_ascend.eplb.core.loader.device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.tool.eplb_utils import ExpertMapUtils

class EplbUpdator:

    def __init__(self, expert_map_path):
        self.init_eplb(expert_map_path)

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor
        self.eplb_loader = D2DExpertWeightLoader(eplb_adaptor=self.adaptor)
        self.num_moe_layers = self.adaptor.num_moe_layers
        self.global_expert_num = self.adaptor.global_expert_num

    def init_eplb(self, expert_map_path):
        self.num_expert_load_gather = 10
        self.redundant_enable = (expert_map_path != None)
        self.num_iterations: torch.int64 = 130
        self.expert_map_path = expert_map_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.num_iterations
        except Exception as e:
                self.num_expert_load_gather = self.num_iterations

        self.weight_update_counter = 0
        self.expert_map_initialized = False
        self.update_in_flight = False

        self.gate_eplb = True

        self.reqs = []
        self.update_info_all = []

        self.cur_iterations: torch.int64 = 0

        self.wait_worker_iterations: torch.int64 = 0
        self.num_wait_worker_iterations: torch.int64 = 20

        self.planner_block_queue = Queue()
        self.block_update_queue = Queue(maxsize=1)

        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers, world_size, num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None,
        })

        self.eplb = EplbProcess(
            shared_dict = self.shared_dict,
            planner_q = self.planner_block_queue,
            block_update_q = self.block_update_queue,
            redundant_enable = self.redundant_enable, 
            policy_type = 6,
            enable_d2d = True
        )

        self.eplb_process = self.eplb._launch_process()

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})")

    def get_update_iteration(self):
        self.cur_iterations = self.cur_iterations + 1
        load_gather_iteration = self.cur_iterations % self.num_expert_load_gather == 0 if not self.gate_eplb else self.cur_iterations == self.num_iterations 
        upate_iteration = self.cur_iterations % self.num_iterations == 0 if not self.gate_eplb else self.cur_iterations == self.num_iterations 
        return load_gather_iteration, upate_iteration

    def get_init_expert_map(self):
        try:
            if not self.expert_map_initialized:
                self.shared_dict["expert_maps"] = self.adaptor.get_init_expert_map_from_file(self.num_moe_layers, self.expert_map_path)
                self.expert_map_initialized = True
        except Exception as e:
            logger.warning(f"[ModelRunner] Failed to wake EPLB process: {e}", exc_info=True)

    def wakeup_eplb_worker(self):
        self.planner_block_queue.put(1)

    def forward_before(self):

        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.update_in_flight and self.weight_update_counter == 0 and self.wait_worker_iterations == self.num_wait_worker_iterations:
            self.wait_worker_iterations = 0
            packed_update_info = self.block_update_queue.get()
            self.update_info_all = self.unpack_update_batch(packed_update_info)
            self.weight_loading = True

        if self.update_in_flight and self.weight_loading and self.weight_update_counter < self.num_moe_layers:
            (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = self.update_info_all.pop(0)
            rank_id = torch.distributed.get_rank()
            self.eplb_loader.set_log2phy_map(log2phy_map)
            expert_send_info_this_rank = expert_send_info[rank_id] if rank_id in expert_send_info else []
            expert_recv_info_this_rank = expert_recv_info[rank_id] if rank_id in expert_recv_info else []
            #logger.info(f"check update info, layer = {layer_id}, send = {expert_send_info_this_rank}, recv = {expert_recv_info_this_rank}")
            self.eplb_loader.generate_expert_d2d_transfer_task(expert_send_info_this_rank,
                expert_recv_info_this_rank, updated_expert_map, layer_id + 3)
            self.weight_update_counter += 1
            if self.weight_update_counter == self.num_moe_layers:
                self.weight_update_counter = 0
                self.update_in_flight = False
                self.update_info_all = []
        # set asynchronous stream for d2d expert weight update
        self.reqs = []
        self.eplb_loader.asyn_expert_weight_transfer(self.reqs)


    def forward_end(self,dummy_run=False):
        self.adaptor.collect_topk_ids(dummy_run)
        if not self.update_in_flight:
            load_gather_iteration, update_iteration = self.get_update_iteration()
            if load_gather_iteration:
                moe_load = self.compute_and_set_moe_load()
            if update_iteration:
                self.wakeup_eplb_worker()
                self.update_in_flight = True
                self.wait_worker_iterations = 0
                self.weight_loading = False

        if self.update_in_flight:
            self.wait_worker_iterations = self.wait_worker_iterations + 1

        self.eplb_loader.update_expert_map_and_weight(self.reqs, self.redundant_enable)

    def compute_and_set_moe_load(self,dummy_run=False):
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
            logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")
        else:
            moe_load = local_load.unsqueeze(1)
            self.shared_dict["moe_load"] = moe_load.cpu()
            logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")
        return moe_load

    def warm_up_eplb(self):

        self.get_init_expert_map()
        self.adaptor.collect_topk_ids(dummy_run=False)
        self.compute_and_set_moe_load()

        src_tensor = torch.empty((1,), device=self.device)
        self_rank = dist.get_rank()

        comm_op_list = []

        for dst_rank in range(self.world_size):
            if dst_rank == self_rank:
                continue
            comm_op_list.append(
                dist.P2POp(dist.isend, src_tensor, dst_rank)
            )

        for src_rank in range(self.world_size):
            if src_rank == self_rank:
                continue
            comm_op_list.append(
                dist.P2POp(dist.irecv, src_tensor, src_rank)
        )
        if comm_op_list:
            reqs = dist.batch_isend_irecv(comm_op_list)

        for req in reqs:
            req.wait()

    def unpack_update_batch(self, packed_update_info):
        """
        Unpack the IPC batch back into original update_info_list.
        """
        send_all, recv_all, stacked_maps, stacked_log2phy, layer_id_tensor = packed_update_info

        maps     = stacked_maps.unbind(0)
        layer_ids = layer_id_tensor.tolist()

        if self.redundant_enable:
            log2phy_list = stacked_log2phy.unbind(0)
        else:
            log2phy_list = [None] * len(maps)

        _zip = zip
        _send = send_all
        _recv = recv_all
        _maps = maps
        _l2p  = log2phy_list
        _lids = layer_ids

        recovered = [
            (_s, _r, _m, _lp, _lid)
            for _s, _r, _m, _lp, _lid
            in _zip(_send, _recv, _maps, _l2p, _lids)
        ]
        return recovered

    def get_expert_load(self):
        expert_maps = self.shared_dict["expert_maps"]
        moe_load = self.shared_dict["moe_load"]  # Tensor [L, W, global_experts_num]  
        num_local_experts = expert_maps.max() + 1
        load_info, _ = ExpertMapUtils.global2local_load(moe_load, expert_maps, num_local_experts)
        
        L, W, _ = load_info.shape

        expert_load: Dict[str, List[dict]] = {}
        for c in range(W):
            layers: List[dict] = []
            for l in range(L):
                counts_1d = load_info[l, c]         
        
                layer_val = {
                    f"expert_{e}": int(v)            
                    for e, v in enumerate(counts_1d.tolist())
                }
                layers.append({f"layer_{l}": layer_val})
            expert_load[f"card_{c}"] = layers

        return {"expert_load": expert_load}

    def update_expert_load_statistical_period(self, num_expert_load_gather: int, num_iterations: int):
        logger.info(f" start update {self.num_expert_load_gather=}, {self.num_iterations}...")
        self.num_expert_load_gather = num_expert_load_gather
        self.num_iterations = num_iterations
        logger.info(f" update {self.num_expert_load_gather=}, {self.num_iterations} success...")

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.eplb_process.is_alive():
            self.eplb_process.terminate()
            self.eplb_process.join()
            logger.info("[ModelRunner] EPLB process terminated")
