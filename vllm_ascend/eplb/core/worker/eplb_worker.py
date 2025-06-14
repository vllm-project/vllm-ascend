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

import time
import numpy as np
import torch
import torch_npu
import logging
import torch.distributed as dist
from multiprocessing import Process, Queue, Manager
from abc import ABC, abstractmethod
from vllm.logger import logger

from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory, DynamicConfig


class EplbWorker:

    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type, DynamicConfig())
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D

        # Get initial expert_map
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            self.num_local_experts = self.old_expert_maps.max() + 1

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        #根据负载信息，获取更新后的专家表
        load_info, old_placement = self.global2local(load_info, self.old_expert_maps, self.num_local_experts)
        changed, priority, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)

        logger.debug(f"[EPLB Process  new_map differs, performing D2D")

        update_info = self.compose_expert_update_info(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.info("EPLB Process complete")

        return update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(f"Replicated experts are placed on the same NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid")
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement[expert_not_move], old_placement[expert_not_move]):
                    logger.error(f"There exists expert movement inside NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid")
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        num_ranks = current_expert_maps.shape[1]
        num_experts = current_expert_maps.shape[2]

        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer = dict()
            expert_recv_info_this_layer = dict()

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (expert_send_info_this_layer, expert_recv_info_this_layer, updated_expert_maps_this_layer, layer_id)

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where((current_expert_maps_this_layer == -1) \
                & (updated_expert_maps_this_layer != -1))

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where((current_expert_maps_this_layer != -1) \
                & (updated_expert_maps_this_layer == -1))

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                #TODO: improve selection criterion of npu sending expert_id considering such as intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (expert_send_info_this_layer, expert_recv_info_this_layer, updated_expert_maps_this_layer, layer_id)


    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        通过 policy 实例的 rebalance_experts 方法计算 new_map。
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):

        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self,
        workload: torch.Tensor,
        placement: torch.Tensor,
        E_local: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        L, G, _ = placement.shape
        device = placement.device

        wt_local = torch.full((L, G, E_local),
                              fill_value=-1,
                              dtype=workload.dtype,
                              device=device)
        pt_local = torch.full((L, G, E_local),
                              fill_value=-1,
                              dtype=torch.long,
                              device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]
        values = workload[l_idx, g_idx, k_idx]

        wt_local[l_idx, g_idx, slot_idx] = values
        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return wt_local, pt_local


    def local2global(self,
        placement_local: torch.Tensor
    ) -> torch.Tensor:

        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global),
                                      fill_value=-1,
                                      dtype=torch.long,
                                      device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global


class EplbProcess:
    def __init__(self, shared_dict, planner_q, block_update_q, policy_type: int = 0, enable_d2d: bool = True):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q = planner_q
        self.block_update_q = block_update_q

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type, self.enable_d2d)


    def worker_process(self,planner_q,block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up, call do_update, then notify main process update is complete.
        """
        while True:
            try:

                planner_q.get()

                update_info = self.worker.do_update()

                logger.info("update_info:", update_info)

                for (a,b,c,d) in update_info:
                    while True:
                        if not block_update_q.empty():
                            continue
                        block_update_q.put((a,b,c,d))
                        break

            except Exception as e:
                logger.warning(f"[EPLB subprocess Exiting due to error: {e}", exc_info=True)
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(
            target=self.worker_process,
            args=(self.planner_q,self.block_update_q),
            daemon=True
        )

        proc.start()
        return  proc

