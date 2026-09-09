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
from enum import Enum

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group


class ExpertWeightUpdateState(Enum):
    WAITING = 0  # waiting for updated expert_map by EplbWorker
    READY = 1  # ready for d2d expert weights updating
    TRANSFERRING = 2  # d2d finished and waiting for updating expert_map into model


class D2DExpertWeightLoader:
    def __init__(self):
        self.comm_op_list = None
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1  # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.num_layers = 0
        self.comm_group = get_dynamic_eplb_group()

    def set_adator(self, eplb_adaptor):
        self.eplb_adaptor = eplb_adaptor

    def initialize_redundant_expert_weights(self):
        """Populate per-layer redundant slots before they become routable.

        Fused expert checkpoints load the original experts as one 3D tensor,
        so upstream's per-expert mapping cannot duplicate the trailing physical
        slots. Initialize those slots from the original physical experts with
        the same P2P path used by runtime migrations.
        """
        adaptor = self.eplb_adaptor
        if not adaptor.moe_layers:
            return

        moe_config = adaptor.moe_layers[0].moe_config
        num_logical_experts = int(moe_config.num_logical_experts)
        num_physical_experts = int(moe_config.num_experts)
        num_redundant_experts = num_physical_experts - num_logical_experts
        if num_redundant_experts <= 0:
            return

        local_capacity = adaptor.num_local_experts
        if num_physical_experts != local_capacity * adaptor.ep_size:
            raise ValueError(
                "Initial redundant expert layout must be evenly distributed: "
                f"physical={num_physical_experts}, local={local_capacity}, ep={adaptor.ep_size}."
            )

        rank = adaptor.ep_rank
        step = {"send": [], "recv": []}
        local_copies = []
        for layer_id in range(adaptor.num_moe_layers):
            for redundant_id in range(num_redundant_experts):
                logical_id = redundant_id % num_logical_experts
                src_rank, source_slot = divmod(logical_id, local_capacity)
                dst_rank, target_slot = divmod(num_logical_experts + redundant_id, local_capacity)
                if src_rank == dst_rank:
                    if rank == src_rank:
                        local_copies.append((layer_id, source_slot, target_slot))
                elif rank == src_rank:
                    step["send"].append((dst_rank, layer_id, source_slot))
                elif rank == dst_rank:
                    step["recv"].append((src_rank, layer_id, target_slot))

        for layer_id, source_slot, target_slot in local_copies:
            for source_tensor, target_tensor in zip(
                adaptor.expert_param_per_layer[layer_id][source_slot],
                adaptor.expert_param_per_layer[layer_id][target_slot],
            ):
                target_tensor.copy_(source_tensor)

        requests = self.start_global_slot_transfer(step)
        for request in requests:
            request.wait()
        if rank == 0:
            logger.info(
                "[eplb/d2d_loader] Initialized %s redundant experts across %s layers",
                num_redundant_experts,
                adaptor.num_moe_layers,
            )

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once(
                "[eplb/d2d_loader] Current D2D weight update is on-going, cannot accept new update task"
            )
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]:
                self.comm_op_list.append(
                    dist.P2POp(
                        dist.isend, src_tensor, self.comm_group.ranks[dst_rank], group=self.comm_group.device_group
                    )
                )

        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            expert_weight_key = self.eplb_adaptor.expert_weight_key_per_layer[layer_id]
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[expert_weight_key][buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(
                        dist.irecv, buffer_tensor, self.comm_group.ranks[recv_rank], group=self.comm_group.device_group
                    )
                )
            local_expert_to_replace = self.updated_expert_map[global_expert_id_to_recv].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):
        self.updated_log2phy_map = log2phy_map

    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be launched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            ret_list = dist.batch_isend_irecv(self.comm_op_list)
            reqs.extend(ret_list)

        self.state = ExpertWeightUpdateState.TRANSFERRING

    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been launched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            return

        # Waiting for send/recv tasks finish
        if reqs:
            with record_function_or_nullcontext("EPLB weight D2D wait"):
                for req in reqs:
                    req.wait()

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, local_expert_to_replace, buffer_tensor_id)

        logger.debug(
            "[eplb/d2d_loader] Layer %s D2D transfer completed, updated_experts=%s",
            self.layer_id,
            len(self.recv_expert_list),
        )

        if self.layer_id == self.eplb_adaptor.num_moe_layers - 1:
            logger.info(
                "[eplb/d2d_loader] Full expert weight update cycle completed, total_layers=%s",
                self.eplb_adaptor.num_moe_layers,
            )

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING

    def apply_global_map_updates(self, updates):
        """Apply the map half of one staged policy-4 transaction."""
        for update in updates:
            layer_id = update["layer_id"]
            self.eplb_adaptor.do_update_expert_map(layer_id, torch.tensor(update["rank_map"], dtype=torch.int32))
            self.eplb_adaptor.do_update_log2phy_map(layer_id, torch.tensor(update["log2phy_map"], dtype=torch.int32))

    def start_global_slot_transfer(self, step):
        """Start D2D into slots deactivated before this model iteration."""
        comm_ops = []
        for dst_rank, relative_layer, source_slot in step["send"]:
            layer_id = relative_layer
            for source_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][source_slot]:
                comm_ops.append(
                    dist.P2POp(
                        dist.isend,
                        source_tensor,
                        self.comm_group.ranks[dst_rank],
                        group=self.comm_group.device_group,
                    )
                )
        for src_rank, relative_layer, target_slot in step["recv"]:
            layer_id = relative_layer
            for target_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][target_slot]:
                comm_ops.append(
                    dist.P2POp(
                        dist.irecv,
                        target_tensor,
                        self.comm_group.ranks[src_rank],
                        group=self.comm_group.device_group,
                    )
                )
        return dist.batch_isend_irecv(comm_ops) if comm_ops else []

    def finish_global_slot_transfer(self, requests, step):
        """Wait for staged weights, then expose them to token routing."""
        for request in requests:
            request.wait()
        self.apply_global_map_updates(step["activate"])
