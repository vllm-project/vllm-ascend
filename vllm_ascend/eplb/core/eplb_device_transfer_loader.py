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
import hashlib
import os
from enum import Enum

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group


def _tensor_checksum(t: torch.Tensor, tag: str = "") -> str:
    """Compute a deterministic hex checksum for a tensor. Debug only."""
    data = t.contiguous().cpu().float().numpy().tobytes()
    h = hashlib.sha256(data).hexdigest()[:16]
    logger.info("[EPLB_CHK] %s shape=%s dtype=%s device=%s hash=%s", tag, t.shape, t.dtype, t.device, h)
    return h


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

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once("current d2d weight update tasks are on-going, cannot accept new weight update task")
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []

        # DEBUG: dump expert mapping before update
        old_map = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id]
        old_experts = sorted([(i, int(old_map[i])) for i in range(len(old_map)) if old_map[i] != -1],
                             key=lambda x: x[1])
        logger.info("[EPLB_DEBUG] rank=%s layer=%s BEFORE: local_slot->global_expert mapping: %s",
            dist.get_rank(), layer_id, old_experts)
        logger.info("[EPLB_DEBUG] rank=%s layer=%s SEND plan: %s",
            dist.get_rank(), layer_id, expert_send_info)
        logger.info("[EPLB_DEBUG] rank=%s layer=%s RECV plan: %s",
            dist.get_rank(), layer_id, expert_recv_info)
        logger.info("[EPLB_DEBUG] rank=%s layer=%s NEW expert_map: %s",
            dist.get_rank(), layer_id,
            [(i, int(updated_expert_map[i])) for i in range(len(updated_expert_map)) if updated_expert_map[i] != -1])
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            # Plan now carries the local expert ID directly (not a slot index).
            # expert_param_per_layer is indexed by local expert ID.
            local_expert_id = global_expert_id_to_send
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]:
                _tensor_checksum(src_tensor, f"SEND layer={layer_id} expert={global_expert_id_to_send} local_expert={local_expert_id} dst_rank={dst_rank}")
                self.comm_op_list.append(
                    dist.P2POp(dist.isend, src_tensor.contiguous(), self.comm_group.ranks[dst_rank], group=self.comm_group.device_group)
                )

        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.irecv, buffer_tensor, self.comm_group.ranks[recv_rank], group=self.comm_group.device_group)
                )
            local_expert_to_replace = global_expert_id_to_recv  # direct expert ID
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
        for req in reqs:
            req.wait()

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # checksum received buffer before applying
        for buffer_tensor_id, recv_expert_info in enumerate(self.recv_expert_list):
            local_expert_to_replace, _ = recv_expert_info
            for buf in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                _tensor_checksum(buf, f"RECV layer={self.layer_id} buffer_id={buffer_tensor_id} local_expert={local_expert_to_replace}")

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, local_expert_to_replace, buffer_tensor_id)
            # checksum after copy_
            for updated_tensor in self.eplb_adaptor.expert_param_per_layer[self.layer_id][local_expert_to_replace]:
                _tensor_checksum(updated_tensor, f"COPY_DONE layer={self.layer_id} local_expert={local_expert_to_replace}")

        # DEBUG: dump expert mapping after update
        new_map = self.eplb_adaptor.expert_map_per_layer_cpu[self.layer_id]
        new_experts = sorted([(i, int(new_map[i])) for i in range(len(new_map)) if new_map[i] != -1],
                             key=lambda x: x[1])
        logger.info("[EPLB_DEBUG] rank=%s layer=%s AFTER:  local_slot->global_expert mapping: %s",
            dist.get_rank(), self.layer_id, new_experts)

        if self.layer_id == self.num_layers - 1:
            logger.info("[EPLB] finished update expert weight.")

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING
