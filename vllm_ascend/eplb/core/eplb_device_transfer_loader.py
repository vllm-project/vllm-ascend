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
import os
from enum import Enum

import torch.distributed as dist
from vllm.logger import logger

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

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once("current d2d weight update tasks are on-going, cannot accept new weight update task")
            return

        logger.info(
            "[EPLB-DEBUG] pid=%s layer=%s generate_d2d_task: send_info=%s recv_info=%s state=%s",
            os.getpid(), layer_id, expert_send_info, expert_recv_info, self.state,
        )

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.isend, src_tensor, self.comm_group.ranks[dst_rank], group=self.comm_group.device_group)
                )
                logger.info(
                    "[EPLB-DEBUG] pid=%s layer=%s isend: local_expert=%s global_expert=%s dst_ep_rank=%s dst_global_rank=%s",
                    os.getpid(), layer_id, local_expert_id, global_expert_id_to_send, dst_rank,
                    self.comm_group.ranks[dst_rank],
                )

        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.irecv, buffer_tensor, self.comm_group.ranks[recv_rank], group=self.comm_group.device_group)
                )
                logger.info(
                    "[EPLB-DEBUG] pid=%s layer=%s irecv: buffer_id=%s global_expert=%s src_ep_rank=%s src_global_rank=%s",
                    os.getpid(), layer_id, buffer_tensor_id, global_expert_id_to_recv, recv_rank,
                    self.comm_group.ranks[recv_rank],
                )
            local_expert_to_replace = self.updated_expert_map[global_expert_id_to_recv].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        logger.info(
            "[EPLB-DEBUG] pid=%s layer=%s generate_d2d_task DONE: total_ops=%s recv_list=%s",
            os.getpid(), layer_id, len(self.comm_op_list), self.recv_expert_list,
        )
        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):
        self.updated_log2phy_map = log2phy_map

    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be launched
        if self.state != ExpertWeightUpdateState.READY:
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s asyn_transfer SKIP: state=%s (not READY)",
                os.getpid(), self.layer_id, self.state,
            )
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s asyn_transfer: launching batch_isend_irecv with %s ops",
                os.getpid(), self.layer_id, len(self.comm_op_list),
            )
            ret_list = dist.batch_isend_irecv(self.comm_op_list)
            reqs.extend(ret_list)
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s asyn_transfer: batch_isend_irecv launched, %s reqs",
                os.getpid(), self.layer_id, len(ret_list),
            )
        else:
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s asyn_transfer: no ops to launch (empty comm_op_list)",
                os.getpid(), self.layer_id,
            )

        self.state = ExpertWeightUpdateState.TRANSFERRING

    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been launched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s update_map_weight SKIP: state=%s (not TRANSFERRING)",
                os.getpid(), self.layer_id, self.state,
            )
            return

        # Waiting for send/recv tasks finish
        logger.info(
            "[EPLB-DEBUG] pid=%s layer=%s update_map_weight: waiting for %s reqs to complete",
            os.getpid(), self.layer_id, len(reqs),
        )
        for i, req in enumerate(reqs):
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s update_map_weight: waiting req %s/%s",
                os.getpid(), self.layer_id, i, len(reqs),
            )
            req.wait()
            logger.info(
                "[EPLB-DEBUG] pid=%s layer=%s update_map_weight: req %s/%s done",
                os.getpid(), self.layer_id, i, len(reqs),
            )

        logger.info(
            "[EPLB-DEBUG] pid=%s layer=%s update_map_weight: all reqs done, updating maps and weights",
            os.getpid(), self.layer_id,
        )

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

        if self.layer_id == self.num_layers - 1:
            logger.info("[EPLB] finished update expert weight.")

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING
        logger.info(
            "[EPLB-DEBUG] pid=%s update_map_weight DONE, state reset to WAITING",
            os.getpid(),
        )
