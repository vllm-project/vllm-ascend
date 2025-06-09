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
from enum import Enum

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm_ascend.eplb.core.loader.abstract_loader import ExpertWeightLoader

class ExpertWeightUpdateState(Enum):
    WAITING = 0      # waiting for updated expert_map by EplbWorker
    READY = 1        # ready for d2d expert weights updating
    TRANSFERING = 2  # d2d finished and waiting for updating expert_map into model

class D2DExpertWeightLoader(ExpertWeightLoader):

    def __init__(self, model):
        self.comm_op_list = None
        self.model = model
        self.param_dict = dict(self.model.named_parameters())
        # TODO: init self.expert_weight_names depending on different model types, only deepseek v3 w8a8 is supported here
        config = self.model.config
        self.expert_weight_names = ["w13_weight", "w2_weight", "w13_weight_scale", "w13_weight_offset",
            "w2_weight_scale", "w2_weight_offset"]

        self.buffer_tensor_dict = dict()
        num_buffer_tensor = 1 # TO DO: provide number of buffer tensor by vllm configuration
        params_dtype = torch.int8 # TO DO: provide number of buffer tensor by vllm configuration
        self.init_buffer_tensor_dict(num_buffer_tensor, params_dtype)

        self.expert_map = dict()
        num_moe_layers = 2 # TO DO: provide number of num_moe_layers by vllm configuration
        for layer_idx in range(num_moe_layers):
            self.expert_map[3 + layer_idx] = self.model.get_expert_map(3 + layer_idx)

        self.updated_expert_map = None
        self.layer_id = -1

        self.state = ExpertWeightUpdateState.WAITING
        self.pull_tensor_list = []
        self.mock_flag = True

    def generate_expert_d2d_send_task(self, expert_send_info, expert_recv_info,
        updated_expert_map, layer_id):
        if self.state != ExpertWeightUpdateState.WAITING:
            return -1

        if not (expert_send_info or expert_recv_info):
            return -1

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            for src_tensor in self.get_expert_tensor(layer_id, global_expert_id_to_send):
                self.comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

        buffer_tensor_id = 0
        for pull_info in expert_recv_info:
            pull_rank, global_expert_id_to_pull = pull_info
            for buffer_tensor in self.get_buffer_tensor(buffer_tensor_id):
                self.comm_op_list.append(dist.P2POp(dist.irecv, buffer_tensor, pull_rank))
            self.pull_tensor_list.append((self.updated_expert_map[global_expert_id_to_pull].item(), buffer_tensor_id))
        buffer_tensor_id += 1

        self.state = ExpertWeightUpdateState.READY

        return 0

    def asyn_expert_weight_send(self, reqs):
        if self.state != ExpertWeightUpdateState.READY:
            return -1
        if self.comm_op_list is not None:
            reqs = dist.batch_isend_irecv(self.comm_op_list)
        self.state = ExpertWeightUpdateState.TRANSFERING
        return 0

    def update_expert_map_and_weight(self, reqs):
        if self.state != ExpertWeightUpdateState.TRANSFERING:
            return -1
        for req in reqs:
            req.wait()
        if self.comm_op_list is not None:
            self.comm_op_list = None
        self.expert_map[self.layer_id].copy_(self.updated_expert_map)
        for pull_info in self.pull_tensor_list:
            local_expert_id, buffer_tensor_id = pull_info
            self.copy_buffer_tensor(self.layer_id, local_expert_id, buffer_tensor_id)
        self.pull_tensor_list = []
        self.state = ExpertWeightUpdateState.WAITING
        return 0

    def init_buffer_tensor_dict(self, num_buffer_tensor, params_dtype):
        for name in self.expert_weight_names:
            complete_name = "model.layers.3.mlp.experts." + name
            expert_tensor = self.param_dict[complete_name].data[0:num_buffer_tensor]
            self.buffer_tensor_dict[name] = torch.empty_like(expert_tensor)

    def get_buffer_tensor(self, buffer_tensor_id):
        for name in self.expert_weight_names:
            yield self.buffer_tensor_dict[name][buffer_tensor_id]

    def get_expert_tensor(self, layer_id, global_expert_id_to_send):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(layer_id) + ".mlp.experts." + name
            local_expert_id = self.expert_map[layer_id][global_expert_id_to_send].item()
            yield self.param_dict[complete_name].data[local_expert_id]

    def copy_buffer_tensor(self, layer_id, expert_id_before_replace, buffer_tensor_id):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(layer_id) + ".mlp.experts." + name
            local_expert_id = self.expert_map[layer_id][expert_id_before_replace].item()
            expert_tensor = self.param_dict[complete_name].data[local_expert_id]
            expert_tensor.copy_(self.buffer_tensor_dict[name][buffer_tensor_id])

    def generate_mock_update_info(self, rank_id):
        if rank_id == 0:
            expert_send_info = [(1, 0)]
            expert_recv_info = [(1, 64)]
            updated_expert_map_list = [-1] + [i for i in range(1, 64)] + [0] + [j for j in [-1] * 191]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 1:
            expert_send_info = [(0, 64)]
            expert_recv_info = [(0, 0)]
            updated_expert_map_list = [0] + [k for k in [-1] * 63] + [i for i in range(1, 64)] + [j for j in [-1] * 129]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 2:
            expert_send_info = [(3, 128)]
            expert_recv_info = [(3, 192)]
            updated_expert_map_list = [k for k in [-1] * 129] + [i for i in range(1, 64)] + [0] + [j for j in [-1] * 63]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 3:
            expert_send_info = [(2, 192)]
            expert_recv_info = [(2, 128)]
            updated_expert_map_list = [k for k in [-1] * 128] + [0] + [k for k in [-1] * 64] + [i for i in range(1, 64)]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        self.mock_flag = False
        return (expert_send_info, expert_recv_info, updated_expert_map, layer_id)



    def load_impl(self, old_expert_table, new_expert_table):
        raise NotImplementedError