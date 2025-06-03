#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import torch
import torch_npu
import torch.distributed as dist

class EPLBD2DUpdator:
    def __init__(self, model, intermediate_size_per_partition, hidden_sizes, num_buffer_tensor, expert_map):
        self.comm_op_list = None
        self.model = model
        self.param_dict = dict(self.model.name_parameters())
        self.expert_params_name = {
            "w13_weight":(2 * intermediate_size_per_partition, hidden_sizes),
            "w2_weight":(hidden_sizes, intermediate_size_per_partition),
            "w13_weight_scale":(2 * intermediate_size_per_partition, 1),
            "w13_weight_offset":(2 * intermediate_size_per_partition, 1),
            "w2_weight_scale":(hidden_sizes, 1),
            "w2_weight_offset":(hidden_sizes, 1)
        }
        self.init_buffer_tensor_dict(num_buffer_tensor, params_dtype=torch.int8)
        self.expert_map = expert_map
        self.updated_expert_map = None

    def update_expert_weights_update_info(self, expert_transfer_info, expert_pull_info,
        updated_expert_map, layer_id):
        if self.comm_op_list is not None:
            return -1

        self.updated_expert_map = updated_expert_map

        self.comm_op_list = []
        for transfer_info in expert_transfer_info:
            dst_rank = transfer_info[0]
            global_expert_id_to transfer = transfer_info[1]
            for src_tensor in self.get_expert_tensor(layer_id, global_expert_id_to_transfer):
                self.comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

        buffer_tensor_id = 0
        for pull_rank in expert_pull_info:
            for buffer_tensor in self.get_buffer_tensor(buffer_tensor_id):
                self.comm_op_list.append(dist.P2POp(dist.irecv, buffer_tensor, pull_rank))
        buffer_tensor_id += 1

        return 0

    def asyn_expert_weight_transfer(self):
        reqs = []
        if self.comm_op_list is not None:
            reqs = dist.batch_isend_irecv(self.comm_op_list)
        return reqs

    def update_expert_map(self, reqs):
        for req in reqs:
            req.wait()
        if self.comm_op_list is not None:
            self.comm_op_list = None
        self.expert_map[layer_id] = self.updated_expert_map

    def init_buffer_tensor_dict(self, num_buffer_tensor, params_dtype):
        for name,dim in self.expert_params_name.items():
            num_row, num_col = *dim
            self.buffer_tensor_dict[name] = torch.empty(
                num_buffer_tensor, num_row, num_col, dtype=params_dtype
            )

    def get_buffer_tensor(self, buffer_tensor_id):
        for name in self.expert_params_name.keys():
            yield self.buffer_tensor_dict[name][buffer_tensor_id]

    def get_expert_tensor(self, layer_id, global_expert_id_to_transfer):
        for name in self.expert_params_name.keys():
            complete_name = "model.layers." + str(layer_id) + "mlp.experts." + name
            local_expert_id = self.expert_map[global_expert_id_to_transfer]
            yield self.param_dict[complete_name].data[local_expert_id]

    def copy_buffer_tensor(self, layer_id, expert_id_before_replace, buffer_tensor_id):
        for name in self.expert_params_name.keys():
            complete_name = "model.layers." + str(layer_id) + "mlp.experts." + name
            local_expert_id = self.expert_map[expert_id_before_replace]
            expert_tensor = self.param_dict[complete_name].data[local_expert_id]
            expert_tensor.copy_(self.buffer_tensor_dict[name][buffer_tensor_id])