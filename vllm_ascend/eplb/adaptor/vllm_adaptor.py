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

from vllm_ascend.eplb.adaptor.abstract_adaptor import EplbAdaptor
from vllm.logger import logger

class VllmEplbAdaptor(EplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.param_dict = dict(self.model.named_parameters())

        # TODO: init self.expert_weight_names depending on different model types, only deepseek v3 w8a8 is supported here
        self.expert_weight_names = ["w13_weight", "w2_weight", "w13_weight_scale", "w13_weight_offset",
            "w2_weight_scale", "w2_weight_offset"]

        self.buffer_tensor_dict = dict()
        num_buffer_tensor = 100 # TO DO: provide number of buffer tensor by vllm configuration
        params_dtype = torch.int8 # TO DO: provide number of buffer tensor by vllm configuration
        self.init_buffer_tensor_dict(num_buffer_tensor, params_dtype)

        self.expert_map_per_layer = dict()
        num_moe_layers = 2 # TO DO: provide number of num_moe_layers by vllm configuration
        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer[3 + layer_idx] = self.model.get_expert_map(3 + layer_idx)

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
            local_expert_id = self.expert_map_per_layer[layer_id][global_expert_id_to_send].item()
            yield self.param_dict[complete_name].data[local_expert_id]

    def get_rank_expert_workload(self, num_moe_layers):
        return self.model.get_all_moe_loads(num_moe_layers)

    def get_init_expert_map(self, num_moe_layers):
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        if dist.is_initialized():
            world_size = dist.get_world_size()

        rank = dist.get_rank()

        tensor_list = [
            torch.zeros_like(expert_map) for _ in range(world_size)
        ]

        dist.all_gather(tensor_list, expert_map)
        gathered = torch.stack(tensor_list, dim=0)
        all_maps = gathered.permute(1, 0, 2).contiguous()

        all_expert_maps = all_maps.to(torch.device("cpu"))
        return all_expert_maps

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, expert_id_before_replace, buffer_tensor_id):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(layer_id) + ".mlp.experts." + name
            local_expert_id = self.expert_map_per_layer[layer_id][expert_id_before_replace].item()
            expert_tensor = self.param_dict[complete_name].data[local_expert_id]
            expert_tensor.copy_(self.buffer_tensor_dict[name][buffer_tensor_id])