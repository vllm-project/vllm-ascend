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
import numpy as np

from vllm_ascend.eplb.adaptor.abstract_adaptor import EplbAdaptor
from vllm.logger import logger
import random


class VllmEplbAdaptor(EplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.rank_id = torch.distributed.get_rank()
        self.param_dict = dict(self.model.named_parameters())
        self.num_dense_layers = self.model.config.first_k_dense_replace
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers
        self.global_expert_num = self.model.config.n_routed_experts

        # TODO: init self.expert_weight_names depending on different model types, only deepseek v3 w8a8 is supported here
        self.expert_weight_names = ["w13_weight", "w2_weight", "w13_weight_scale", "w13_weight_offset",
            "w2_weight_scale", "w2_weight_offset"]

        self.expert_map_per_layer = dict()     # reference to expert map on device for expert map update
        self.expert_map_per_layer_cpu = dict() # copy of expert map on CPU to avoid device synchronize frequently
        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] =\
                self.model.get_expert_map(self.num_dense_layers + layer_idx)

        self.buffer_tensor_dict = dict()
        # TODO: here we set number of buffer tensor equal to number of expert in each laryer, which can be improved
        num_buffer_tensor = torch.where(self.expert_map_per_layer[self.num_dense_layers] != -1)[0].numel()
        self.init_buffer_tensor_dict(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] =\
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)

    def init_buffer_tensor_dict(self, num_buffer_tensor):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(self.num_dense_layers) + ".mlp.experts." + name
            expert_tensor = self.param_dict[complete_name].data[0:num_buffer_tensor]
            self.buffer_tensor_dict[name] = torch.empty_like(expert_tensor)

    def get_buffer_tensor(self, buffer_tensor_id):
        return [self.buffer_tensor_dict[name][buffer_tensor_id] for name in self.expert_weight_names]

    def get_expert_tensor(self, layer_id, global_expert_id_to_send):
        local_expert_id = self.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
        return [self.param_dict["model.layers." + str(layer_id) + ".mlp.experts." + name].data[local_expert_id]
            for name in self.expert_weight_names]

    def get_rank_expert_workload(self, num_moe_layers):
        return self.model.get_all_moe_loads(num_moe_layers)

    def get_init_expert_map(self, num_moe_layers):
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        gathered = torch.empty((world_size, *expert_map.shape),  # [W, L, E]
            dtype=expert_map.dtype,
            device=expert_map.device)

        dist.all_gather_into_tensor(gathered, expert_map)
        all_maps = gathered.permute(1, 0, 2)
        all_expert_maps = all_maps.cpu()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_idx] = \
                all_expert_maps[layer_idx][self.rank_id]

        return all_expert_maps

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer[layer_id].copy_(updated_expert_map)
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(layer_id) + ".mlp.experts." + name
            expert_tensor = self.param_dict[complete_name].data[local_expert_to_replace]
            expert_tensor.copy_(self.buffer_tensor_dict[name][buffer_tensor_id])

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map[self.rank_id])
