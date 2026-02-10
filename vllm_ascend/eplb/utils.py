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
# Todo: Once https://github.com/vllm-project/vllm/pull/23553 is merged in vllm. Remove this model register.
import types

import torch


def get_expert_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.expert_map


def get_log2phy_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_moe_loads(self):
    num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
    num_layers = self.model.config.num_hidden_layers
    all_moe_loads = torch.stack(
        [self.model.layers[layer_id].mlp.experts.moe_load for layer_id in range(num_dense_layers, num_layers)],
        dim=0,
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
    num_layers = self.model.config.num_hidden_layers
    for layer_id in range(num_dense_layers, num_layers):
        self.model.layers[layer_id].mlp.experts.clear_moe_load()


def model_register(model):
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)
