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
    return self.model.layers[layer_id].mlp.experts.get_map()


def get_log2phy_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_expert_map(self, num_moe_layers):
    all_loads = []
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    for layer_id in range(num_moe_layers):
        load_tensor = self.get_expert_map(
            layer_id + num_dense_layers)  # (num_experts_per_layer,)
        all_loads.append(load_tensor)

    return torch.stack(all_loads, dim=0)


def get_all_moe_loads(self):
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    all_moe_loads = torch.stack(
        [self.model.layers[layer_id + num_dense_layers].mlp.experts.moe_load \
            for layer_id in range(self.num_moe_layers)],
        dim=0
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    for layer_id in range(self.num_moe_layers):
        self.model.layers[layer_id +
                          num_dense_layers].mlp.experts.clear_moe_load()


def model_register(model, model_config):
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_expert_map = types.MethodType(get_all_expert_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)

    config = model_config.hf_config

    if config.model_type == "qwen3_moe":
        model.num_moe_layers = config.num_hidden_layers
    elif config.model_type == "deepseek_v2" or config.model_type == "deepseek_v3":
        num_dense_layers = config.first_k_dense_replace
        model.num_moe_layers = config.num_hidden_layers - num_dense_layers
    else:
        raise NotImplementedError("EPLB is not supported.")
