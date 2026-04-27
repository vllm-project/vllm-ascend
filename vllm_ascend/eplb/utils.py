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


def get_all_moe_loads(self, policy_type=0, pd_delay=0):
    num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
    num_layers = self.model.config.num_hidden_layers
    moe_load_old = None
    if policy_type == 4:
        all_moe_loads = torch.stack(
            [self.model.layers[layer_id].mlp.experts.moe_load1 for layer_id in range(num_dense_layers, num_layers)],
            dim=0,
        )
        activate_req = self.model.layers[0].mlp.experts.token2req
        all_moe_loads_local = torch.stack(
            [self.model.layers[layer_id + num_dense_layers].mlp.experts.moe_load_local \
                for layer_id in range(self.num_moe_layers)],
            dim=0
        )
        moe_load_old = all_moe_loads_local[:, activate_req].sum(dim=1)
        all_moe_loads_local = all_moe_loads_local / (all_moe_loads_local.sum(dim=2, keepdim=True) + 1e-12)
        all_moe_loads = all_moe_loads / (all_moe_loads.sum(dim=2, keepdim=True) + 1e-12)
        all_moe_loads[:, 0].zero_()  # avoid self-load affecting load balancing
        all_moe_loads_local[:, 0].zero_()  # avoid self-load affecting load balancing
        all_moe_loads *= pd_delay
        all_moe_loads += all_moe_loads_local * (1-pd_delay)
        if activate_req is not None and activate_req.shape[0] > 0:
            return all_moe_loads[:, activate_req].sum(dim=1).contiguous(), moe_load_old
        else:
            return all_moe_loads[:, 0].contiguous(), moe_load_old
    else:
        moe_load_old = torch.stack(
            [self.model.layers[layer_id].mlp.experts.moe_load for layer_id in range(num_dense_layers, num_layers)],
            dim=0,
        )
        return None, moe_load_old

def set_all_token2req(self, token2req):
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    x = token2req.npu()
    for layer_id in range(self.num_moe_layers):
        self.model.layers[layer_id + num_dense_layers].mlp.experts.token2req = x



def clear_all_moe_loads(self):
    num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
    num_layers = self.model.config.num_hidden_layers
    for layer_id in range(num_dense_layers, num_layers):
        self.model.layers[layer_id].mlp.experts.clear_moe_load()


def model_register(model):
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.set_all_token2req = types.MethodType(set_all_token2req, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)
