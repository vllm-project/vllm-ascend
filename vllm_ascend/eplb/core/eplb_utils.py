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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove eplb utils.
import json
import os.path
from collections import defaultdict

import numpy as np
import torch
from vllm.logger import logger


def expert_file_to_tensor(expert_map_path, layer_id):
    with open(expert_map_path, "r") as f:
        data = json.load(f)
    physical_count = 0
    device_data = []
    if layer_id > data["moe_layer_count"]:
        raise ValueError("Invalid EPLB Table")
    if layer_id == data["moe_layer_count"]:
        logger.warning("Init expert map of mtp/eagle when using sample.")
        return None, None
    for device in data["layer_list"][layer_id]["device_list"]:
        physical_count += len(device["device_expert"])
        device_data.append(device["device_expert"])
    global_placement = torch.tensor(device_data, dtype=torch.int32)
    return global_placement, physical_count


def generate_global_placement(n_expert, ep_size, n_redundant):
    all_experts = np.arange(n_expert)
    groups = np.array_split(all_experts, ep_size)
    for i in range(n_redundant):
        j = i % ep_size + 1
        if len(groups[-j]) == 0:
            groups[-j] = np.append(groups[-j], j)
        else:
            groups[-j] = np.append(groups[-j], (groups[-j][-1] + 1) % n_expert)
    return torch.tensor(groups, dtype=torch.int32)


def init_eplb_config(eplb_config, layer_id, moe_config):
    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    global_placement = None
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    if expert_map_path:
        if not (os.path.exists(expert_map_path)
                and os.access(expert_map_path, os.R_OK)):
            raise ValueError("Invalid EPLB path")
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(
            expert_map_path, layer_id)
        if physical_count is not None:
            n_redundant = physical_count - n_experts
            if not moe_config.supports_eplb:
                raise ValueError(
                    "Eplb supports only w8a8_dynamic quantization.")
        else:
            eplb_enable = False

    if global_placement is None:
        global_placement = generate_global_placement(n_experts, ep_size,
                                                     n_redundant)

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, n_redundant
    global_expert_map = []
    for rankid in range(ep_size):
        expert_map = torch.full((n_experts, ), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0],
                                                   dtype=torch.int32)
        global_expert_map.append(expert_map)
    log2phy = generate_log2phy_map(
        global_expert_map, moe_config.ep_rank).npu() if eplb_enable else None

    return torch.stack(global_expert_map), log2phy, n_redundant


def generate_log2phy_map(global_expert_map, ep_rank):
    log2phy_map = defaultdict(list)
    valid_count = torch.sum(global_expert_map[0] != -1)
    for rankid, map_per_rank in enumerate(global_expert_map):
        for idx, val in enumerate(map_per_rank):
            val = val.item()
            if val != -1:
                log2phy_map[idx].append(val + rankid * valid_count)

    for key in log2phy_map.keys():
        num_of_duplications = len(log2phy_map[key])
        log2phy_map[key] = log2phy_map[key][ep_rank % num_of_duplications]

    log2phy_map = torch.scatter(
        torch.zeros(len(log2phy_map.keys()), dtype=torch.int32), 0,
        torch.tensor(list(log2phy_map.keys()), dtype=torch.int64),
        torch.tensor(list(log2phy_map.values()), dtype=torch.int32))

    return log2phy_map
