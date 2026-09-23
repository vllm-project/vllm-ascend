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
from collections import defaultdict

import numpy as np
import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

# Linear EP placement is identical across MoE layers for a given
# (ep_size, ep_rank, n_experts). Reuse it so construct does not rebuild
# the map 60+ times.
_LINEAR_EXPERT_MAP_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def expert_file_to_tensor(expert_map_path, layer_id):
    with open(expert_map_path) as f:
        data = json.load(f)
    physical_count = 0
    device_data = []
    if layer_id > data["moe_layer_count"]:
        raise ValueError("Invalid EPLB Table")
    if layer_id == data["moe_layer_count"]:
        logger.warning("[eplb/utils] Init expert map of mtp/eagle when using sample.")
        for device in data["layer_list"][0]["device_list"]:
            physical_count += len(device["device_expert"])
        return None, physical_count
    for device in data["layer_list"][layer_id]["device_list"]:
        physical_count += len(device["device_expert"])
        device_data.append(device["device_expert"])
    global_placement = torch.tensor(device_data, dtype=torch.int32)
    return global_placement, physical_count


def generate_global_placement(n_expert, ep_size, n_redundant, num_shared_experts):
    n_expert -= num_shared_experts
    if (n_expert + n_redundant) % ep_size != 0:
        raise ValueError("(n_expert + n_redundant) % ep_size must be 0")
    if num_shared_experts == 0:
        # Match vLLM's checkpoint-loading physical layout exactly:
        # [all logical experts, redundant copies of logical experts 0..N].
        # The flattened position is the global physical expert ID and must
        # agree with RoutedExperts.make_expert_params_mapping.
        physical_to_logical = np.concatenate((np.arange(n_expert), np.arange(n_redundant) % n_expert))
        return torch.tensor(physical_to_logical.reshape(ep_size, -1), dtype=torch.int32)

    # Shared-expert mix placement has a separate Ascend-only layout.
    all_experts = np.arange(n_expert)
    groups = np.array_split(all_experts, ep_size)
    for i in range(n_redundant):
        j = i % ep_size + 1
        if len(groups[-j]) == 0:
            groups[-j] = np.append(groups[-j], j)
        else:
            groups[-j] = np.append(groups[-j], (groups[-j][-1] + 1) % n_expert)
    if num_shared_experts > 0:
        for i, group in enumerate(groups):
            groups[i] = np.append(group, n_expert + i % num_shared_experts)
    return torch.tensor(groups, dtype=torch.int32)


def init_eplb_config(eplb_config, layer_id, moe_config, mix_placement=False, num_shared_experts=1, tp_size=None):
    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    global_placement = None
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    num_shared_experts = num_shared_experts if mix_placement else 0

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, None, n_redundant, None

    if expert_map_path:
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        n_redundant = physical_count - n_experts
    elif not eplb_enable:
        cache_key = (ep_size, moe_config.ep_rank, n_experts)
        expert_map = _LINEAR_EXPERT_MAP_CACHE.get(cache_key)
        if expert_map is None:
            _, expert_map, _ = determine_expert_map(ep_size, moe_config.ep_rank, n_experts)
            _LINEAR_EXPERT_MAP_CACHE[cache_key] = expert_map
        if expert_map is None:
            return None, None, None, 0, None
        expert_map = expert_map.clone()
        try:
            if torch.npu.is_available():
                expert_map = expert_map.to(device=f"npu:{torch.npu.current_device()}")
        except Exception:
            pass
        return None, expert_map, None, 0, None

    if global_placement is None:
        global_placement = generate_global_placement(n_experts, ep_size, n_redundant, num_shared_experts)
        if mix_placement:
            n_redundant += ep_size - 1
    # Flattened placement lists the logical ID at every physical position, so
    # it is exactly the physical -> logical expert ID mapping for the layout.
    phys_to_logical = global_placement.reshape(-1).contiguous()
    global_expert_map = []
    phys_offset = 0
    for rankid in range(ep_size):
        # Full-length map over physical expert IDs: entry p is the local slot
        # of physical expert p on this rank, or -1 when not owned here.
        # Physical IDs reach num_experts + n_redundant, so the map must
        # cover that whole range for dispatcher masking (issue #14080).
        expert_map = torch.full((phys_to_logical.numel(),), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        local_slots = torch.arange(local_placement.shape[0], dtype=torch.int32)
        expert_map[phys_offset + local_slots] = local_slots
        phys_offset += local_placement.shape[0]
        global_expert_map.append(expert_map)
        if rankid == moe_config.ep_rank:
            local_expert_map = expert_map
    log2phy = (
        generate_log2phy_map(
            global_expert_map,
            moe_config.ep_rank,
            tp_size=int(tp_size) if tp_size is not None else None,
            phys_to_logical=phys_to_logical,
        ).npu()
        if eplb_enable
        else None
    )

    return torch.stack(global_expert_map), local_expert_map, log2phy, n_redundant, phys_to_logical


def generate_log2phy_map(
    global_expert_map,
    ep_rank,
    tp_size: int | None = None,
    phys_to_logical: torch.Tensor | None = None,
):
    """Build the logical -> global physical expert ID map for ``ep_rank``.

    Without ``phys_to_logical`` the map rows already index logical experts
    (legacy layout). With it, the rows are full-length physical maps
    (entry p = local slot of physical expert p) and the physical copies of
    each logical expert are grouped through ``phys_to_logical``. The output
    stays logical-length so ``log2phy[logical_topk_ids]`` keeps working.
    """
    log2phy_map = defaultdict(list)
    valid_count = torch.sum(global_expert_map[0] != -1)
    if phys_to_logical is None:
        for rankid, map_per_rank in enumerate(global_expert_map):
            for idx, val in enumerate(map_per_rank):
                val = val.item()
                if val != -1:
                    log2phy_map[idx].append(val + rankid * valid_count)
    else:
        for rankid, map_per_rank in enumerate(global_expert_map):
            for phys_id, val in enumerate(map_per_rank):
                val = val.item()
                if val != -1:
                    log2phy_map[int(phys_to_logical[phys_id].item())].append(val + rankid * valid_count)

    for key in log2phy_map:
        num_of_duplications = len(log2phy_map[key])
        if tp_size is not None and tp_size > 1:
            tp_rank = ep_rank % tp_size
            dp_like_rank = ep_rank // tp_size
            replica_index = (tp_rank + dp_like_rank + key) % num_of_duplications
        else:
            replica_index = ep_rank % num_of_duplications
        log2phy_map[key] = log2phy_map[key][replica_index]

    log2phy_map = torch.scatter(
        torch.zeros(len(log2phy_map), dtype=torch.int32),
        0,
        torch.tensor(list(log2phy_map), dtype=torch.int64),
        torch.tensor(list(log2phy_map.values()), dtype=torch.int32),
    )

    return log2phy_map
