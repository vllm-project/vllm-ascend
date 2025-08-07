import json
import random
from typing import Dict, List

import torch
from vllm_ascend.distributed.parallel_state import get_ep_group


class ExpertLoadBalancer(object):

    def __init__(self, expert_map_path, global_expert_num):
        self.expert_map_path = expert_map_path
        self.global_expert_num = global_expert_num
        self.expert_map_tensor, self.layers_num, self.ranks_num = (
            self._expert_file_to_tensor())

    def _expert_file_to_tensor(self):
        with open(self.expert_map_path, "r") as f:
            data = json.load(f)
        layers_num = data["moe_layer_count"]
        gpus_num = data["layer_list"][0]["device_count"]

        tensor_data = []
        for layer in data["layer_list"]:
            device_data = []
            for device in layer["device_list"]:
                device_data.append(device["device_expert"])
            tensor_data.append(device_data)
        expert_map_tensor = torch.tensor(tensor_data, dtype=torch.int32)
        return expert_map_tensor, layers_num, gpus_num

    def generate_index_dicts(self, tensor_2d):
        dict_list = []
        current_idx = 0

        for row in tensor_2d:
            value_to_index = {}
            for i in range(row.size(0)):
                value = row[i].item()
                value_to_index[value] = current_idx + i
            dict_list.append(value_to_index)
            current_idx += row.size(0)

        return dict_list

    def generate_expert_placement_map(self):
        expert_placement_map = torch.full(
            (self.layers_num, self.ranks_num, self.global_expert_num),
            -1,
            dtype=torch.int32,
        )
        for layer_id in range(self.layers_num):
            for gpu_id in range(self.ranks_num):
                e_ids = self.expert_map_tensor[layer_id, gpu_id]
                expert_placement_map[layer_id, gpu_id,
                                     e_ids] = torch.arange(len(e_ids),
                                                           dtype=torch.int32)
        return expert_placement_map

    def generate_log2phy_expert_map(self, layer_id, rank_id):
        concatenated = torch.flatten(self.expert_map_tensor[layer_id])
        rank_expert_to_global = self.generate_index_dicts(
            self.expert_map_tensor[layer_id])
        result_dict: Dict[int, List[int]] = {}
        for idx, value in enumerate(concatenated):
            key = value.item()
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(idx)

        max_num_experts = max(len(locs) for locs in result_dict.values())
        log2phy_map = torch.full((self.global_expert_num, max_num_experts),
                                 0,
                                 dtype=torch.int32)
        num_experts = torch.ones(self.global_expert_num, dtype=torch.int32)
        # self.update_expert_map(result_dict, log2phy_map, max_num_experts, rank_id)
        self.update_expert_loc_map_v1(result_dict, rank_id)
        for log_ids, phy_ids in result_dict.items():
            log2phy_map[log_ids, :len(phy_ids)] = torch.tensor(phy_ids)
            num_experts[log_ids] = len(phy_ids)
        return log2phy_map, num_experts

    def update_expert_map(self, expert_loc, log2phy_map, max_num_dups, rank_id):
        ep_size = get_ep_group().world_size
        redundancy_shared_expert_num = self.get_global_redundant_expert_num()
        n_total_experts = self.global_expert_num + redundancy_shared_expert_num

        for i in range(self.global_expert_num):
            same_rank_candidates = []
            same_node_candidates = []
            experts_per_device = n_total_experts // ep_size
            all_candidates = []
            phy_list = expert_loc[i]
            current_device = rank_id
            for phy in phy_list:
                phy_device = phy // experts_per_device
                if phy_device == current_device:
                    same_rank_candidates.append(phy)
                elif (phy_device // self.ranks_num) == (current_device // self.ranks_num):
                    same_node_candidates.append(phy)
                else:
                    all_candidates.append(phy)
            tmp_expert_loc_map = torch.zeros([max_num_dups], dtype=torch.int32)

            if same_rank_candidates:
                expert_loc[i] = same_rank_candidates
            elif same_node_candidates:
                expert_loc[i] = same_node_candidates
            tmp_expert_loc_map[: len(expert_loc[i])] = torch.tensor(expert_loc[i], dtype=torch.int32)

            log2phy_map[i] = tmp_expert_loc_map

    def update_expert_loc_map_v1(self, expert_loc, current_rank):

        device_per_host = 16
        ep_size = get_ep_group().world_size
        current_node, current_rank_in_node = current_rank // device_per_host, current_rank % device_per_host
        redundancy_shared_expert_num = self.get_global_redundant_expert_num()
        n_total_experts = self.global_expert_num + redundancy_shared_expert_num
        experts_per_device = n_total_experts // ep_size
        num_hosts = self.ranks_num // device_per_host
        for i in range(self.global_expert_num):
            same_rank_candidates, same_node_candidates, all_candidates = [], [], []
            phy_list, num_replicas = expert_loc[i], len(expert_loc[i])

            for phy in phy_list:
                phy_device = phy // experts_per_device
                if phy_device == current_rank:
                    same_rank_candidates.append(phy)
                elif (phy_device // device_per_host) == (current_rank // device_per_host):
                    same_node_candidates.append(phy)
                else:
                    all_candidates.append(phy)

            is_imbalanced = False
            if num_replicas > num_hosts and num_replicas % num_hosts != 0:
                replica_per_node = {}
                for phy in phy_list:
                    phy_device = phy // experts_per_device
                    phy_node = phy_device // device_per_host
                    local_rank = phy_device % device_per_host
                    if phy_node not in replica_per_node:
                        replica_per_node[phy_node] = []
                    replica_per_node[phy_node].append(local_rank)
                base_replicas_per_host = num_replicas // num_hosts
                if len(replica_per_node[current_node]) == base_replicas_per_host:
                    available_ranks = list(set(range(device_per_host)) - set(replica_per_node[current_node]))
                    expected_load = round(device_per_host / (base_replicas_per_host + 1))
                    if current_rank_in_node in available_ranks:
                        if available_ranks.index(current_rank_in_node) >= (expected_load - 1) * base_replicas_per_host:
                            is_imbalanced = True

            if same_rank_candidates:
                expert_loc[i] = same_rank_candidates
            elif same_node_candidates and not is_imbalanced:
                expert_loc[i] = same_node_candidates

        return expert_loc


    def get_rank_placement_map(self, layer_id, rank_id):
        expert_placement_map = self.generate_expert_placement_map()
        layer_expert_map = expert_placement_map[layer_id]
        rank_expert_map = layer_expert_map[rank_id].to(
            torch.npu.current_device())
        rank_local_expert_num = torch.sum(torch.ne(rank_expert_map, -1)).item()
        return rank_local_expert_num, rank_expert_map

    def get_rank_log2phy_map(self, layer_id, rank_id):
        layer_log2phy_map = self.generate_log2phy_expert_map(layer_id, rank_id)
        return layer_log2phy_map

    def get_global_redundant_expert_num(self):
        global_redundant_expert_num = (
            len(self.expert_map_tensor[0][0]) * self.ranks_num -
            self.global_expert_num)
        return global_redundant_expert_num
