import json
import random
from typing import Dict, List

import torch


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

    def generate_log2phy_expert_map(self, layer_id):
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
        num_experts = torch.ones(self.global_expert_num, 1, dtype=torch.int32)
        for log_ids, phy_ids in result_dict.items():
            log2phy_map[log_ids, :len(phy_ids)] = torch.tensor(phy_ids)
            num_experts[log_ids] = len(phy_ids)
        return log2phy_map, num_experts

    def get_rank_placement_map(self, layer_id, rank_id):
        expert_placement_map = self.generate_expert_placement_map()
        layer_expert_map = expert_placement_map[layer_id]
        rank_expert_map = layer_expert_map[rank_id].to(
            torch.npu.current_device())
        rank_local_expert_num = torch.sum(torch.ne(rank_expert_map, -1)).item()
        return rank_local_expert_num, rank_expert_map

    def get_rank_log2phy_map(self, layer_id):
        layer_log2phy_map = self.generate_log2phy_expert_map(layer_id)
        return layer_log2phy_map

    def get_global_redundant_expert_num(self):
        global_redundant_expert_num = (
            len(self.expert_map_tensor[0][0]) * self.ranks_num -
            self.global_expert_num)
        return global_redundant_expert_num
