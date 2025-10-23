import json
import random
from typing import Dict, List

import torch
import torch.distributed as dist


class ExpertLoadBalancer(object):

    def __init__(self, expert_map_path, global_expert_num):
        self.expert_map_path = expert_map_path
        self.global_expert_num = global_expert_num
        self.tensor_data = []
        self.expert_map_tensor, self.layers_num, self.ranks_num = (
            self._expert_file_to_tensor())
        self.expert_placement_map = self.generate_expert_placement_map()

    def _expert_file_to_tensor(self):
        with open(self.expert_map_path, "r") as f:
            data = json.load(f)
        layers_num = data["moe_layer_count"]
        gpus_num = data["layer_list"][0]["device_count"]

        for layer in data["layer_list"]:
            device_data = []
            for device in layer["device_list"]:
                device_data.append(device["device_expert"])
            self.tensor_data.append(device_data)
        expert_map_tensor = torch.tensor(self.tensor_data, dtype=torch.int32)
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

        log2phy_map = torch.full((self.ranks_num, self.global_expert_num),
                                 -1,
                                 dtype=torch.int32)
        for rank in range(self.ranks_num):
            for key in result_dict:
                indices_in_concat = result_dict[key]
                if key in rank_expert_to_global[rank]:
                    log2phy_map[rank][key] = rank_expert_to_global[rank][key]
                else:
                    chosen_index = random.choice(indices_in_concat)
                    log2phy_map[rank][key] = chosen_index
        return log2phy_map

    def get_rank_placement_map(self, layer_id, rank_id):
        layer_expert_map = self.expert_placement_map[layer_id]
        rank_expert_map = layer_expert_map[rank_id].to(
            torch.npu.current_device())
        rank_local_expert_num = torch.sum(torch.ne(rank_expert_map, -1)).item()
        return rank_local_expert_num, rank_expert_map

    def get_rank_log2phy_map(self, layer_id, rank_id):
        layer_log2phy_map = self.generate_log2phy_expert_map(layer_id)
        return layer_log2phy_map[rank_id]

    def get_global_redundant_expert_num(self):
        global_redundant_expert_num = (
            len(self.expert_map_tensor[0][0]) * self.ranks_num -
            self.global_expert_num)
        return global_redundant_expert_num

    def check_expert_map_tensor(self):
        if dist.is_initialized():
            try:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                all_expert_maps = [None for _ in range(world_size)]
                dist.all_gather_object(all_expert_maps, self.tensor_data)
                for rank_id, expert_map_tensor in enumerate(all_expert_maps):
                    if self.tensor_data != expert_map_tensor:
                        raise ValueError(
                            f"The expert map of rank{rank} is not equal to rank{rank_id}"
                        )
                return True
            except Exception as e:
                raise ValueError(
                    f"The expert maps of all ranks are inconsistency: {e}")
