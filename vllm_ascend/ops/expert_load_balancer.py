import json
import random
from typing import Dict, List, Optional

import torch

from vllm_ascend.ascend_config import get_ascend_config


class ExpertLoadBalancer:
    """
    ExpertLoadBalancer is a singleton class responsible for managing and
    recording the mapping and load balancing of experts across multiple layers
    and devices in a distributed Mixture-of-Experts (MoE) model.
    """

    _instance = None
    """The singleton instance of ExpertLoadBalancer."""

    def __init__(self, expert_map_path: Optional[str], global_expert_num: int):
        """
        This method should only be called once, and it raises an exception if
        an instance already exists.

        Args:
            expert_map_path (str): Path to the expert map file. If None, only
            used for recording expert load.
            global_expert_num (int): Total number of global experts.
        Raises:
            Exception: If an instance of ExpertLoadBalancer already exists.
        """

        if ExpertLoadBalancer._instance is not None:
            raise Exception(
                "This class is a singleton, cannot be instantiated "
                "more than once.")

        self.global_expert_num = global_expert_num

        # If expert_map_path is not provided, we only record the expert load.
        if expert_map_path is not None:
            self.expert_map_tensor, self.layers_num, self.ranks_num = (
                self._expert_file_to_tensor(expert_map_path))
        else:
            self.expert_map_tensor = None
            # TODO: change the num layer source
            self.layers_num = 58
            self.ranks_num = None

        self._torchair_graph_enabled = \
            get_ascend_config().torchair_graph_config.enabled

        self._all_layers_logical_expert_load_record = \
            torch.zeros((self.layers_num, self.global_expert_num),
                        dtype=torch.int64,
                        device=torch.npu.current_device())
        # Always enable expert load recording if torchair graph is enabled.
        self._recording = self._torchair_graph_enabled

    @staticmethod
    def get_instance():
        if ExpertLoadBalancer._instance is None:
            raise ValueError(
                "ExpertLoadBalancer instance has not been initialized.")
        return ExpertLoadBalancer._instance

    @staticmethod
    def init_instance(expert_map_path: Optional[str], global_expert_num: int):
        """Initialize the singleton instance of ExpertLoadBalancer."""
        ExpertLoadBalancer._instance = ExpertLoadBalancer(
            expert_map_path, global_expert_num)
        return ExpertLoadBalancer._instance

    def _expert_file_to_tensor(self, expert_map_path: str):
        with open(expert_map_path, "r") as f:
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
        expert_placement_map = self.generate_expert_placement_map()
        layer_expert_map = expert_placement_map[layer_id]
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

    def accumulate_expert_distribution_record(self, layer_id: int,
                                              topk_ids: torch.Tensor):
        if not self._recording:
            return
        flattened_topk_ids = topk_ids.flatten().to(torch.int64)
        ones = torch.ones_like(flattened_topk_ids)
        self._all_layers_logical_expert_load_record[layer_id].scatter_add_(
            0, flattened_topk_ids, ones)

    def start_expert_distribution_record(self):
        """Start recording the expert distribution."""
        self._all_layers_logical_expert_load_record.zero_()
        self._recording = True

    def stop_expert_distribution_record(self):
        """Stop recording the expert distribution."""
        # If torchair graph is not enabled, we do not turn off the recording.
        self._recording = self._torchair_graph_enabled

    def export_local_expert_distribution_record(self):
        """Export the local expert distribution record and reset it."""
        local_record = self._all_layers_logical_expert_load_record.clone()
        self._all_layers_logical_expert_load_record.zero_()
        return local_record
