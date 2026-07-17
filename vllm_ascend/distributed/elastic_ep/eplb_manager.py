import json
import os
import uuid
from typing import Any

import numpy as np
import torch
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_eplb_group

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.eplb.core.eplb_worker import EplbWorker


class ElasticEplbManager:
    def __init__(self, worker):
        self.rank_id = get_eplb_group().rank_in_group
        self.model = worker.get_model()
        model_config = self.model.config
        self.num_dense_layers = getattr(model_config, "first_k_dense_replace", 0)
        self.num_moe_layers = model_config.num_hidden_layers - self.num_dense_layers
        self.expert_maps = None

    def get_expert_maps(self):
        all_layer_global_expert_map = []
        for layer_id in range(self.num_moe_layers):
            map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
        self.expert_maps = torch.stack(all_layer_global_expert_map)
        return self.expert_maps.clone()

    def eplb(self, old_ep_size, new_ep_size):
        assert old_ep_size != new_ep_size

        expert_map = []
        for layer_id in range(self.num_moe_layers):
            expert_map.append(self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.expert_map)
        expert_map = torch.stack(expert_map).unsqueeze(1).npu()
        expert_maps = get_eplb_group().all_gather(expert_map, dim=1).cpu()
        for layer_id in range(self.num_moe_layers):
            # Scale Up
            if old_ep_size < new_ep_size:
                expert_maps_this_layer = expert_maps[layer_id]
            # Scale Down
            else:
                if self.rank_id >= old_ep_size:
                    return
                num_logical_experts = expert_maps.shape[-1]
                experts_per_npu = expert_maps.max() + 1
                num_npus = expert_maps.shape[1]
                assert experts_per_npu * num_npus >= num_logical_experts
                expert_maps_this_layer = expert_maps[layer_id][:new_ep_size]
            self.model.model.layers[
                self.num_dense_layers + layer_id
            ].mlp.experts.global_expert_map = expert_maps_this_layer.cpu()
            self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.log2phy.copy_(
                generate_log2phy_map(expert_maps_this_layer, self.rank_id)
            )


def generate_global_placement(n_expert, ep_size, n_redundant):
    if (n_expert + n_redundant) % ep_size != 0:
        raise ValueError("(n_expert + n_redundant) % ep_size must be 0")
    all_experts = np.arange(n_expert + n_redundant)
    groups = np.array_split(all_experts, ep_size)
    groups = [group % n_expert for group in groups]
    return torch.tensor(groups, dtype=torch.int32)


def export_tensor_to_file(expert_maps, expert_map_record_path: str):
    num_local_experts = expert_maps.max() + 1

    expert_maps_list = expert_maps.tolist()
    record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}

    for layer_idx, layer_data in enumerate(expert_maps_list):
        layer_record: dict[str, Any] = {
            "layer_id": layer_idx,
            "device_count": len(layer_data),
            "device_list": [],
        }

        for device_idx, experts in enumerate(layer_data):
            placement = [experts.index(i) for i in range(num_local_experts)]
            device_record = {"device_id": device_idx, "device_expert": placement}
            layer_record["device_list"].append(device_record)

        record["layer_list"].append(layer_record)

    with open(expert_map_record_path, "w") as f:
        json.dump(record, f, indent=4)


def generate_expert_maps_file():
    hf_text_config = get_current_vllm_config().model_config.hf_text_config
    num_dense_layers = getattr(hf_text_config, "first_k_dense_replace", 0)
    num_hidden_layers = getattr(hf_text_config, "num_hidden_layers", 0)
    num_moe_layers = num_hidden_layers - num_dense_layers
    num_logical_experts = getattr(hf_text_config, "n_routed_experts", None) or getattr(
        hf_text_config, "num_experts", None
    )
    assert num_logical_experts is not None and num_logical_experts > 0
    num_redundant_experts = get_ascend_config().eplb_config.num_redundant_experts
    parallel_config = get_current_vllm_config().parallel_config
    ep_size = parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
    global_placement = generate_global_placement(num_logical_experts, ep_size, num_redundant_experts).unsqueeze(0)
    global_placement = global_placement.repeat(num_moe_layers, 1, 1)
    global_expert_maps = EplbWorker.local2global(global_placement)
    file_path = os.path.join("/tmp", uuid.uuid4().hex + ".json")
    export_tensor_to_file(global_expert_maps, file_path)

    return file_path
