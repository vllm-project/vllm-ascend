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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this adaptor.
import json
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.eplb.adaptor.abstract_adaptor import EplbAdaptor


class VllmEplbAdaptor(EplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_dict = dict(self.model.named_parameters())
        if self.model.config.model_type == "qwen3_moe":
            self.num_dense_layers = 0
            self.global_expert_num = self.model.config.num_experts
        else:
            self.num_dense_layers = self.model.config.first_k_dense_replace
            self.global_expert_num = self.model.config.n_routed_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers
        self.init_redundancy_expert = get_ascend_config(
        ).init_redundancy_expert

        for i in range(self.num_dense_layers,
                       self.model.config.num_hidden_layers):
            self.param_dict["model.layers." + str(i) + ".mlp.experts." + "w13_weight_list"] = \
                self.model.model.layers[i].mlp.experts.w13_weight_list
            self.param_dict["model.layers." + str(i) + ".mlp.experts." + "w2_weight_list"] = \
                self.model.model.layers[i].mlp.experts.w2_weight_list
            self.param_dict["model.layers." + str(i) + ".mlp.experts." + "w13_weight_scale_fp32_list"] = \
                self.model.model.layers[i].mlp.experts.w13_weight_scale_fp32_list
            self.param_dict["model.layers." + str(i) + ".mlp.experts." + "w2_weight_scale_list"] = \
                self.model.model.layers[i].mlp.experts.w2_weight_scale_list
        # TODO: init self.expert_weight_names depending on different model types, only deepseek v3 w8a8 and qwen3-moe is supported here
        if self.model.quant_config is not None:
            self.expert_weight_names = [
                "w13_weight_list", "w2_weight_list",
                "w13_weight_scale_fp32_list", "w13_weight_offset",
                "w2_weight_scale_list", "w2_weight_offset"
            ]
        else:
            self.expert_weight_names = ["w13_weight", "w2_weight"]

        self.expert_map_per_layer = dict(
        )  # reference to expert map on device for expert map update
        self.expert_map_per_layer_cpu = dict(
        )  # copy of expert map on CPU to avoid device synchronize frequently
        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_expert_map(self.num_dense_layers + layer_idx)

        # TODO: here we set number of buffer tensor equal to number of expert in each laryer, which can be improved
        num_buffer_tensor = torch.where(
            self.expert_map_per_layer[self.num_dense_layers] != -1)[0].numel()
        self.buffer_tensor_list: list[list[Any]] = [
            [] for _ in range(num_buffer_tensor)
        ]
        self.init_buffer_tensor(num_buffer_tensor)

        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)

        self.all_topk_ids = []

    def init_buffer_tensor(self, num_buffer_tensor):
        for buffer_id in range(num_buffer_tensor):
            for name in self.expert_weight_names:
                complete_name = "model.layers." + str(
                    self.num_dense_layers) + ".mlp.experts." + name
                if name in [
                        "w13_weight_list", "w2_weight_list",
                        "w13_weight_scale_fp32_list", "w2_weight_scale_list"
                ]:
                    expert_tensor = self.param_dict[complete_name][0]
                    expert_tensor = expert_tensor.clone()
                else:
                    expert_tensor = self.param_dict[complete_name][0].data[0]
                buffer_tensor = torch.empty_like(expert_tensor)
                self.buffer_tensor_list[buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        num_local_expert = self.param_dict["model.layers." + str(self.num_dense_layers) + \
                                           ".mlp.experts." + self.expert_weight_names[0]].data.shape[0]
        for moe_layer_id in range(self.num_moe_layers):
            layer_idx = self.num_dense_layers + moe_layer_id
            self.expert_param_per_layer[layer_idx] = list()
            for local_expert_id in range(num_local_expert):
                per_expert_param = list()
                for name in self.expert_weight_names:
                    if name in [
                            "w13_weight_list", "w2_weight_list",
                            "w13_weight_scale_fp32_list",
                            "w2_weight_scale_list"
                    ]:
                        per_expert_param.append(
                            self.param_dict["model.layers." + str(layer_idx) +
                                            ".mlp.experts." +
                                            name][local_expert_id])
                    else:
                        per_expert_param.append(
                            self.param_dict["model.layers." + str(layer_idx) +
                                            ".mlp.experts." +
                                            name][0].data[local_expert_id])
                self.expert_param_per_layer[layer_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def get_init_expert_map(self, num_moe_layers):
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        if dist.is_initialized():
            world_size = dist.get_world_size()

        gathered = torch.empty(
            (world_size, *expert_map.shape),  # [W, L, E]
            dtype=expert_map.dtype,
            device=expert_map.device)

        dist.all_gather_into_tensor(gathered, expert_map)
        all_maps = gathered.permute(1, 0, 2)
        all_expert_maps = all_maps.cpu()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_idx] = \
                all_expert_maps[layer_idx][self.rank_id]

        return all_expert_maps

    def get_init_expert_map_from_file(self, num_moe_layers, expert_map_path):

        try:
            expert_map_tensor, layers_num, ranks_num = self._expert_file_to_tensor(
                expert_map_path)
            expert_map_all = self.local2global(expert_map_tensor)
        except (TypeError, FileNotFoundError, OSError):
            expert_map_all = self.determine_expert_map_all()

        for layer_idx in range(num_moe_layers):
            if self.model.config.model_type == "qwen3_moe":
                self.expert_map_per_layer_cpu[layer_idx] = \
                    expert_map_all[layer_idx][self.rank_id]
            else:
                self.expert_map_per_layer_cpu[layer_idx + self.num_dense_layers] = \
                    expert_map_all[layer_idx][self.rank_id]
        return expert_map_all

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
        logger.error(f"failed to read expert_map_path: {expert_map_path}")

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        if self.rank_id == 0:
            num_local_experts = expert_maps.max() + 1
            expert_maps_local = self.global2local(expert_maps,
                                                  num_local_experts)

            expert_maps_list = expert_maps_local.tolist()
            record: dict[str, Any] = {
                "moe_layer_count": len(expert_maps_list),
                "layer_list": []
            }

            for layer_idx, layer_data in enumerate(expert_maps_list):
                layer_record: dict[str, Any] = {
                    "layer_id": layer_idx,
                    "device_count": len(layer_data),
                    "device_list": []
                }

                for device_idx, experts in enumerate(layer_data):
                    device_record = {
                        "device_id": device_idx,
                        "device_expert": experts
                    }
                    layer_record["device_list"].append(device_record)

                record["layer_list"].append(layer_record)

            with open(expert_map_record_path, "w") as f:
                json.dump(record, f, indent=4)

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer[layer_id].copy_(updated_expert_map)
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace,
                                buffer_tensor_id):
        for expert_tensor, buffer_tensor in zip(
                self.expert_param_per_layer[layer_id][local_expert_to_replace],
                self.buffer_tensor_list[buffer_tensor_id]):
            expert_tensor.copy_(buffer_tensor)
            logger.debug(f"Expert tensor shape is :{expert_tensor.shape}")

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def global2local(self, placement: torch.Tensor,
                     E_local: int) -> torch.Tensor:

        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local),
                              fill_value=-1,
                              dtype=torch.long,
                              device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:

        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global),
                                      fill_value=-1,
                                      dtype=torch.long,
                                      device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def determine_expert_map_all(self):
        if self.world_size == 1:
            local_ids = torch.arange(self.global_expert_num, dtype=torch.int32)
            return local_ids.view(1, 1, -1).expand(self.num_moe_layers, 1, -1)

        local_num_experts = self.global_expert_num // self.world_size

        expert_map_all = torch.full(
            (self.num_moe_layers, self.world_size, self.global_expert_num),
            -1,
            dtype=torch.int32)

        for r in range(self.world_size):
            if r < self.world_size - 1:
                start = r * local_num_experts
                end = (r + 1) * local_num_experts
                local_count = local_num_experts
            else:
                start = r * local_num_experts
                end = self.global_expert_num
                local_count = self.global_expert_num - r * local_num_experts

            if r < self.init_redundancy_expert:
                local_count += 1
                if end < self.global_expert_num:
                    end += 1
                else:
                    start -= 1

            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[:, r, start:end] = local_ids.unsqueeze(0).expand(
                self.num_moe_layers, -1)

        return expert_map_all
