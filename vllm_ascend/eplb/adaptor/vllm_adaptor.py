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


class VllmEplbAdaptor:
    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        self.num_local_experts = self.model.model.layers[-1].mlp.experts.local_num_experts
        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = self.num_local_experts
        self.buffer_tensor_list: list[list[Any]] = [[] for _ in range(num_buffer_tensor)]
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = self.model.get_log2phy_map(
                self.num_dense_layers + layer_idx
            )

    def init_buffer_tensor(self, num_buffer_tensor):
        for buffer_id in range(num_buffer_tensor):
            for name in self.expert_weight_names:
                complete_name = "model.layers." + str(self.num_dense_layers) + ".mlp.experts." + name
                expert_tensor = self.param_dict[complete_name][0]
                buffer_tensor = torch.empty_like(expert_tensor)
                self.buffer_tensor_list[buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        self.param_dict = dict()
        if self.model.quant_config is not None:
            self.expert_weight_names = [
                "w13_weight_list",
                "w2_weight_list",
                "w13_weight_scale_fp32_list",
                "w2_weight_scale_list",
            ]
        else:
            self.expert_weight_names = ["w13_weight", "w2_weight"]

        for layer_idx in range(self.num_dense_layers, self.model.config.num_hidden_layers):
            self.expert_param_per_layer[layer_idx] = list()
            for name in self.expert_weight_names:
                param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
                param_value = getattr(self.model.model.layers[layer_idx].mlp.experts, name)
                self.param_dict[param_key] = param_value
            for local_expert_id in range(self.num_local_experts):
                per_expert_param = list()
                for name in self.expert_weight_names:
                    per_expert_param.append(
                        self.param_dict["model.layers." + str(layer_idx) + ".mlp.experts." + name][local_expert_id]
                    )
                self.expert_param_per_layer[layer_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        if self.rank_id == 0:
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

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace], self.buffer_tensor_list[buffer_tensor_id]
        ):
            expert_tensor.copy_(buffer_tensor)
            logger.debug(f"Expert tensor shape is :{expert_tensor.shape}")

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def get_global_expert_map(self):
        all_layer_global_expert_map = []
        for layer_id in range(self.num_moe_layers):
            map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_id] = map_cpu[self.rank_id]

        return torch.stack(all_layer_global_expert_map)
