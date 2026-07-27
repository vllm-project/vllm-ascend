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
from inspect import getattr_static
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType

EPLB_EXPERT_WEIGHT_NAMES = {
    (QuantType.NONE, False): ("w13_weight", "w2_weight"),
    (QuantType.NONE, True): ("w13_weight_list", "w2_weight_list"),
    (QuantType.W8A8, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
    ),
    (QuantType.W8A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
        "fused_w1_scale_list",
        "fused_w2_scale_list",
    ),
    (QuantType.W4A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
        "w13_scale_bias_list",
        "w2_scale_bias_list",
    ),
    (QuantType.W4A4MXFP, False): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W4A4MXFP, True): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W8A8MXFP, False): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W8A8MXFP, True): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
}


class VllmEplbAdaptor:
    _registered_moe_layers: list["torch.nn.Module"] = []

    @staticmethod
    def register_layer(layer: "torch.nn.Module") -> None:
        """Register a MoE layer for EPLB. Called during layer initialization.

        Only real layers call this; PPMissingLayer won't, so the registry
        naturally contains only layers on this PP rank.
        """
        VllmEplbAdaptor._registered_moe_layers.append(layer)

    def __init__(self, model, **args):
        super().__init__(**args)
        if hasattr(model, "language_model"):
            self.model = model.language_model
            self.config = model.config.text_config
        else:
            self.model = model
            self.config = model.config
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.config, "first_k_dense_replace", 0)

        self.moe_layers = VllmEplbAdaptor._registered_moe_layers
        self.num_moe_layers = len(self.moe_layers)

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        # Get num_local_experts from first real MoE layer
        first_layer = self.moe_layers[0]
        self.num_local_experts = first_layer.local_num_experts
        self.ep_rank = first_layer.ep_rank

        self.expert_param_per_layer = dict()
        self.expert_weight_key_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = self.num_local_experts
        self.buffer_tensor_list: dict[Any, list[list[Any]]] = dict()
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for local_idx, layer in enumerate(self.moe_layers):
            self.log2phy_map_per_layer[local_idx] = layer.get_log2phy_map()

    def init_buffer_tensor(self, num_buffer_tensor):
        buffer_tensor_shapes: dict[Any, list[torch.Size]] = dict()
        for local_idx, _ in enumerate(self.moe_layers):
            expert_weight_key = self.expert_weight_key_per_layer[local_idx]
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            expert_tensors = [self.param_dict[f"{local_idx}.{name}"][0] for name in expert_weight_names]
            expert_tensor_shapes = [tensor.shape for tensor in expert_tensors]
            if expert_weight_key in self.buffer_tensor_list:
                assert expert_tensor_shapes == buffer_tensor_shapes[expert_weight_key], (
                    f"EPLB expert weight shapes mismatch for {expert_weight_key}: "
                    f"expected {buffer_tensor_shapes[expert_weight_key]}, got {expert_tensor_shapes}"
                )
                continue
            buffer_tensor_shapes[expert_weight_key] = expert_tensor_shapes
            self.buffer_tensor_list[expert_weight_key] = [[] for _ in range(num_buffer_tensor)]
            for buffer_id in range(num_buffer_tensor):
                for expert_tensor in expert_tensors:
                    buffer_tensor = torch.empty_like(expert_tensor)
                    self.buffer_tensor_list[expert_weight_key][buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        self.param_dict = dict()

        for local_idx, layer in enumerate(self.moe_layers):
            quant_type = QuantType.NONE if self.model.quant_config is None else layer.quant_type
            expert_weight_key = (quant_type, get_ascend_config().enable_fused_mc2 == 1)
            if expert_weight_key[0] == QuantType.W4A8MXFP:
                raise RuntimeError(f"EPLB not support {quant_type}")
            if expert_weight_key not in EPLB_EXPERT_WEIGHT_NAMES:
                raise ValueError(f"EPLB not support {quant_type} with fused MC2 {expert_weight_key[1]}")
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            self.expert_weight_key_per_layer[local_idx] = expert_weight_key
            self.expert_param_per_layer[local_idx] = list()
            for name in expert_weight_names:
                param_key = f"{local_idx}.{name}"
                # Inspect the attribute without invoking __getattr__. Besides
                # avoiding side effects from dynamic proxies, this prevents an
                # unspecced MagicMock from fabricating the optional accessor.
                # The refactored MoERunner declares the accessor because its
                # RoutedExperts child owns weights; legacy layers expose them
                # directly.
                get_parameter = (
                    layer.get_eplb_parameter if getattr_static(layer, "get_eplb_parameter", None) is not None else None
                )
                self.param_dict[param_key] = get_parameter(name) if get_parameter is not None else getattr(layer, name)
            for local_expert_id in range(self.num_local_experts):
                per_expert_param = list()
                for name in expert_weight_names:
                    per_expert_param.append(self.param_dict[f"{local_idx}.{name}"][local_expert_id])
                self.expert_param_per_layer[local_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        loads = [layer.moe_load for layer in self.moe_layers]
        self.moe_load = torch.stack(loads, dim=0) if loads else torch.empty(0)
        return self.moe_load

    def clear_all_moe_loads(self):
        for layer in self.moe_layers:
            layer.clear_moe_load()

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
        expert_weight_key = self.expert_weight_key_per_layer[layer_id]
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace],
            self.buffer_tensor_list[expert_weight_key][buffer_tensor_id],
        ):
            expert_tensor.copy_(buffer_tensor)
            logger.debug("Expert tensor shape is :%s", expert_tensor.shape)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def get_global_expert_map(self):
        all_layer_global_expert_map = []
        for local_idx, layer in enumerate(self.moe_layers):
            map_cpu = layer.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[local_idx] = map_cpu[self.ep_rank]

        return torch.stack(all_layer_global_expert_map)
