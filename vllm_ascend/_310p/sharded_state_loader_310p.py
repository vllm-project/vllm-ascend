#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import json
import os
from pathlib import Path

import torch
import torch_npu
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import ShardedStateLoader

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


class ShardedStateLoader310(ShardedStateLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if torch_npu.get_npu_format(tensor) == ACL_FORMAT_FRACTAL_NZ:
                tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
            state_dict_part[key] = tensor

        filename = ShardedStateLoader.DEFAULT_PATTERN.format(rank=rank, part=part_idx)
        save_file(
            state_dict_part,
            os.path.join(path, filename),
        )

    @staticmethod
    def generate_quant_description(model: torch.nn.Module, path: str):
        """Generate a mapping of parameter names to their corresponding quantization types."""
        quant_description = {}
        quantize_type = model.quant_config.quant_description.get("model_quant_type", "FLOAT")
        quant_description["model_quant_type"] = quantize_type
        quant_description["version"] = "1.0.0"
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                if tensor.dtype in [torch.int8, torch.int32, torch.int64]:
                    quant_description[name] = quantize_type
                else:
                    quant_description[name] = "FLOAT"
            else:
                quant_description[name] = "FLOAT"

        json_path = Path(path) / "parameters_type_map.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(quant_description, f, indent=2)
