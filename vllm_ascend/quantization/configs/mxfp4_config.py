# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from typing import Any, cast

from vllm.model_executor.layers.quantization import register_quantization_config

from .compressed_tensors_config import AscendCompressedTensorsConfig


@register_quantization_config("mxfp4")
class AscendMxfp4Config(AscendCompressedTensorsConfig):
    """Load native packed MXFP4 checkpoints through the Ascend W4A8 backend."""

    def get_name(self) -> str:
        return "mxfp4"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendMxfp4Config":
        config_groups = config.get("config_groups")
        if not config_groups:
            raise ValueError("Ascend mxfp4 requires compressed-tensors config_groups.")

        # Check the raw metadata before the shared parser can discard an
        # activation recipe whose packing format is not an activation format.
        for group_name, group in config_groups.items():
            if group.get("input_activations") is not None:
                raise ValueError(
                    f"Ascend mxfp4 only supports checkpoints without an input activation recipe; "
                    f"found one in {group_name}."
                )

        parsed = cast("AscendMxfp4Config", super().from_config(config))
        if not parsed.target_scheme_map:
            raise ValueError("Ascend mxfp4 requires at least one quantized target.")
        for target, scheme in parsed.target_scheme_map.items():
            quant_format = scheme["format"] if scheme["format"] is not None else parsed.quant_format
            if not parsed._is_only_weight_mxfp4_packed(scheme["weights"], scheme["input_activations"], quant_format):
                raise ValueError(
                    f"Unsupported Ascend mxfp4 recipe for {target}: expected mxfp4-pack-quantized "
                    "with static symmetric FP4 group weights, group_size=32, and no input activation recipe."
                )
        return parsed
