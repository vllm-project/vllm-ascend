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

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder


class AscendAttentionMetadataBuilder310(AscendAttentionMetadataBuilder):
    """
    Metadata builder specialized for the Huawei Ascend 310P NPU.

    This class extends the base Ascend attention metadata builder to use
    the 310P-specific attention mask builder, ensuring that masks are
    generated in the correct format (FRACTAL_NZ) and logic required by
    the 310P hardware.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """
        Initializes the metadata builder and the 310P-specific mask builder.

        Args:
            kv_cache_spec (AttentionSpec): Specification for the KV cache (block size, etc.).
            layer_names (list[str]): List of layer names in the model.
            vllm_config (VllmConfig): Global vLLM configuration object.
            device (torch.device): The device (NPU) to run operations on.
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Override the mask builder with the 310P-specific version
        max_model_len = vllm_config.model_config.max_model_len
        self.attn_mask_builder: Any = AttentionMaskBuilder310(self.device, max_model_len)
