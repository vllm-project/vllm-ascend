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

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder as _BaseBuilder


class AscendAttentionMetadataBuilder310P(_BaseBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.attn_mask_builder: Any = AttentionMaskBuilder(self.device)
