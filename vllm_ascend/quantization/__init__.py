#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Ascend quantization module.

This module provides quantization support for Ascend NPU.

Supported quantization tools:
- ModelSlim: Use AscendModelSlimConfig
- LLM-Compressor (compressed_tensors): Use AscendCompressedTensorsConfig

Public API:
- Config classes: AscendModelSlimConfig, AscendCompressedTensorsConfig
- For scheme implementations, import from vllm_ascend.quantization.methods
"""

# LLM-Compressor (compressed_tensors) quantization config
from .compressed_tensors_config import AscendCompressedTensorsConfig

# ModelSlim quantization config
from .modelslim_config import AscendModelSlimConfig

__all__ = [
    "AscendModelSlimConfig",
    "AscendCompressedTensorsConfig",
]
