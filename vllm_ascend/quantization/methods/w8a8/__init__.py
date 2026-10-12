#
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
#
"""W8A8 quantization methods for Ascend NPU."""

from .fp8_block import AscendFp8BlockFusedMoEMethod, AscendFp8BlockLinearMethod, resolve_block_scales
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod
from .w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod, AscendW8A8MXFP8DynamicLinearMethod
from .w8a8_pdmix import AscendW8A8PDMixLinearMethod
from .w8a8_static import AscendW8A8LinearMethod
from .w8a8fp8_dynamic import AscendW8A8FP8DynamicFusedMoEMethod, AscendW8A8FP8DynamicLinearMethod

__all__ = [
    "AscendFp8BlockFusedMoEMethod",
    "AscendFp8BlockLinearMethod",
    "AscendW8A8DynamicFusedMoEMethod",
    "AscendW8A8DynamicLinearMethod",
    "AscendW8A8FP8DynamicFusedMoEMethod",
    "AscendW8A8FP8DynamicLinearMethod",
    "AscendW8A8LinearMethod",
    "AscendW8A8MXFP8DSDynamicLinearMethod",
    "AscendW8A8MXFP8DynamicLinearMethod",
    "AscendW8A8PDMixLinearMethod",
    "resolve_block_scales",
]
