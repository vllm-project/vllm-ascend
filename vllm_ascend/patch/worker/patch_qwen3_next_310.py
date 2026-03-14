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

from vllm.logger import logger

# Reuse the 310P fallback implementation that bypasses Triton and dynamic
# custom-op registration in Qwen3NextGatedDeltaNet.forward.
import vllm_ascend._310p.patch.patch_qwen3_next  # noqa: F401

try:
    # Qwen3.5 text trunk reuses Next linear-attention core in some branches.
    from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
    from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet

    Qwen3_5GatedDeltaNet._forward_core = Qwen3NextGatedDeltaNet._forward_core
except Exception:
    # Keep best-effort compatibility with branches where qwen3_5 module
    # is unavailable or has a different class layout.
    pass

logger.info_once("Loaded 310P Qwen3Next/Qwen3.5 fallback patch.")
