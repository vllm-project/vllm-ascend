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

from vllm.config import CUDAGraphMode, VllmConfig

from vllm_ascend.utils import is_310p


def is_310p_dflash_piecewise(vllm_config: VllmConfig) -> bool:
    """Return whether the exact 310P DFlash Piecewise scope is active."""
    speculative_config = vllm_config.speculative_config
    return (
        is_310p()
        and speculative_config is not None
        and speculative_config.method == "dflash"
        and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    )
