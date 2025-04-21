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
# What's Patched and how it works:
# ** File: platform/patch_0_8_4/patch_config.py**
#   1. `vllm.config.ModelConfig.__init__()`
#    Why:
#       It is hard coded for sleep mode to support cuda platform only
#    How：
#       Using a new method to check if sleep mode is available
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       https://github.com/vllm-project/vllm/pull/16562
#    Future Plan:
#       This patch is only used for 084 and can't be revert. just keep as it is.

import vllm_ascend.patch.platform.patch_0_8_4.patch_config  # noqa
import vllm_ascend.patch.platform.patch_0_8_4.patch_distributed  # noqa
