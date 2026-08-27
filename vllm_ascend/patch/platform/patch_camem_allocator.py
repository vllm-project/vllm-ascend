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
import vllm.config.model as model_config_module


def _patched_is_cumem_allocator_available() -> bool:
    # NPUPlatform declares sleep mode support and vllm-ascend uses CaMemAllocator
    # in the worker path. Avoid importing the extension here because ModelConfig
    # validation runs before custom op initialization.
    return True


if hasattr(model_config_module, "is_cumem_allocator_available"):
    model_config_module.is_cumem_allocator_available = _patched_is_cumem_allocator_available
