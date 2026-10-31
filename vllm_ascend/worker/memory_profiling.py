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

from vllm.utils.mem_constants import GiB_bytes


def validate_memory_profile(init_free_memory: int, after_profile_free_memory: int) -> None:
    """Accept equal or reduced free memory and reject increased snapshots."""
    if after_profile_free_memory <= init_free_memory:
        return

    init_free_memory_gib = init_free_memory / GiB_bytes
    after_profile_free_memory_gib = after_profile_free_memory / GiB_bytes
    raise ValueError(
        "Error in memory profiling. "
        f"Initial free memory {init_free_memory_gib} GiB, "
        f"current free memory {after_profile_free_memory_gib} GiB. "
        "This happens when other processes sharing the same container "
        "release GPU memory while vLLM is profiling during initialization. "
        "To fix this, ensure consistent GPU memory allocation or "
        "isolate vLLM in its own container."
    )
