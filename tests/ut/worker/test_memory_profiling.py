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

import pytest
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.worker.memory_profiling import validate_memory_profile


@pytest.mark.parametrize(
    "profiled_free_memory_gib",
    [pytest.param(60, id="equal"), pytest.param(59, id="shrink")],
)
def test_validate_memory_profile_accepts_non_increasing_free_memory(profiled_free_memory_gib: int):
    validate_memory_profile(
        init_free_memory=60 * GiB_bytes,
        after_profile_free_memory=profiled_free_memory_gib * GiB_bytes,
    )


def test_validate_memory_profile_rejects_increased_free_memory():
    with pytest.raises(ValueError, match="Initial free memory 60.0 GiB, current free memory 61.0 GiB"):
        validate_memory_profile(
            init_free_memory=60 * GiB_bytes,
            after_profile_free_memory=61 * GiB_bytes,
        )
