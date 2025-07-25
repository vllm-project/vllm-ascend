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


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    # fix pytorch schema check error, remove this line after pytorch
    # is upgraded to 2.7.0
    import vllm_ascend.patch.worker.patch_common.patch_utils  # noqa: F401

    import os
    if os.getenv("VLLM_ASCEND_ENABLE_PD_CHUNK_SCHEDULE") == '1':
        import vllm_ascend.patch.platform.patch_0_9_1.patch_scheduler  # isort: skip  # noqa: F401

    from .models import register_model

    import vllm_ascend.patch.platform.patch_0_9_1.patch_decorator  # noqa: F401
    register_model()
