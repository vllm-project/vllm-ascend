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

from vllm_ascend.utils import vllm_version_is

# Import specific patches for different versions
if vllm_version_is("0.10.0"):
    from vllm_ascend.patch.platform import patch_0_10_0  # noqa: F401
    from vllm_ascend.patch.platform import patch_common  # noqa: F401
else:
    from vllm_ascend.patch.platform import patch_common  # noqa: F401
    from vllm_ascend.patch.platform import patch_main  # noqa: F401
