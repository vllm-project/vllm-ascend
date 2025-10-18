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
import os

from vllm.logger import logger

import vllm_ascend.patch.platform.patch_common.patch_config  # noqa
import vllm_ascend.patch.platform.patch_common.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_common.patch_mamba_config  # noqa


def patch_v1_executor():
    try:
        dynamic_eplb = os.getenv("DYNAMIC_EPLB", False) or os.getenv(
            "EXPERT_MAP_RECORD", False)
        if dynamic_eplb:
            import vllm_ascend.patch.platform.patch_common.patch_multiproc_executor  # noqa
        else:
            logger.warning("Do not patch v1 executor.")
    except RuntimeError as e:
        logger.warning(
            f"Fail to patch v1 executor, please add environment params DYNAMIC_EPLB or EXPERT_MAP_RECORD : {e}"
        )


patch_v1_executor()
