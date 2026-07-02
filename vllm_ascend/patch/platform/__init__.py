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

import os

import vllm_ascend.patch.platform.patch_camem_allocator  # noqa
import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_utils  # noqa
import vllm_ascend.patch.platform.patch_mla_prefill_backend  # noqa
import vllm_ascend.patch.platform.patch_pp_mtp  # noqa
from vllm_ascend.utils import is_310p, vllm_version_is

if not is_310p():
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa
import vllm_ascend.patch.platform.patch_minimax_m2_config  # noqa
import vllm_ascend.patch.platform.patch_minimax_usage_accounting  # noqa
import vllm_ascend.patch.platform.patch_glm_tool_call_streaming  # noqa
import vllm_ascend.patch.platform.patch_glm47_tool_call_parser  # noqa
import vllm_ascend.patch.platform.patch_minimax_m2_tool_call_parser  # noqa
import vllm_ascend.patch.platform.patch_deepseek_v4_tool_call_parser  # noqa
import vllm_ascend.patch.platform.patch_weight_transfer_engine  # noqa
import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa
import vllm_ascend.patch.platform.patch_tool_choice_none_content  # noqa
import vllm_ascend.patch.platform.patch_mamba_manager  # noqa

if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

import vllm_ascend.patch.platform.patch_balance_schedule  # noqa

import vllm_ascend.patch.platform.patch_kv_cache_coordinator  # noqa
import vllm_ascend.patch.platform.patch_speculative_config  # noqa

if not vllm_version_is("0.23.0"):
    import vllm_ascend.patch.platform.patch_fused_moe  # noqa
