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

import vllm_ascend.patch.platform.patch_deepseek_v4_vision  # noqa
import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_utils  # noqa
import vllm_ascend.patch.platform.patch_mamba_block_aligned_split  # noqa
import vllm_ascend.patch.platform.patch_mla_prefill_backend  # noqa
import vllm_ascend.patch.platform.patch_pp_mtp  # noqa
import vllm_ascend.patch.platform.patch_use_v2_model_runner  # noqa
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if get_current_hardware_profile().supports(HardwareCapability.STANDARD_MAMBA_PATCH):
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa
import vllm_ascend.patch.platform.patch_minimax_m2_config  # noqa

import vllm_ascend.patch.platform.patch_structured_output  # noqa
import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa
import vllm_ascend.patch.platform.patch_mamba_manager  # noqa

if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
import vllm_ascend.patch.platform.patch_dyntra_lb_core  # noqa

import vllm_ascend.patch.platform.patch_kv_cache_coordinator  # noqa
import vllm_ascend.patch.platform.patch_speculative_config  # noqa

import vllm_ascend.patch.platform.patch_eplb  # noqa
import vllm_ascend.patch.platform.patch_fused_moe  # noqa
import vllm_ascend.patch.platform.patch_dp_device_ids  # noqa
import vllm_ascend.patch.platform.patch_glm5next_config  # noqa
import vllm_ascend.patch.platform.patch_iquestcoder_config  # noqa

# ** File: platform/patch_kv_cache_utils.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.kv_cache_utils`
#    Why:
#       vLLM's generic KV-cache planner cannot represent GLM-Next's cache
#       topology, where MLA and compressed indexer caches share block IDs while
#       indexer-state and Mamba caches use independent groups.
#    How:
#       Use GLM-Next-specific grouping, tensor layout, and memory accounting.
#       Other models continue to use the original vLLM implementation.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/53906
#       https://github.com/vllm-project/vllm/pull/55219
#    Future Plan:
#       Remove this patch when upstream's generic KV-cache layout supports the
#       Ascend GLM-Next cache topology.
#
# ** File: platform/patch_mamba_config.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config`
#    Why:
#       GLM-Next's compressed indexer requires both Ascend C128 block alignment
#       and C16 alignment after applying `index_kpool`. The generic Mamba
#       configuration does not satisfy these requirements.
#    How:
#       Detect sparse index-kpool models, align the attention block size, and
#       pad the Mamba page using the complete recurrent-state size.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/53906
#       https://github.com/vllm-project/vllm/pull/55449
#    Future Plan:
#       Remove this patch when vLLM supports backend-specific block and page
#       alignment constraints.
#
# ** File: platform/patch_mamba_block_aligned_split.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split`
#    Why:
#       For sparse index-kpool models, upstream splits prefill chunks using the
#       Mamba block size instead of the resolved common cache-group boundary,
#       which can produce incorrect cache-state mappings.
#    How:
#       Align sparse index-kpool prefill chunks to the scheduler block size while
#       preserving small chunks and the existing PD speculative-window behavior.
#    Related PR (if no, explain why):
#       No upstream PR currently covers this Ascend-specific behavior.
#       Related issue: https://github.com/vllm-project/vllm/issues/54392
#    Future Plan:
#       Remove this patch when upstream supports per-group or backend-defined
#       prefill boundaries.
