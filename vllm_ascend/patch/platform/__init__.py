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

import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_glm53_reasoning  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_utils  # noqa
import vllm_ascend.patch.platform.patch_mamba_block_aligned_split  # noqa
import vllm_ascend.patch.platform.patch_mla_prefill_backend  # noqa
import vllm_ascend.patch.platform.patch_parallel_config  # noqa
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

import vllm_ascend.patch.platform.patch_swa_bounded_replay  # noqa
import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
import vllm_ascend.patch.platform.patch_engine_core  # noqa

import vllm_ascend.patch.platform.patch_kv_cache_coordinator  # noqa
import vllm_ascend.patch.platform.patch_speculative_config  # noqa
import vllm_ascend.patch.platform.patch_replayssm_config  # noqa

import vllm_ascend.patch.platform.patch_eplb  # noqa
import vllm_ascend.patch.platform.patch_fused_moe  # noqa

# The Ascend Engram adapter requires `vllm.config.engram` (vLLM main/v0.31.0+).
import vllm_ascend.patch.platform.patch_engram_config  # noqa

import vllm_ascend.patch.platform.patch_indexer_kv_dtype  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_dtype  # noqa
import vllm_ascend.patch.platform.patch_example_connector  # noqa

# ** File: platform/patch_example_connector.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.ec_transfer.ec_connector.example_connector.ECExampleConnector.start_load_caches`
#    Why:
#       The upstream method only qualifies the safetensors load device with an
#       index when `current_platform.is_cuda_alike()` returns True. Ascend NPU
#       returns False, so in TP>1 every rank loads the EC cache to the default
#       device (npu:0), causing rank 1 to access cross-device tensors and
#       trigger NPU AI Core errcode=95 (MTE DDR address out of range).
#    How:
#       Monkey-patch `start_load_caches` to extend the device-index
#       qualification to `device == "npu"`. The patch only takes effect on EC
#       consumer nodes; producers and non-EC deployments fall through to the
#       original upstream behavior.
#    Related PR (if no, explain why):
#       No upstream PR currently covers this Ascend-specific behavior.
#    Future Plan:
#       Remove this patch when upstream's `is_cuda_alike()` covers Ascend NPU
#       or when the device-index qualification is applied unconditionally.
#
# ** File: platform/patch_kv_cache_utils.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.kv_cache_utils` (v0.32.0) /
#      `vllm.v1.core.kv_cache_planning` (main, vLLM #61183)
#    Why:
#       vLLM PR #40860 added a restriction that hybrid KV cache groups with
#       multiple block sizes do not support decode context parallelism.
#       This restriction is correct for CUDA but not for Ascend, which
#       implements context parallelism for MLA and SWA-MLA layers separately,
#       and whose GLM-Next layout keeps a non-cacheable fixed tail group.
#    How:
#       Resolve the scheduler block size as lcm(group block sizes) * dcp for
#       the multiple-group + DCP case, and drop a non-cacheable tail group
#       before deriving the hash block size.
#    Related PR (if no, explain why):
#       vLLM PR #40860 ([Feat] DeepSeek V4 Rebased).
#    Future Plan:
#       Remove this patch once upstream vLLM supports hybrid KV cache + CP for
#       non-CUDA backends, or exposes a platform hook for this behavior.
#
#       On main, every other Ascend planning rule (grouping, materialization,
#       memory accounting for DSV4/DSV4.1, GLM-Next, Kimi-K3 DSpark, and
#       TurboQuant) lives in `vllm_ascend.core.kv_cache_config_builder`, which
#       the platform selects through
#       `NPUPlatform.get_kv_cache_config_builder_cls`. The v0.31.0 lane, which
#       has no pluggable builder, still applies the same rules from this file.
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
