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


import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_dp_supervisor  # noqa
import vllm_ascend.patch.platform.patch_kimi_k31_config  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_admission  # noqa
import vllm_ascend.patch.platform.patch_mamba_block_aligned_split  # noqa
import vllm_ascend.patch.platform.patch_parallel_config  # noqa
import vllm_ascend.patch.platform.patch_pp_mtp  # noqa
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if get_current_hardware_profile().supports(HardwareCapability.STANDARD_MAMBA_PATCH):
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa
import vllm_ascend.patch.platform.patch_structured_output  # noqa
import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa
import vllm_ascend.patch.platform.patch_torch_utils  # noqa
import vllm_ascend.patch.platform.patch_mamba_manager  # noqa

import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
import vllm_ascend.patch.platform.patch_engine_core  # noqa
import vllm_ascend.patch.platform.patch_mm_staging_pool  # noqa
import vllm_ascend.patch.platform.patch_metrics_level  # noqa

import vllm_ascend.patch.platform.patch_kv_cache_coordinator  # noqa
import vllm_ascend.patch.platform.patch_speculative_config  # noqa
import vllm_ascend.patch.platform.patch_replayssm_config  # noqa

import vllm_ascend.patch.platform.patch_eplb  # noqa
import vllm_ascend.patch.platform.patch_fused_moe  # noqa

# The Ascend Engram adapter requires `vllm.config.engram` (vLLM main/v0.31.0+).
import vllm_ascend.patch.platform.patch_engram_config  # noqa

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

# ** File: platform/patch_kv_cache_admission.py **
# Targets SWA and chunked-local admission in SingleTypeKVCacheManager.
# First-time admission shifts the cap past skipped prefix entries. Existing
# requests offset the cap only with non-NULL blocks held exclusively by that
# request; releasing shared blocks does not return capacity to the pool.
# Uncapped allocation and full-attention managers retain upstream behavior.
# Upstream plan: contribute physical-block accounting to vLLM and remove this
# patch once both supported vLLM versions cover first-time and shared-block
# admission accounting.

# ** File: platform/patch_mm_staging_pool.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.multimodal.inputs._nested_tensors_h2d`,
#      `MultiModalFlatField._reduce_data`, `MultiModalBatchedField._reduce_data`
#    Why:
#      torch_npu 2.10 pinned allocations go through aclrtMallocHostWithCfg (VA
#      flag), binding every allocation to 2MiB base-heap physical pages with no
#      small-page fallback. The multimodal staging/merge path allocates
#      per-request temporaries and cross-request merge outputs via
#      new_empty(pin_memory=True), producing a high-frequency stream of
#      variable-size driver grabs (tiny metadata + image staging/merges,
#      observed 85MB-1GB). Under host memory pressure the 2MiB buddy inventory
#      drains and these grabs fail with error 207001/EL0018, crashing serving.
#    How:
#      Fixed pinned staging pool allocated once per process at first use
#      (VA-backed, full DMA speed); all mm staging/merge temporaries are
#      routed through pool slices (multi-slot contiguous slices for oversize
#      merges, event-gated slot recycle recorded after the H2D enqueue).
#      Every pool interaction degrades to the original path on failure.
#      Configured via additional_config.mm_staging_config (pinned_pool_mb,
#      slot_mb); pinned_pool_mb=0 disables the pool. The patch never imports
#      vllm.multimodal itself: a sys.meta_path import hook applies the
#      attribute patches right after vllm imports vllm.multimodal.inputs on
#      its own (importing the module during platform-plugin registration
#      starves the EngineCore input rings; validated 2026-11-22 on the
#      issue-21094 boxes).
#    Related PR (if no, explain why):
#      No upstream vllm PR: NPU-driver-specific allocation behavior (VA
#      base-heap pinning) does not exist on CUDA; fix belongs to the platform
#      patch layer. Allocator-side alternative tracked in torch_npu MR 49638.
#    Future Plan:
#      Keep while torch_npu lacks a failure fallback for the VA pinned path.
