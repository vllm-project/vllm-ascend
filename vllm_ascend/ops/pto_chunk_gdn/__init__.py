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
"""PTO chunk-GDN megakernel module for vLLM-Ascend.

Provides a Bisheng-JIT-compiled fused Ascend NPU megakernel for the chunk
GatedDeltaNet (GDN) recurrent layer used in Qwen3.5 / Qwen3.6 models,
replacing the default Triton baseline during prefill.

Enable via:
    VLLM_ASCEND_PTO_CHUNK_GDN=1   (set before launching vLLM)

The decode phase always uses the original Triton implementation.
"""
from vllm_ascend.ops.pto_chunk_gdn.compile import BLOCK_DIM, PTO_LIB_PATH
from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel
from vllm_ascend.ops.pto_chunk_gdn.worker_hook import apply_pto_gdn_patch, is_pto_gdn_patch_active

__all__ = [
    "BLOCK_DIM",
    "PTO_LIB_PATH",
    "run_mega_kernel",
    "apply_pto_gdn_patch",
    "is_pto_gdn_patch_active",
]
