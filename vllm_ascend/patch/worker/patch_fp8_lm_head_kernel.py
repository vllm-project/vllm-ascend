#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Register the Ascend FP8 scaled-MM kernel in every NPU worker process.

The upstream platform FP8 kernel table is per process: every worker that
loads a model with an online FP8 method must register the OOT kernel before
``Fp8PtpcOnlineLinearMethod.create_weights`` runs, which is why this is a
worker patch imported from ``vllm_ascend.patch.worker``. The registration,
the batch-invariance binding and the vLLM version guard live with the kernel
in ``vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm``.
"""

try:
    from vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm import (
        register_ascend_fp8_lm_head_kernel,
    )
except ImportError:
    # vLLM builds without the scaled_mm kernel package (< 0.26.0) cannot run
    # this OOT FP8 kernel at all; skip registration there.
    register_ascend_fp8_lm_head_kernel = None

if register_ascend_fp8_lm_head_kernel is not None:
    register_ascend_fp8_lm_head_kernel()
