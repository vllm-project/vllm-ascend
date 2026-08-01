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

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
import vllm.v1.worker.gpu.sample.sampler as gpu_sampler
import vllm.v1.worker.gpu.sample.states as states
from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """Same as upstream apply_top_k_top_p, with CPU/Triton branches disabled.

    Equivalent to commenting out these two upstream blocks:
      - if current_platform.is_cpu(): ...
      - if HAS_TRITON and logits.shape[0] >= 8: ...
    Always falls through to the PyTorch implementation.

    V2 imports apply_top_k_top_p into ``gpu.sample.sampler`` (hot path in
    ``Sampler.sample``) and ``gpu.sample.states`` at import time, so the source
    module and both local bindings must be rebound for nightly to take effect.
    """
    # Log at entry so we can tell the patched function was invoked even when
    # k/p are both None (early return) or when temperature fails earlier.
    logger.info_once(
        "[patch_topk_topp] Ascend V2 patch invoked: forcing apply_top_k_top_p_pytorch "
        "(batch=%s, has_k=%s, has_p=%s)",
        logits.shape[0] if logits is not None else None,
        k is not None,
        p is not None,
    )

    if p is None and k is None:
        return logits

    # Disabled on Ascend (commented out upstream branches):
    # if current_platform.is_cpu():
    #     if HAS_TRITON:
    #         return apply_top_k_top_p_triton(logits, k, p)
    #     return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
    #
    # if HAS_TRITON and logits.shape[0] >= 8:
    #     return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation.
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p
# V2 Sampler.sample() hot path uses this import-time binding.
gpu_sampler.apply_top_k_top_p = apply_top_k_top_p
# V2 SamplingStates.apply_top_k_top_p uses this import-time binding.
states.apply_top_k_top_p = apply_top_k_top_p
logger.info_once(
    "[patch_topk_topp] Ascend V2 patch loaded: rebound apply_top_k_top_p on "
    "topk_topp_sampler, gpu.sample.sampler, and gpu.sample.states"
)
