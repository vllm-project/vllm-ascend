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


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """Same as upstream apply_top_k_top_p, with CPU/Triton branches disabled.

    Equivalent to commenting out these two upstream blocks:
      - if current_platform.is_cpu(): ...
      - if HAS_TRITON and logits.shape[0] >= 8: ...
    Always falls through to the PyTorch implementation.
    """
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
