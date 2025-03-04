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

from typing import Dict

import torch
from vllm.model_executor.layers import rejection_sampler


def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    k: int,
    seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q = -torch.log(torch.rand_like(q))
    else:
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            # Note: generator might be None for non seeded
            q[start:end].exponential_(1.0, generator=generator)
            start = end

    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


rejection_sampler._multinomial = _multinomial
