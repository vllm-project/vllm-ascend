# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/sampler.py.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.penalties import \
    apply_penalties_and_temperature


class AscendSampler(Sampler):

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override sample method because we need to override triton operators
        called in the method.
        """
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply penalties and temperature in place.
        apply_penalties_and_temperature(logits, sampling_metadata)
        # Apply min_p in place.
        if sampling_metadata.min_p is not None:
            apply_min_p(logits, sampling_metadata.min_p)
        # Apply top_k and/or top_p. This might return a new tensor.
        logits = apply_top_k_top_p(logits, sampling_metadata.top_k,
                                   sampling_metadata.top_p)

        sampled = gumbel_sample(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.seeds,
            sampling_metadata.pos,
            apply_temperature=False,
        )
        return sampled, logits
