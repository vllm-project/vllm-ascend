#
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
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)
from vllm_ascend.utils import global_stream, npu_stream_switch

_CPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, int]] = {}


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """310P-specific random sampling with CPU exponential generation for q."""
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        q = q.cpu()
        if len(generators) != q.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                cache_entry = _CPU_GENERATOR_CACHE_310P.get(i)
                if cache_entry is None or cache_entry[1] != id(generator):
                    cpu_generator = torch.Generator(device="cpu")
                    try:
                        # Keep RNG stream consistent with the original generator.
                        cpu_generator.set_state(generator.get_state())
                    except Exception:
                        cpu_generator.manual_seed(generator.initial_seed())
                    cache_entry = (cpu_generator, id(generator))
                    _CPU_GENERATOR_CACHE_310P[i] = cache_entry
                cpu_generator, _ = cache_entry
                q[i].exponential_(generator=cpu_generator)
        q = q.npu()
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if vllm_is_batch_invariant():
            return super().forward_native(logits, generators, k, p)

        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)
