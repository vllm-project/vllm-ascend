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
from vllm.v1.sample.sampler import _SAMPLING_EPS

from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)


def _copy_to_default_format(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
    ).copy_(tensor)


def _random_sample_310p(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    uniforms_cpu = torch.empty(
        (logits.shape[0], 1),
        device="cpu",
        dtype=torch.float32,
    )
    if len(generators) != logits.shape[0]:
        uniforms_cpu.uniform_()
    for i, generator in generators.items():
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.set_state(generator.get_state())
        uniforms_cpu[i].uniform_(generator=cpu_generator)
        generator.set_state(cpu_generator.get_state())

    logits_cpu = logits.to("cpu", dtype=torch.float32)
    probs_cpu = logits_cpu.softmax(dim=-1, dtype=torch.float32)
    cdf_cpu = probs_cpu.cumsum(dim=-1)
    cdf_cpu[:, -1] = 1.0
    sampled = torch.searchsorted(cdf_cpu, uniforms_cpu, right=False)
    return sampled.to(torch.int64).view(-1)


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

        sampled = _random_sample_310p(logits, generators)
        return sampled.to(device=logits.device, non_blocking=True), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        logprobs_mode_override=None,
    ):
        if not sampling_metadata.all_greedy:
            logits = _copy_to_default_format(logits)
        return super().sample(logits, sampling_metadata, logprobs_mode_override)

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        temp_cpu = temp.to("cpu", dtype=torch.float32)
        if not all_random:
            temp_cpu = torch.where(
                temp_cpu < _SAMPLING_EPS,
                torch.ones_like(temp_cpu),
                temp_cpu,
            )
        inv_temp = temp_cpu.reciprocal().to(
            device=logits.device,
            dtype=logits.dtype,
            non_blocking=True,
        )
        return logits.mul_(inv_temp.unsqueeze(dim=1))

    def do_async_exponential(self, b_s, head_dim, generators):
        return
