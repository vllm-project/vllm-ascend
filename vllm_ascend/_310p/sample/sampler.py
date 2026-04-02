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
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    uniform_rows = []
    row_shape = (probs.shape[1],)
    for i in range(probs.shape[0]):
        generator = generators.get(i)
        uniform_rows.append(
            torch.rand(
                row_shape,
                device=probs.device,
                dtype=probs.dtype,
                generator=generator,
            )
        )
    uniforms = torch.stack(uniform_rows, dim=0)
    min_uniform = torch.tensor(
        torch.finfo(uniforms.dtype).tiny,
        device=uniforms.device,
        dtype=uniforms.dtype,
    )
    q = -torch.log(torch.maximum(uniforms, min_uniform))
    sampled_scores = torch.div(probs, q)
    return torch.topk(sampled_scores, k=1, dim=-1).indices.view(-1)


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

        probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
        sampled = _random_sample_310p(probs, generators)
        return sampled, logits_to_return


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
        return logits * inv_temp.unsqueeze(dim=1)

    def do_async_exponential(self, b_s, head_dim, generators):
        return
