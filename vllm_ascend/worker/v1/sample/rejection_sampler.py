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

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as probabilistic_rejection_sample,
)

NO_LOGPROBS = -1


@dataclass
class BridgeSamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None


@triton.jit
def _strict_rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1,
            target_sampled,
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def _strict_rejection_sample_cpu(
    target_sampled: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = int(cu_num_logits.shape[0]) - 1
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
    for req_idx in range(num_reqs):
        start_idx = int(cu_num_logits[req_idx].item())
        end_idx = int(cu_num_logits[req_idx + 1].item())
        count = 0
        rejected = False
        for row in range(start_idx, end_idx - 1):
            if rejected:
                continue
            sampled[req_idx, count] = target_sampled[row]
            count += 1
            if target_sampled[row] != draft_sampled[row + 1]:
                rejected = True
        if not rejected:
            sampled[req_idx, count] = target_sampled[end_idx - 1]
            count += 1
        num_sampled[req_idx] = count
    return sampled, num_sampled


def strict_rejection_sample(
    target_sampled: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_sampled.device.type == "cpu":
        return _strict_rejection_sample_cpu(
            target_sampled,
            draft_sampled,
            cu_num_logits,
            num_speculative_steps,
        )

    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
    _strict_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


class AscendRejectionSampler:
    """Speculative rejection sampler for the v1 Ascend GPU-sampler bridge."""

    def __init__(
        self,
        sampler,
        spec_config,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.rejection_sample_method = getattr(
            spec_config,
            "rejection_sample_method",
            "strict",
        )
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if self.rejection_sample_method == "synthetic":
            synthetic_acceptance_rates = getattr(
                spec_config,
                "synthetic_acceptance_rates",
                None,
            )
            if synthetic_acceptance_rates is not None:
                from vllm.v1.spec_decode.utils import (
                    unconditional_to_conditional_rates,
                )

                self.synthetic_conditional_rates = torch.tensor(
                    unconditional_to_conditional_rates(synthetic_acceptance_rates),
                    dtype=torch.float32,
                    device=device,
                )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch,
        draft_logits: torch.Tensor | None = None,
    ) -> BridgeSamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        num_nans = None

        if self.rejection_sample_method == "strict":
            sampler_output = self.sampler(logits, input_batch)
            sampled, num_sampled = strict_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
            logprobs_tensors = sampler_output.logprobs_tensors
        elif self.rejection_sample_method == "probabilistic":
            if draft_logits is None:
                raise NotImplementedError(
                    "probabilistic rejection sampling requires draft logits from "
                    "the drafter."
                )
            pos = input_batch.positions[input_batch.logits_indices]
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )
            logprobs_tensors = None
        elif self.rejection_sample_method == "synthetic":
            raise NotImplementedError(
                "synthetic rejection sampling is not supported by the Ascend "
                "GPU-sampler bridge yet."
            )
        else:
            raise ValueError(
                f"Unknown rejection sample method: {self.rejection_sample_method}"
            )

        return BridgeSamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
