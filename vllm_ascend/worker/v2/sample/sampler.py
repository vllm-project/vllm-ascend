# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from collections.abc import Callable

import numpy as np
import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample import logprob as upstream_logprob
from vllm.v1.worker.gpu.sample import sampler as upstream_sampler
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

_ORIG_SAMPLER_CALL: Callable[..., SamplerOutput] = upstream_sampler.Sampler.__call__
_ORIG_SAMPLE_RANDOM: Callable[..., tuple[torch.Tensor, torch.Tensor]] = upstream_sampler.Sampler._sample_random


def _is_all_greedy(sampler: upstream_sampler.Sampler, idx_mapping_np: np.ndarray) -> bool:
    if idx_mapping_np.size == 0:
        return True
    if not bool(np.all(idx_mapping_np >= 0)):
        return False
    return bool(np.all(sampler.sampling_states.temperature.np[idx_mapping_np] == 0.0))


def _get_pure_decode_counts(
    sampler: upstream_sampler.Sampler,
    num_reqs: int,
    ref_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return immutable (ones, zeros) views sized to num_reqs for pure decode steps."""
    ones: torch.Tensor | None = getattr(sampler, "_ascend_decode_num_sampled", None)
    zeros: torch.Tensor | None = getattr(sampler, "_ascend_decode_num_rejected", None)
    if (
        ones is None
        or zeros is None
        or ones.shape[0] < num_reqs
        or ones.device != ref_tensor.device
        or ones.dtype != ref_tensor.dtype
    ):
        capacity = max(num_reqs, int(getattr(sampler.sampling_states, "max_num_reqs", num_reqs)))
        ones = torch.ones(capacity, dtype=ref_tensor.dtype, device=ref_tensor.device)
        zeros = torch.zeros(capacity, dtype=ref_tensor.dtype, device=ref_tensor.device)
        sampler._ascend_decode_num_sampled = ones  # type: ignore[attr-defined]
        sampler._ascend_decode_num_rejected = zeros  # type: ignore[attr-defined]
    return ones[:num_reqs], zeros[:num_reqs]


def _get_num_sampled_and_rejected_fast(
    sampler: upstream_sampler.Sampler,
    logits: torch.Tensor,
    input_batch: InputBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not input_batch.has_prefill and input_batch.num_draft_tokens == 0 and logits.shape[0] == input_batch.num_reqs:
        return _get_pure_decode_counts(sampler, input_batch.num_reqs, input_batch.seq_lens)
    return get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs),
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        sampler.req_states.prefill_len.gpu,
    )


def ascend_sampler_call(
    self: upstream_sampler.Sampler,
    logits: torch.Tensor,
    input_batch: InputBatch,
) -> SamplerOutput:
    idx_mapping_np = input_batch.idx_mapping_np
    logprobs_dims = self.get_logprobs_dims(idx_mapping_np)

    # Fast path for plain greedy steps (no logprobs, processors, trace replay,
    # sampling mask, NaN counting, or fp64 gumbel; exact Sampler type only so
    # subclasses such as GPUWatermarkSampler keep their overridden sample()).
    if (
        type(self) is upstream_sampler.Sampler
        and logprobs_dims is None
        and not self.compute_nans
        and self.trace_replay_state is None
        and not self.return_sampling_mask
        and not self.use_fp64_gumbel
        and _is_all_greedy(self, idx_mapping_np)
        and not bool(np.any(self.needs_logits_processing[idx_mapping_np]))
    ):
        sampled = torch.argmax(logits, dim=-1)
        num_sampled, num_rejected = _get_num_sampled_and_rejected_fast(self, logits, input_batch)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=None,
        )

    expanded_idx_mapping = input_batch.expanded_idx_mapping
    idx_mapping = input_batch.idx_mapping
    cu_num_logits_np = input_batch.cu_num_logits_np
    expanded_local_pos = input_batch.expanded_local_pos
    pos = input_batch.positions[input_batch.logits_indices]
    input_ids = input_batch.input_ids[input_batch.logits_indices]
    seq_lens_upper_bound_np = input_batch.seq_lens_cpu_upper_bound.numpy()

    num_nans = get_num_nans(logits) if self.compute_nans else None

    sampled, processed_logits = self.sample(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        return_logprobs=logprobs_dims is not None,
    )

    if self.trace_replay_state is not None:
        self.trace_replay_state.apply_trace(sampled, idx_mapping)

    if logprobs_dims is not None:
        num_logprobs, max_per_req_token_ids = logprobs_dims
        if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            logits = processed_logits
        expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
        cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
        logprobs_tensors = upstream_logprob.compute_topk_scores(
            logits,
            num_logprobs,
            sampled,
            cu_num_logits,
            logprob_token_ids_state=self.logprob_token_ids_state,
            expanded_idx_mapping=input_batch.expanded_idx_mapping,
            max_per_req_token_ids=max_per_req_token_ids,
            logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
        )
    else:
        logprobs_tensors = None

    num_sampled, num_rejected = _get_num_sampled_and_rejected_fast(self, logits, input_batch)

    sampling_mask_tensors = None
    if self.return_sampling_mask:
        max_num_kept = int(np.max(self.sampling_states.top_k.np[idx_mapping_np]))
        sampling_mask_tensors = SamplingMaskTensors.from_logits(
            processed_logits,
            input_batch.cu_num_logits,
            num_sampled,
            max_num_kept,
        )

    return SamplerOutput(
        sampled_token_ids=sampled.view(-1, 1),
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        sampling_mask_tensors=sampling_mask_tensors,
    )


def ascend_sample_random(
    self: upstream_sampler.Sampler,
    processed_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    has_topp_only_rows: bool,
    use_fused_sampler: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_fused_sampler and not self.use_fp64_gumbel and _is_all_greedy(self, idx_mapping_np):
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows)
        return torch.argmax(processed_logits, dim=-1), processed_logits
    return _ORIG_SAMPLE_RANDOM(
        self,
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        top_k,
        top_p,
        has_topp_only_rows,
        use_fused_sampler,
    )
