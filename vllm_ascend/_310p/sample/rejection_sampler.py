#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from contextlib import contextmanager

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from vllm_ascend._310p.sample.sampler import fill_exponential_310p
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)


@contextmanager
def _force_pytorch_rejection_path(fn, *, greedy_fn=None):
    """Route the base rejection sampler through its PyTorch fallbacks on 310P.

    310P has no working Triton, so ``HAS_TRITON`` is forced off (otherwise the base
    ``rejection_sample`` hits ``cal_grid_and_block_size`` -> ``get_vectorcore_num``
    and fails with "Device properties not initialized"). The PyTorch
    recovered-token sampler is bound at the same time. Both module globals are
    restored on exit so nothing else is affected.
    """
    original_has_triton = rejection_sampler_module.HAS_TRITON
    original_recovered = rejection_sampler_module.sample_recovered_tokens
    original_greedy = rejection_sampler_module.rejection_greedy_sample_pytorch
    rejection_sampler_module.HAS_TRITON = False
    rejection_sampler_module.sample_recovered_tokens = fn
    if greedy_fn is not None:
        rejection_sampler_module.rejection_greedy_sample_pytorch = greedy_fn
    try:
        yield
    finally:
        rejection_sampler_module.HAS_TRITON = original_has_triton
        rejection_sampler_module.sample_recovered_tokens = original_recovered
        rejection_sampler_module.rejection_greedy_sample_pytorch = original_greedy


def _rejection_greedy_sample_pytorch_310(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    draft_tokens_per_req,
    max_spec_len,
    is_greedy=None,
):
    """310P-safe greedy verification using only aligned matrix writes.

    K=15 makes a flattened request row 60 bytes wide. A row slice or the bonus
    cell at column 15 can therefore expose a non-64-byte-aligned pointer to an
    address Add kernel. Gather ragged inputs into K-wide matrices from their
    aligned storage bases, build a K+1 result, and write the complete output
    matrix once so no sub-row destination reaches an NPU kernel.
    """
    batch_size = output_token_ids.size(0)
    device = output_token_ids.device
    draft_counts = tuple(int(count) for count in draft_tokens_per_req)
    if len(draft_counts) != batch_size:
        raise ValueError("draft_tokens_per_req must contain one entry per request")
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    gather_rows: list[list[int]] = []
    valid_rows: list[list[bool]] = []
    bonus_rows: list[list[bool]] = []
    cursor = 0
    for count in draft_counts:
        if count < 0 or count > max_spec_len:
            raise ValueError(f"invalid draft token count {count}; max_spec_len={max_spec_len}")
        fallback_index = cursor if count else 0
        gather_rows.append(
            [cursor + position if position < count else fallback_index for position in range(max_spec_len)]
        )
        valid_rows.append([position < count for position in range(max_spec_len)])
        bonus_rows.append([position == count for position in range(max_spec_len + 1)])
        cursor += count

    if cursor != draft_token_ids.numel():
        raise ValueError("draft_tokens_per_req does not match flattened draft_token_ids")

    if cursor:
        gather_indices = torch.tensor(gather_rows, dtype=torch.long, device=device)
        flat_indices = gather_indices.reshape(-1)
        aligned_draft = torch.index_select(draft_token_ids, 0, flat_indices).reshape(batch_size, max_spec_len)
        aligned_target = torch.index_select(target_argmax, 0, flat_indices).reshape(batch_size, max_spec_len)
    else:
        aligned_draft = torch.zeros(
            (batch_size, max_spec_len),
            dtype=draft_token_ids.dtype,
            device=device,
        )
        aligned_target = torch.zeros(
            (batch_size, max_spec_len),
            dtype=target_argmax.dtype,
            device=device,
        )

    valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=device)
    bonus_mask = torch.tensor(bonus_rows, dtype=torch.bool, device=device)
    positions = torch.arange(max_spec_len, device=device).reshape(1, -1).expand(batch_size, -1)
    mismatch_positions = torch.where(
        valid_mask & (aligned_draft != aligned_target),
        positions,
        max_spec_len,
    )
    first_mismatch = torch.min(mismatch_positions, dim=1).values
    greedy_rows = is_greedy.reshape(batch_size, 1)
    copy_mask = valid_mask & (positions <= first_mismatch.reshape(batch_size, 1)) & greedy_rows
    copy_mask = torch.cat(
        (
            copy_mask,
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    count_tensor = torch.tensor(draft_counts, dtype=torch.long, device=device)
    needs_bonus = is_greedy & (first_mismatch >= count_tensor)
    write_mask = copy_mask | (bonus_mask & needs_bonus.reshape(batch_size, 1))

    target_candidates = torch.cat(
        (
            aligned_target.to(output_token_ids.dtype),
            torch.zeros(
                (batch_size, 1),
                dtype=output_token_ids.dtype,
                device=device,
            ),
        ),
        dim=1,
    )
    bonus_candidates = bonus_token_ids.reshape(batch_size, 1).to(output_token_ids.dtype)
    bonus_candidates = bonus_candidates.expand(-1, max_spec_len + 1)
    candidates = torch.where(bonus_mask, bonus_candidates, target_candidates)
    output_token_ids.copy_(torch.where(write_mask, candidates, output_token_ids))


def _get_rejection_sample_greedy_310_op():
    try:
        return torch.ops._C_ascend.npu_rejection_sample_greedy_310
    except AttributeError:
        return None


def _rejection_greedy_sample_310(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    draft_tokens_per_req,
    max_spec_len,
    is_greedy=None,
):
    """Use the fused 310P op for all-greedy batches, with a safe fallback."""
    if is_greedy is None:
        rejection_op = _get_rejection_sample_greedy_310_op()
        if rejection_op is not None:
            rejection_op(
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                output_token_ids,
                max_spec_len,
            )
            return

    _rejection_greedy_sample_pytorch_310(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        draft_tokens_per_req,
        max_spec_len,
        is_greedy,
    )


class AscendRejectionSampler310(AscendRejectionSampler):
    """310P rejection sampler: PyTorch recovered-token path with CPU RNG (no Triton)."""

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        with _force_pytorch_rejection_path(
            self.sample_recovered_tokens,
            greedy_fn=_rejection_greedy_sample_310,
        ):
            return super().forward(metadata, draft_probs, logits, sampling_metadata)

    def sample_recovered_tokens(
        self,
        max_spec_len: int,
        num_draft_tokens: list[int],
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        device: torch.device,
        use_block_verify: bool = False,
        target_indices: torch.Tensor | None = None,
        global_vocab_size: int | None = None,
        enable_reduce_sampling: bool = False,
    ) -> torch.Tensor:
        batch_size = len(num_draft_tokens)
        vocab_size = target_probs.shape[-1]

        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
        has_draft_mask = num_draft_tensor > 0
        fill_exponential_310p(q, sampling_metadata.generators, has_draft_mask)

        recovered_token_ids = torch.empty_like(draft_token_ids)
        if use_block_verify:
            sample_recovered_tokens_blockwise_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        else:
            sample_recovered_tokens_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        return recovered_token_ids
