# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from dataclasses import dataclass, fields

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin


@triton.jit
def _fused_prepare_inputs_kernel(
    pos_ptr,
    seq_lens_ptr,
    input_ids_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    logits_indices_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id >= num_reqs:
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)

    cu_num_logits_start = tl.load(cu_num_logits_ptr + req_id)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + req_id + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS
    logits_start = end - num_logits

    for i in tl.range(0, num_logits, LOGITS_BLOCK_SIZE):
        lblock = i + tl.arange(0, LOGITS_BLOCK_SIZE)
        lmask = lblock < num_logits
        tl.store(
            expanded_idx_mapping_ptr + cu_num_logits_start + lblock,
            req_state_idx,
            mask=lmask,
        )
        tl.store(
            expanded_local_pos_ptr + cu_num_logits_start + lblock,
            lblock,
            mask=lmask,
        )
        tl.store(
            logits_indices_ptr + cu_num_logits_start + lblock,
            logits_start + lblock,
            mask=lmask,
        )

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        return

    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    if num_draft_tokens > 0:
        for i in tl.range(0, num_draft_tokens, LOGITS_BLOCK_SIZE):
            dblock = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            dmask = dblock < num_draft_tokens
            draft_toks = tl.load(
                draft_tokens_ptr + req_state_idx * draft_tokens_stride + dblock,
                mask=dmask,
            )
            tl.store(
                input_ids_ptr + end - num_draft_tokens + dblock,
                draft_toks,
                mask=dmask,
            )


def fused_prepare_inputs(
    input_buffers: "AscendInputBuffers",
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    total_num_logits: int,
    num_new_sampled_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse prepare_pos_seq_lens, expand_idx_mapping, and combine_sampled_and_draft_tokens."""
    assert num_new_sampled_tokens in (0, 1), f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    num_reqs = idx_mapping.shape[0]
    if (
        input_buffers.expanded_idx_mapping.shape[0] >= total_num_logits
        and input_buffers.expanded_idx_mapping.dtype == idx_mapping.dtype
    ):
        expanded_idx_mapping = input_buffers.expanded_idx_mapping[:total_num_logits]
    else:
        expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    if input_buffers.expanded_local_pos.shape[0] >= total_num_logits:
        expanded_local_pos = input_buffers.expanded_local_pos[:total_num_logits]
    else:
        expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=idx_mapping.device)
    if input_buffers.logits_indices.shape[0] >= total_num_logits:
        logits_indices = input_buffers.logits_indices[:total_num_logits]
    else:
        logits_indices = torch.empty(total_num_logits, dtype=torch.int64, device=input_buffers.device)
    logits_block_size = max(16, triton.next_power_of_2(draft_tokens.shape[-1] + num_new_sampled_tokens))
    _fused_prepare_inputs_kernel[(num_reqs + 1,)](
        input_buffers.positions,
        input_buffers.seq_lens,
        input_buffers.input_ids,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        last_sampled_tokens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        expanded_idx_mapping,
        expanded_local_pos,
        logits_indices,
        input_buffers.seq_lens.shape[0],
        BLOCK_SIZE=1024,
        LOGITS_BLOCK_SIZE=logits_block_size,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
    )
    return expanded_idx_mapping, expanded_local_pos, logits_indices


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        del self.query_start_loc

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc: torch.Tensor = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
            device=device,
        )

        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()

        self.expanded_idx_mapping: torch.Tensor = torch.zeros(
            max_num_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.expanded_local_pos: torch.Tensor = torch.zeros(
            max_num_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.logits_indices: torch.Tensor = torch.zeros(
            max_num_tokens,
            dtype=torch.int64,
            device=device,
        )

        # Logical query token count, scoped to _dummy_run before graph padding.
        # None keeps the default layout for other dummy/capture paths.
        self.dummy_num_tokens: int | None = None


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    # InputBatch gained max_query_len default field,
    # requiring the child's first field to also have a default.
    seq_lens_np: np.ndarray = None  # type: ignore[assignment, no-redef]
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    is_dummy: bool = False
    contiguous_logits_indices: bool = False

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: AscendInputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "AscendInputBatch":
        """Override the make_dummy method to calculate seq_lens_np."""
        # The incoming count includes graph padding. Save it for the model input
        # views below, while building query metadata from the original scheduled
        # count used for DP synchronization. Only _dummy_run supplies that override.
        # TODO: Remove this workaround once upstream dummy-batch preparation
        # preserves logical query lengths separately from graph padding.
        num_tokens_after_padding = num_tokens
        if input_buffers.dummy_num_tokens is not None:
            # PIECEWISE pads model inputs without adding scheduled queries.
            num_tokens = input_buffers.dummy_num_tokens
            assert num_tokens <= num_tokens_after_padding
        input_batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
            max_query_len=max_query_len,
            is_padding=is_padding,
        )
        if num_tokens_after_padding != num_tokens:
            input_batch.num_tokens_after_padding = num_tokens_after_padding
            input_buffers.input_ids[num_tokens:num_tokens_after_padding].zero_()
            input_buffers.positions[num_tokens:num_tokens_after_padding].zero_()
            input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(True)
            input_batch.input_ids = input_buffers.input_ids[:num_tokens_after_padding]
            input_batch.positions = input_buffers.positions[:num_tokens_after_padding]
            input_batch.is_padding = input_buffers.is_padding[:num_tokens_after_padding]
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
        input_buffers.seq_lens_np[num_reqs:] = 0
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
        update_cos_sin(input_batch.positions)
        base_fields = {field.name: getattr(input_batch, field.name) for field in fields(InputBatch)}
        return cls(
            **base_fields,
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.DecodeOnly,
            is_dummy=True,
        )
