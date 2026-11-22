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
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
        *,
        tree_enabled: bool = False,
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

        # Logical query token count, scoped to _dummy_run before graph padding.
        # None keeps the default layout for other dummy/capture paths.
        self.dummy_num_tokens: int | None = None
        # Siblings share RoPE positions but keep separate KV slots.
        self.kv_cache_positions = torch.empty_like(self.positions) if tree_enabled else None


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    # InputBatch gained max_query_len default field,
    # requiring the child's first field to also have a default.
    seq_lens_np: np.ndarray = None  # type: ignore[assignment, no-redef]
    tree_visibility: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    tree_tokens: torch.Tensor | None = None
    tree_depths: torch.Tensor | None = None
    tree_parents: torch.Tensor | None = None
    tree_first_child: torch.Tensor | None = None
    tree_next_sibling: torch.Tensor | None = None
    kv_cache_positions: torch.Tensor | None = None
    tree_acceptance: "TreeAcceptance | None" = None
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    is_dummy: bool = False
    # Set when the host mirror of device sequence lengths is exact.
    seq_lens_host_exact: bool = False

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


@triton.jit
def _prepare_tree_spec_pos_seq_lens_kernel(
    pos_ptr,
    kv_cache_positions_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    is_prefilling_ptr,
    num_computed_tokens_ptr,
    tree_depths_ptr,
    tree_depths_stride,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
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

    is_prefill = tl.load(is_prefilling_ptr + req_id).to(tl.int32)
    if is_prefill == 1:
        for i in tl.range(0, query_len, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < query_len
            linear = num_computed_tokens + block
            tl.store(pos_ptr + start + block, linear, mask=mask)
            tl.store(kv_cache_positions_ptr + start + block, linear, mask=mask)
        return

    # RoPE: root at num_computed; drafts use tree depth (siblings may share).
    # KV slots: unique token index so reshape_and_cache does not collide.
    tl.store(pos_ptr + start, num_computed_tokens)
    tl.store(kv_cache_positions_ptr + start, num_computed_tokens)
    for i in tl.range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        depth = tl.load(tree_depths_ptr + req_state_idx * tree_depths_stride + block - 1, mask=mask)
        rope_pos = depth + num_computed_tokens
        kv_cache_positions = num_computed_tokens + block
        tl.store(pos_ptr + start + block, rope_pos, mask=mask)
        tl.store(kv_cache_positions_ptr + start + block, kv_cache_positions, mask=mask)


def prepare_tree_spec_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    is_prefilling: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    tree_depths: torch.Tensor,
    pos: torch.Tensor,
    kv_cache_positions: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Fill RoPE ``pos`` and unique KV ``kv_cache_positions`` on device."""
    num_reqs = idx_mapping.shape[0]
    _prepare_tree_spec_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        kv_cache_positions,
        seq_lens,
        idx_mapping,
        query_start_loc,
        is_prefilling,
        num_computed_tokens,
        tree_depths,
        tree_depths.stride(0),
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )
