# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
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
from contextlib import contextmanager
from typing import Any

import torch
import vllm
from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator, gumbel_sample, update_eagle_inputs
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class AscendEagleSpeculator(EagleSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Override GPU EagleSpeculator.__init__ for Ascend NPUs.
        attnention metadata building in Ascend backend needs more information,
        such as seq_lens_cpu from input_batch, so we need to override __init__.
        """
        super().__init__(vllm_config, device)
        # when in decode phase of eagle speculator, we need some value in
        # main model's input_batch. so we keep a reference here.
        self.input_batch: InputBatch | None = None

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Override GPU EagleSpeculator.propose for Ascend NPUs,
        because npu attention metadata needs more information,
        we need to cache input_batch, so we can use it later in
        generate_draft.
        """
        self.input_batch = input_batch
        # wrap build_attn_metadata to use Ascend attention metadata building.
        # so we can call super().propose() directly.
        with build_attn_metadata_wrapper(), torch_gather_wrapper():
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
            )

    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ):
        """Override GPU EagleSpeculator.generate_draft for Ascend NPUs, because
        attn_metadata is created in super propose method, it does not have some
        attribute that Ascend attention backend needs, so we update it.
        """
        self._update_decode_attn_metadata(attn_metadata, num_reqs)

        # NOTE(drslark): following lines (from 145 to 184) come from raw gpu's generate_draft logic
        pos = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            # Run the eagle model.
            last_hidden_states, hidden_states = self.run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            last_hidden_states = last_hidden_states[:num_reqs]
            hidden_states = hidden_states[:num_reqs]
            logits = self.model.compute_logits(last_hidden_states)

            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            draft_tokens = gumbel_sample(
                logits,
                idx_mapping,
                self.temperature,
                self.seeds,
                pos + 1,
                apply_temperature=True,
                processed_logits_out=self.draft_logits[:, step] if self.draft_logits is not None else None,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                # Update the inputs for the next step.
                update_eagle_inputs(
                    draft_tokens,
                    hidden_states,
                    self.input_buffers,
                    self.hidden_states,
                    self.max_model_len,
                )
                if attn_metadata is not None:
                    self.block_tables.compute_slot_mappings(idx_mapping, query_start_loc, pos, num_tokens_padded)

                    # npu's own update logic
                    self._update_decode_attn_metadata(attn_metadata, num_reqs)

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override GPU EagleSpeculator.run_model for Ascend NPUs, because
        in decode phase, we need to update seq_lens_cpu in attn_metadata after
        run model.
        """
        last_hidden_states, hidden_states = super().run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

        # attn_metadata is None in profile_run and dummy_run.
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                # seq_lens in AscendMetadata is a cpu tensor.
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()
        return last_hidden_states, hidden_states

    def _update_decode_attn_metadata(self, attn_metadata: dict[str, Any], num_reqs: int):
        """Update attention metadata for decode phase on Ascend NPUs."""
        if attn_metadata is None:
            return

        attn_state = AscendAttentionState.DecodeOnly
        seq_lens_cpu = self._get_seq_lens_cpu()

        # NOTE(drslark) to achieve fully alignment with vllm, `num_rejected` should be subtracted from `seq_lens`
        # to avoid extra sync overhead, `v2` is currently aligned with NPU `v1` only

        # follows the logic in `prepare_eagle_decode` and `update_eagle_inputs`
        seq_lens_cpu[:num_reqs] = torch.clamp(seq_lens_cpu[:num_reqs] + 1, max=self.max_model_len)
        seq_lens_cpu[num_reqs:].fill_(0)

        seq_lens_list = seq_lens_cpu.tolist()
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for metadata in attn_metadata.values():
            metadata.attn_state = attn_state
            metadata.seq_lens_cpu = seq_lens_cpu
            metadata.seq_lens_list = seq_lens_list

    def _get_seq_lens_cpu(self) -> torch.Tensor:
        """Get seq_lens_cpu from input_batch."""
        assert self.input_batch is not None
        seq_lens_cpu = torch.from_numpy(self.input_batch.seq_lens_np)
        return seq_lens_cpu


@contextmanager
def build_attn_metadata_wrapper():
    """Context manager to override attention metadata building for Ascend NPUs."""
    original_func = vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata
    try:
        vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata = build_attn_metadata
        yield
    finally:
        vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata = original_func


# TODO Remove this patch when cann fix the gather bug.
# NOTE(Ronald1995): torch.gather will pollute the cache such as self.input_buffers.positions
# the bug is reported to huawei CANN team, but not fixed yet.
# NOTE(drslark): make a temporary patch only for `torch.gather`
_original_gather = torch.gather


def gather(input, dim, index, *, sparse_grad=False, out=None):
    if out is None:
        return _original_gather(input, dim, index, sparse_grad=sparse_grad)
    out[:] = _original_gather(input, dim, index, sparse_grad=sparse_grad)
    return out


@contextmanager
def torch_gather_wrapper():
    """Context manager to override torch.gather for Ascend NPUs."""
    original_gather = torch.gather
    try:
        torch.gather = gather
        yield
    finally:
        torch.gather = original_gather


@triton.jit
def _prepare_eagle_docode_kernel(
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    last_token_indices_ptr,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    hidden_size,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            block_float = block.to(tl.float32)
            q = tl.where(block_float < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # output hidden states -> input hidden states.
    src_idx = tl.load(last_token_indices_ptr + req_idx)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + src_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Compute position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
):
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hidden_states.shape[-1]

    _prepare_eagle_docode_kernel[(num_reqs + 1,)](
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_hidden_states,
        input_hidden_states.stride(0),
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        hidden_size,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=4096,
    )
