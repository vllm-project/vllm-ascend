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
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator, gumbel_sample, update_eagle_inputs

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import vllm_version_is
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
        is_profile: Any = None,
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
            if vllm_version_is("0.19.0"):
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
            else:
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
                    is_profile=is_profile,
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
