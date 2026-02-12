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
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator

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
    ):
        """Override GPU EagleSpeculator.propose for Ascend NPUs,
        because npu attention metadata needs more information,
        we need to cache input_batch, so we can use it later in
        generate_draft.
        """
        self.input_batch = input_batch
        # wrap build_attn_metadata to use Ascend attention metadata building.
        # so we can call super().propose() directly.
        with build_attn_metadata_wrapper():
            return super().propose(
                input_batch,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
            )

    def generate_draft(
        self,
        num_reqs: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp,
    ):
        """Override GPU EagleSpeculator.generate_draft for Ascend NPUs, because
        attn_metadata is created in super propose method, it does not have some
        attribute that Ascend attention backend needs, so we update it.
        """
        self._update_decode_attn_metadata(attn_metadata)

        return super().generate_draft(
            num_reqs,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
        )

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
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
        )

        # attn_metadata is None in profile_run and dummy_run.
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                # seq_lens in AscendMetadata is a cpu tensor.
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()
        return last_hidden_states, hidden_states

    def _update_decode_attn_metadata(
        self,
        attn_metadata: dict[str, Any],
    ):
        """Update attention metadata for decode phase on Ascend NPUs."""
        attn_state = AscendAttentionState.DecodeOnly
        seq_lens_cpu = self._get_seq_lens_cpu()
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for metadata in attn_metadata.values():
            metadata.attn_state = attn_state
            metadata.seq_lens_cpu = seq_lens_cpu

    def _get_seq_lens_cpu(self) -> torch.Tensor:
        """Get seq_lens_cpu from input_batch."""
        assert self.input_batch is not None
        seq_lens_cpu = torch.from_numpy(self.input_batch.seq_lens_np)
        return seq_lens_cpu


@contextmanager
def build_attn_metadata_wrapper():
    """Context manager to override attention metadata building for Ascend NPUs."""
    original_func = vllm.v1.worker.gpu.spec_decode.eagle.build_attn_metadata
    try:
        vllm.v1.worker.gpu.spec_decode.eagle.build_attn_metadata = build_attn_metadata
        yield
    finally:
        vllm.v1.worker.gpu.spec_decode.eagle.build_attn_metadata = original_func
