# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/autoregressive/speculator.py
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
from copy import copy
from typing import Any

import torch
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


class AscendAutoRegressiveSpeculator(AutoRegressiveSpeculator):
    """
    Shared Ascend spec-decode loop for AscendEagle/AscendMTPSpeculator.

    GQA, MLA, and DSA draft decode state share one path. The current MTP path
    uses the draft attention backend recorded by ``set_attn``.

    MLA's per-step state lives in ``.decode`` (cloned per step, written via an
    alias), GQA's is top-level. MLA also rebuilds the base (live ``.decode`` is
    None/wrong-batch) and forwards rotary ``positions`` into
    build_attn_metadata. DSA manages its draft state in its metadata builder
    and skips the generic MLA/GQA init and update logic.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Override the upstream __init__ for Ascend NPUs.

        Ascend attention-metadata building needs more information (e.g.
        seq_lens_cpu from input_batch), so we replace input_buffers with
        AscendInputBuffers after super().__init__.
        """
        super().__init__(vllm_config, device)

        self.attn_backend: type[AttentionBackend] | None = None
        self.draft_vllm_config = self._create_draft_vllm_config()

        del self.input_buffers
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

        # when in decode phase of eagle speculator, we need some value in
        # draft model's input_batch. so we keep a reference here.
        self.input_batch: InputBatch | None = None

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Build the runtime config used while executing the draft model."""
        return replace(
            self.vllm_config,
            model_config=self.draft_model_config,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        assert self.prefill_cudagraph_manager is not None
        assert self.decode_cudagraph_manager is not None
        # The Ascend graph managers are patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # They need this speculator to update full-graph params, so set it here.
        self.prefill_cudagraph_manager.speculator = self
        self.decode_cudagraph_manager.speculator = self
        self.prefill_cudagraph_manager.update_stream = self.update_stream
        self.decode_cudagraph_manager.update_stream = self.update_stream

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

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        assert skip_attn or batch_desc.cg_mode == CUDAGraphMode.FULL, (
            "Ascend fused draft decode requires a captured FULL graph, but "
            f"got {batch_desc.cg_mode.name}. Ensure cudagraph_capture_sizes "
            "covers this draft decode batch size."
        )
        super()._fused_multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
        )

    def build_fia_params(
        self,
        num_reqs_padded: int,
        is_draft_model_prefill: bool,
    ) -> list[dict[str, Any]]:
        if is_draft_model_prefill:
            metadata = next(
                metadata
                for layer_name, metadata in self.model_state.attn_metadata.items()
                if layer_name in self.draft_attn_layer_names
            )
            return [
                {
                    "actual_seq_lengths": metadata.query_start_loc,
                    "actual_seq_lengths_kv": metadata.seq_lens,
                }
            ]

        assert self.input_batch is not None
        num_reqs = self.input_batch.num_reqs
        query_start_loc = list(range(1, num_reqs_padded + 1))
        fia_params: list[dict[str, Any]] = []
        for step in range(1, self.num_speculative_steps):
            seq_lens = [
                min(int(seq_len) + step, self.max_model_len)
                for seq_len in self.input_batch.seq_lens_np[:num_reqs]
            ]
            seq_lens.extend([0] * (num_reqs_padded - num_reqs))
            fia_params.append(
                {
                    "actual_seq_lengths": query_start_loc,
                    "actual_seq_lengths_kv": seq_lens,
                }
            )
        return fia_params
 

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
