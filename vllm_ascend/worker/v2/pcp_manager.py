# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from copy import copy
from dataclasses import replace

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
)
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Allow only the graph-safe PCP decode path on Ascend."""
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            if (
                speculative_config.method != "mtp"
                or speculative_config.num_speculative_tokens != 1
            ):
                raise NotImplementedError(
                    "MRV2 PCP currently supports MTP with exactly one speculative "
                    "token only."
                )
        if cudagraph_mode.has_full_cudagraphs() or speculative_config is not None:
            vllm_config = copy(vllm_config)
        if cudagraph_mode.has_full_cudagraphs():
            if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        if speculative_config is not None:
            vllm_config.speculative_config = None

        PCPManager.validate_config(vllm_config, supports_mm_inputs)

    def set_mtp_enabled(self, enabled: bool) -> None:
        self._mtp_enabled = enabled

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        global_batch = input_batch
        has_draft_tokens = input_batch.num_draft_tokens > 0
        if has_draft_tokens:
            if not getattr(self, "_mtp_enabled", False):
                raise NotImplementedError(
                    "MRV2 PCP batch partition supports MTP1 draft tokens only."
                )
            assert input_batch.num_draft_tokens_per_req is not None
            if np.any(input_batch.num_draft_tokens_per_req > 1):
                raise NotImplementedError(
                    "MRV2 PCP batch partition supports MTP1 draft tokens only."
                )
            community_batch = replace(
                input_batch,
                num_draft_tokens=0,
                num_draft_tokens_per_req=None,
            )
        else:
            community_batch = input_batch

        local_batch = super().partition_batch(community_batch)
        self._global_batch = global_batch
        assert isinstance(local_batch, AscendInputBatch)
        if has_draft_tokens:
            local_batch = self._rebuild_local_mtp_fields(global_batch, local_batch)

        graph_num_tokens = input_batch.num_tokens_after_padding
        is_decode_only = not bool(input_batch.is_prefilling_np.any())
        # FULL_DECODE_ONLY graphs capture one token for every padded request.
        # Keep the request-shaped metadata at that same fixed graph extent.
        graph_num_reqs = graph_num_tokens if is_decode_only else input_batch.num_reqs_after_padding
        if is_decode_only and graph_num_tokens > local_batch.num_tokens_after_padding:
            assert self._input_buffers is not None
            input_buffers = self._input_buffers
            actual_tokens = local_batch.num_tokens
            actual_reqs = local_batch.num_reqs
            if graph_num_tokens > input_buffers.max_num_tokens:
                raise RuntimeError(
                    'PCP graph token count exceeds the local input buffer: '
                    f'{graph_num_tokens} > {input_buffers.max_num_tokens}.'
                )
            if graph_num_reqs > input_buffers.max_num_reqs:
                raise RuntimeError(
                    'PCP graph request count exceeds the local input buffer: '
                    f'{graph_num_reqs} > {input_buffers.max_num_reqs}.'
                )
            input_buffers.input_ids[actual_tokens:graph_num_tokens].zero_()
            input_buffers.positions[actual_tokens:graph_num_tokens].zero_()
            input_buffers.is_padding[actual_tokens:graph_num_tokens].fill_(True)
            input_buffers.seq_lens[actual_reqs:graph_num_reqs].zero_()
            input_buffers.query_start_loc[actual_reqs + 1 : graph_num_reqs + 1].fill_(
                actual_tokens
            )
            seq_lens_cpu_upper_bound = torch.zeros(
                graph_num_reqs,
                dtype=local_batch.seq_lens_cpu_upper_bound.dtype,
            )
            seq_lens_cpu_upper_bound[:actual_reqs].copy_(local_batch.seq_lens_cpu_upper_bound[:actual_reqs])
            local_batch = replace(  # type: ignore[call-arg]
                local_batch,
                num_reqs_after_padding=graph_num_reqs,
                num_tokens_after_padding=graph_num_tokens,
                query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
                seq_lens=input_buffers.seq_lens[:graph_num_reqs],
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                input_ids=input_buffers.input_ids[:graph_num_tokens],
                positions=input_buffers.positions[:graph_num_tokens],
                is_padding=input_buffers.is_padding[:graph_num_tokens],
            )

        local_batch.seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        return local_batch

    def _rebuild_local_mtp_fields(
        self,
        global_batch: AscendInputBatch,
        local_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        assert self._req_states is not None
        assert global_batch.num_draft_tokens_per_req is not None

        global_draft_counts = np.asarray(
            global_batch.num_draft_tokens_per_req, dtype=np.int32
        )
        draft_count_by_req_id = dict(
            zip(global_batch.req_ids, global_draft_counts.tolist(), strict=True)
        )
        local_draft_counts = np.fromiter(
            (draft_count_by_req_id[req_id] for req_id in local_batch.req_ids),
            dtype=np.int32,
            count=local_batch.num_reqs,
        )

        if local_batch.num_tokens == 0:
            local_num_logits = np.zeros(local_batch.num_reqs, dtype=np.int32)
            local_draft_counts.fill(0)
        else:
            local_num_logits = local_draft_counts + 1

        local_cu_num_logits_np = np.empty(local_batch.num_reqs + 1, dtype=np.int32)
        local_cu_num_logits_np[0] = 0
        np.cumsum(local_num_logits, out=local_cu_num_logits_np[1:])
        total_num_logits = int(local_cu_num_logits_np[-1])
        local_cu_num_logits = async_copy_to_gpu(
            local_cu_num_logits_np, device=self.device
        )

        expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
            local_batch.idx_mapping,
            total_num_logits,
            local_cu_num_logits,
            max_expand_len=2,
        )
        logits_indices = combine_sampled_and_draft_tokens(
            local_batch.input_ids,
            local_batch.idx_mapping,
            self._req_states.last_sampled_tokens,
            local_batch.query_start_loc,
            local_batch.seq_lens,
            self._req_states.prefill_len.gpu,
            self._req_states.draft_tokens,
            local_cu_num_logits,
            total_num_logits,
            1,
        )

        return replace(
            local_batch,
            num_draft_tokens=int(local_draft_counts.sum()),
            num_draft_tokens_per_req=local_draft_counts,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            logits_indices=logits_indices,
            cu_num_logits=local_cu_num_logits,
            cu_num_logits_np=local_cu_num_logits_np,
        )

    def prepare_attn(
        self, input_batch: AscendInputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if getattr(self, "_mtp_enabled", False):
            assert self._global_batch is not None
            assert self._block_tables is not None
            self._block_tables.gather_block_tables(
                self._global_batch.idx_mapping,
                self._global_batch.num_reqs,
            )
        return super().prepare_attn(input_batch)

    def prepare_slot_mappings(self) -> torch.Tensor:
        """Pad PCP slot mappings to the fixed FULL-decode graph layout."""
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        graph_num_tokens = self._global_batch.num_tokens_after_padding
        if getattr(self, "_mtp_enabled", False):
            assert isinstance(self._global_batch, AscendInputBatch)
            assert self._global_batch_slot_mappings is not None
            actual_num_tokens = self._global_batch.num_tokens
            self._global_batch_slot_mappings[
                :, actual_num_tokens:graph_num_tokens
            ].fill_(-1)
            self._global_batch.pcp_global_slot_mappings = (
                self._global_batch_slot_mappings[:, :graph_num_tokens]
            )
        is_decode_only = not bool(self._global_batch.is_prefilling_np.any())
        if not is_decode_only or graph_num_tokens <= self._global_batch.num_tokens:
            return slot_mappings

        assert self._gathered_kv_slot_mappings is not None
        graph_num_slots = graph_num_tokens * self.pcp_world_size
        self._gathered_kv_slot_mappings[:, slot_mappings.shape[1] : graph_num_slots].fill_(-1)
        return self._gathered_kv_slot_mappings[:, :graph_num_slots]
