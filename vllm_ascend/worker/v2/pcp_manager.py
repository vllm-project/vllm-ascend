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
from dataclasses import dataclass, replace

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


@dataclass(frozen=True)
class AscendPCPAttentionContext:
    """Canonical global PCP view for one attention step."""

    # The global batch and its associated metadata, used to build DSA attention metadata.
    global_batch: AscendInputBatch
    global_block_tables: tuple[torch.Tensor, ...]
    global_slot_mappings: torch.Tensor
    hidden_restore_idx: torch.Tensor
    local_num_tokens_after_padding: int


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.has_full_cudagraphs():
            if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        PCPManager.validate_config(vllm_config, supports_mm_inputs)

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        # PCP builds the local layout from actual tokens, but a FULL decode
        # graph replays a fixed padded layout on every rank.
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
                    "PCP graph token count exceeds the local input buffer: "
                    f"{graph_num_tokens} > {input_buffers.max_num_tokens}."
                )
            if graph_num_reqs > input_buffers.max_num_reqs:
                raise RuntimeError(
                    "PCP graph request count exceeds the local input buffer: "
                    f"{graph_num_reqs} > {input_buffers.max_num_reqs}."
                )
            input_buffers.input_ids[actual_tokens:graph_num_tokens].zero_()
            input_buffers.positions[actual_tokens:graph_num_tokens].zero_()
            input_buffers.is_padding[actual_tokens:graph_num_tokens].fill_(True)
            input_buffers.seq_lens[actual_reqs:graph_num_reqs].zero_()
            input_buffers.query_start_loc[actual_reqs + 1 : graph_num_reqs + 1].fill_(actual_tokens)
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

    def prepare_slot_mappings(self) -> torch.Tensor:
        """Pad PCP slot mappings to the fixed FULL-decode graph layout."""
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        graph_num_tokens = self._global_batch.num_tokens_after_padding
        is_decode_only = not bool(self._global_batch.is_prefilling_np.any())
        if not is_decode_only or graph_num_tokens <= self._global_batch.num_tokens:
            return slot_mappings

        assert self._gathered_kv_slot_mappings is not None
        graph_num_slots = graph_num_tokens * self.pcp_world_size
        self._gathered_kv_slot_mappings[:, slot_mappings.shape[1] : graph_num_slots].fill_(-1)
        return self._gathered_kv_slot_mappings[:, :graph_num_slots]

    def build_attention_context(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> AscendPCPAttentionContext:
        """Build the PCP context consumed by attention metadata builders."""
        if input_batch.is_dummy:
            local_num_tokens_after_padding = input_batch.num_tokens
            restore_start = self.pcp_rank * local_num_tokens_after_padding
            return AscendPCPAttentionContext(
                global_batch=input_batch,
                global_block_tables=block_tables,
                global_slot_mappings=slot_mappings.view(
                    slot_mappings.shape[0],
                    self.pcp_world_size,
                    local_num_tokens_after_padding,
                )[:, self.pcp_rank],
                hidden_restore_idx=torch.arange(
                    restore_start,
                    restore_start + local_num_tokens_after_padding,
                    device=self.device,
                ),
                local_num_tokens_after_padding=local_num_tokens_after_padding,
            )

        global_batch = self._global_batch
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.gather_block_tables(
                global_batch.idx_mapping,
                global_batch.num_reqs_after_padding,
            ),
            global_slot_mappings=self._global_batch_slot_mappings[:, : global_batch.num_tokens],
            hidden_restore_idx=self._hidden_restore_idx,
            local_num_tokens_after_padding=input_batch.num_tokens_after_padding,
        )
