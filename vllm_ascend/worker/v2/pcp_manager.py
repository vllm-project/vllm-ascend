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

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
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
        """Validate the Ascend MRV2 MLA and GQA PCP implementations."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        if parallel_config.prefill_context_parallel_size <= 1:
            return

        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 does not support PCP and DCP simultaneously yet.")
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError("Ascend MRV2 PCP does not support encoder-decoder models yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("Ascend MRV2 PCP does not support LoRA yet.")

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        super().__init__(
            pcp_world_size,
            pcp_rank,
            device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        self.vllm_config = vllm_config

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        assert self.vllm_config is not None
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens
            - (local_batch.num_draft_tokens_per_req if local_batch.num_draft_tokens_per_req is not None else 0),
        )
        return local_batch

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
