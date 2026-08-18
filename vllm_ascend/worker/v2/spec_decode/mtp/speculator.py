# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator with a global draft view for PCP."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._draft_attn_vllm_config = self.vllm_config
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            draft_parallel_config = replace(
                vllm_config.parallel_config,
                prefill_context_parallel_size=1,
            )
            self._draft_attn_vllm_config = replace(
                vllm_config,
                parallel_config=draft_parallel_config,
            )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return self._draft_attn_vllm_config

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self._draft_attn_vllm_config)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
            assert isinstance(input_batch, AscendInputBatch)
            attn_metadata, slot_mappings = self._build_global_pcp_draft_inputs(
                input_batch
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            *args,
            **kwargs,
        )

    def _build_global_pcp_draft_inputs(
        self,
        input_batch: AscendInputBatch,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        block_tables = tuple(
            table[:num_reqs] for table in self.block_tables.input_block_tables
        )
        slot_mappings = input_batch.pcp_global_slot_mappings
        assert slot_mappings is not None

        max_query_len = int(input_batch.num_scheduled_tokens.max())
        max_seq_len = int(
            input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        )
        global_attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens[:num_reqs],
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=input_batch.seq_lens_np,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            positions=input_batch.positions[:num_tokens],
            attn_state=input_batch.attn_state,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
        )
        slot_mappings_by_layer = {
            layer_name: slot_mappings[group_idx]
            for group_idx, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups
            )
            for layer_name in kv_cache_group.layer_names
            if layer_name in self.draft_attn_layer_names
        }
        return global_attn_metadata, slot_mappings_by_layer
