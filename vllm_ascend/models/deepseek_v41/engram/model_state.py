# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/model_state.py
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

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class EngramModelState(AscendModelState):
    """AscendModelState plus the engram lookback window and overlapped lookups.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start. The runner keeps the full token history on
    device, so the window is gathered there every step. Lookup rows are then
    produced on the model's auxiliary stream, overlapped with the forward:
    ``prepare_engram_inputs`` (with this step's ``cg_mode``) publishes
    persistent buffers plus ready events the model waits on (see
    ``DeepseekV41Model``).
    """

    _engram_graph_inputs: dict[str, torch.Tensor] | None = None

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        self._cg_mode: CUDAGraphMode | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full((self.max_num_reqs, depth), -1, dtype=torch.int32, device=device)
            if getattr(model, "supports_engram_graph_producer", False):
                self._engram_graph_inputs = {
                    # Separate from FIA's max_reqs+2 query buffer: its padding
                    # row must not become a real Engram request/history row.
                    "engram_query_start_loc": torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=device),
                    "engram_valid_token_count": torch.zeros(1, dtype=torch.int32, device=device),
                }

    def finish_execution(self, *, failed: bool) -> None:
        # Engram owns its lookup buffers and events. Keep their retirement
        # beside the input preparation instead of exposing them to the runner.
        try:
            self.model.retire_engram_lookups(reset_events=failed)
        finally:
            super().finish_execution(failed=failed)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        # prepare_inputs runs before any forward context, so the graph mode of
        # this step is only known here; engram overlap dispatch keys off it.
        self._cg_mode = cudagraph_mode
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
        )

    def prepare_engram_inputs(self, input_batch: AscendInputBatch, req_states) -> dict[str, Any]:
        model_inputs: dict[str, Any] = {}
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel

        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        if self._engram_graph_inputs is not None and self._cg_mode == CUDAGraphMode.FULL:
            graph_inputs = self._engram_graph_inputs
            query = graph_inputs["engram_query_start_loc"]
            valid_tokens = 0 if input_batch.is_dummy else input_batch.num_tokens
            query.fill_(valid_tokens)
            if not input_batch.is_dummy:
                query[: input_batch.num_reqs + 1].copy_(input_batch.query_start_loc[: input_batch.num_reqs + 1])
            graph_inputs["engram_valid_token_count"].fill_(valid_tokens)
            model_inputs.update(graph_inputs)
            # FULL replay runs its own producers. No Python-side hash/lookup
            # and no ExternalEvent records may precede it on this route.
            return model_inputs
        prepare_engram = getattr(self.model, "prepare_engram_inputs", None)
        if prepare_engram is not None:
            # Launch hash/lookup now so it overlaps the upcoming forward; the
            # returned events are consumed inside the model. A FULL replay
            # ignores the returned dict but still needs its side effects:
            # persistent buffers filled and bucket events recorded.
            model_inputs.update(
                prepare_engram(
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.num_tokens_after_padding,
                    window,
                    input_batch.query_start_loc,
                    cg_mode=self._cg_mode,
                    force_dummy=input_batch.is_dummy,
                )
            )
        return model_inputs

    def prepare_engram_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs: dict[str, Any] = {}
        window = self.lookback_token_ids
        if window is not None:
            # The captured graph reads this buffer; replays refill it in place.
            window.fill_(-1)
            model_inputs["lookback_token_ids"] = window
            if self._engram_graph_inputs is not None:
                for buffer in self._engram_graph_inputs.values():
                    buffer.zero_()
                model_inputs.update(self._engram_graph_inputs)
                return model_inputs
            prime_engram = getattr(self.model, "prime_engram_v2_graph_inputs", None)
            if prime_engram is not None:
                model_inputs.update(prime_engram(num_tokens))
        return model_inputs
