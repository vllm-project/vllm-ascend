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

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState, ReplayAttnMetadata


@triton.jit
def _pad_v2_replayed_slots_kernel(
    slot_mappings_ptr,
    group_stride,
    cacheable_groups_ptr,
    query_start_loc_ptr,
    positions_ptr,
    replay_start_ptr,
    replay_window,
    pad_slot_id,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(replay_start_ptr + req)
    if start <= 0:
        return
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    for tok in range(begin, end, BLOCK):
        offsets = tok + tl.arange(0, BLOCK)
        pos = tl.load(positions_ptr + offsets, mask=offsets < end, other=0)
        replayed = (offsets < end) & (pos >= start) & (pos < start + replay_window)
        for i in tl.static_range(NUM_GROUPS):
            group = tl.load(cacheable_groups_ptr + i)
            tl.store(
                slot_mappings_ptr + group * group_stride + offsets,
                pad_slot_id,
                mask=replayed,
            )


class EngramModelState(AscendModelState):
    """AscendModelState plus the engram lookback window and overlapped lookups.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start. The runner keeps the full token history on
    device, so the window is gathered there every step. Lookup rows are then
    produced inside the forward context on the model's auxiliary stream:
    ``prepare_engram_inputs`` (with this step's ``cg_mode``) publishes
    persistent buffers plus ready events the model waits on (see
    ``DeepseekV41Model``).
    """

    _engram_graph_inputs: dict[str, torch.Tensor] | None = None
    _engram_inputs: dict[str, Any] | None = None

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = CpuGpuBuffer(self.max_num_reqs, dtype=torch.int32, device=device, pin_memory=False)
        self._replay_groups: tuple[int, torch.Tensor] | None = None
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        self._cg_mode: CUDAGraphMode | None = None
        if depth > 0:
            model.register_forward_pre_hook(self._prepare_engram_before_forward, with_kwargs=True)
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full((self.max_num_reqs, depth), -1, dtype=torch.int32, device=device)
            if getattr(model, "supports_engram_graph_producer", False):
                self._engram_graph_inputs = {
                    # Separate from FIA's max_reqs+2 query buffer: its padding
                    # row must not become a real Engram request/history row.
                    "engram_query_start_loc": torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=device),
                    "engram_valid_token_count": torch.zeros(1, dtype=torch.int32, device=device),
                }

    def add_request(self, req_index: int, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        self._replay_start_np[req_index] = int(getattr(new_req_data, "replay_start", 0))

    def finish_execution(self, *, failed: bool) -> None:
        # Engram owns its lookup buffers and events. Keep their retirement
        # beside the input preparation instead of exposing them to the runner.
        try:
            self._engram_inputs = None
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
        model_specific_attn_metadata=None,
    ) -> dict[str, Any]:
        # prepare_inputs runs before any forward context, so the graph mode of
        # this step is only known here; engram overlap dispatch keys off it.
        self._cg_mode = cudagraph_mode
        if self._replay_groups is None:
            from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable

            specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
            windows = [int(getattr(spec, "prefix_replay_tokens", getattr(spec, "sliding_window", 0))) for spec in specs]
            self._replay_groups = (
                max(windows, default=0),
                torch.tensor(
                    [i for i, spec in enumerate(specs) if is_prefix_cacheable(spec)],
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
        replay_window, cacheable_groups = self._replay_groups
        num_reqs = input_batch.num_reqs
        replay_np = np.where(
            input_batch.is_prefilling_np[:num_reqs],
            self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
            0,
        ).astype(np.int32)
        self._replay_start.np[:num_reqs] = replay_np
        self._replay_start.copy_to_gpu(num_reqs)
        replay_start = self._replay_start.gpu[:num_reqs]
        if replay_window > 0 and replay_np.any() and cacheable_groups.numel() > 0:
            _pad_v2_replayed_slots_kernel[(num_reqs,)](
                slot_mappings,
                slot_mappings.stride(0),
                cacheable_groups,
                input_batch.query_start_loc,
                input_batch.positions,
                replay_start,
                replay_window,
                PAD_SLOT_ID,
                NUM_GROUPS=cacheable_groups.numel(),
                BLOCK=1024,
            )
        if model_specific_attn_metadata is None:
            model_specific_attn_metadata = ReplayAttnMetadata(replay_start)
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )

    def prepare_engram_inputs(self, input_batch: AscendInputBatch, req_states) -> dict[str, Any]:
        model_inputs: dict[str, Any] = {}
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        dummy = input_batch.is_dummy or self.kvpp_is_dummy_run or ring_state_update_skipped()
        batch = input_batch
        token_indices = None
        if self.pcp_context is not None and not dummy:
            # Hash contiguous requests; select local rows only after hashing.
            batch = self.pcp_context.global_batch
            token_indices = self.pcp_context.local_token_indices
        if dummy:
            # Idle ranks still participate in lookup, but have no history.
            window.fill_(-1)
        else:
            all_token_ids = req_states.all_token_ids.gpu
            depth = window.shape[1]
            from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel

            _gather_lookback_kernel[(window.shape[0],)](
                window,
                batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                batch.num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )
        model_inputs["lookback_token_ids"] = window
        if self._engram_graph_inputs is not None and self._cg_mode == CUDAGraphMode.FULL:
            graph_inputs = self._engram_graph_inputs
            query = graph_inputs["engram_query_start_loc"]
            valid_tokens = 0 if dummy else batch.num_tokens
            query.fill_(valid_tokens)
            if not dummy:
                query[: batch.num_reqs + 1].copy_(batch.query_start_loc[: batch.num_reqs + 1])
            graph_inputs["engram_valid_token_count"].fill_(valid_tokens)
            model_inputs.update(graph_inputs)
            # FULL replay runs its own producers. No Python-side hash/lookup
            # and no ExternalEvent records may precede it on this route.
            return model_inputs
        # TODO: Support breakable PIECEWISE by refreshing lookups under the
        # forward context before replay, which bypasses the model pre-hook.
        # Align capture bindings with that path so replay does not retain
        # dummy graph-producer coordinates.
        # DP lookup needs the forward context, which is not established yet.
        self._engram_inputs = {
            "input_ids": batch.input_ids[: batch.num_tokens],
            "positions": batch.positions[: batch.num_tokens],
            "padded_tokens": input_batch.num_tokens_after_padding,
            "lookback_token_ids": window,
            "query_start_loc": None if dummy else batch.query_start_loc[: batch.num_reqs + 1],
            "cg_mode": self._cg_mode,
            "force_dummy": dummy,
            "token_indices": token_indices,
        }
        model_inputs.update(self.model.prepare_engram_graph_inputs(input_batch.num_tokens_after_padding))
        return model_inputs

    def prepare_engram_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        self._engram_inputs = None
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

    def _prepare_engram_before_forward(self, model, args, kwargs) -> None:
        kwargs.update(self.prepare_engram())

    def prepare_engram(self) -> dict[str, Any]:
        """Refresh buffers once inside eager forward or before FULL replay."""
        if self._engram_inputs is None:
            return {}
        model_inputs = self.model.prepare_engram_inputs(**self._engram_inputs)
        self._engram_inputs = None
        return model_inputs
