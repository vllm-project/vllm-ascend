# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoEAllGatherCombineMetadata, MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithAllGather

# Cache row indices only for small decode shapes to avoid holding large dynamic
# prefill buffers across variable sequence lengths.
_MAX_CACHED_ROW_INDICES = 4096


@dataclass(frozen=True, slots=True)
class MoEAllGatherCombineMetadata310(MoEAllGatherCombineMetadata):
    local_row_count: torch.Tensor | None = None


def _is_stream_capturing() -> bool:
    checker = getattr(torch.npu, "is_current_stream_capturing", None)
    return bool(checker()) if checker is not None else False


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._encoded_expert_maps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._row_indices_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_encoded_expert_map(self, expert_map: torch.Tensor) -> torch.Tensor:
        key = id(expert_map)
        cached = self._encoded_expert_maps.get(key)
        if cached is not None and cached[0] is expert_map:
            return cached[1]
        encoded = expert_map.masked_fill(expert_map < 0, self.num_experts_local)
        if not _is_stream_capturing():
            self._encoded_expert_maps[key] = (expert_map, encoded)
        return encoded

    def _get_row_indices(self, num_rows: int, device: torch.device) -> torch.Tensor:
        if num_rows >= _MAX_CACHED_ROW_INDICES:
            return torch.arange(num_rows, device=device, dtype=torch.int32)
        key = (num_rows, device)
        rows = self._row_indices_cache.get(key)
        if rows is None:
            rows = torch.arange(num_rows, device=device, dtype=torch.int32)
            if not _is_stream_capturing():
                self._row_indices_cache[key] = rows
        return rows

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        restore_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        expert_num = self.num_experts_local
        if expert_map is not None:
            # 310P only accepts the full routing range. Pre-encode remote routes
            # (-1) into a final slot once and omit that slot from grouped matmul's
            # cumulative counts.
            encoded_map = self._get_encoded_expert_map(expert_map)
            topk_ids = encoded_map.index_select(0, topk_ids.flatten()).view_as(topk_ids)
            expert_num += 1

        assert hidden_states.shape[-1] % 16 == 0, (
            f"The last dim of hidden_states {hidden_states.shape[-1]} should be aligned with 16."
        )
        sorted_hidden_states, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * self.top_k,
            expert_num=expert_num,
            drop_pad_mode=0,
            active_expert_range=[0, expert_num],
            quant_mode=-1,
            row_idx_type=0,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        local_row_count = None
        if expert_map is not None:
            expert_tokens = expert_tokens[:-1]
            local_row_count = expert_tokens[-1]

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens,
            group_list_type=0,  # cumulative counts
            combine_metadata=MoEAllGatherCombineMetadata310(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
                local_row_count=local_row_count,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        if combine_metadata.local_row_count is not None:
            # The grouped matmuls leave rows beyond the final group unwritten.
            # Keep the original unique permutation for npu_moe_token_unpermute
            # and zero out the unwritten remote rows in place.
            local_rows = combine_metadata.local_row_count
            num_rows = hidden_states.shape[0]
            rows = self._get_row_indices(num_rows, hidden_states.device)
            hidden_states.masked_fill_((rows >= local_rows).unsqueeze(1), 0)
        return super().token_combine(hidden_states, combine_metadata, bias)
