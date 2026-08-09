#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.utils import split_tensor_along_first_dim
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter


def _prepare_hash_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Align token IDs with rows produced by the active MoE prepare path."""
    forward_context = get_forward_context()
    prepared_ids = getattr(forward_context, "input_ids", input_ids).to(torch.int64)
    if forward_context.moe_comm_type == MoECommType.ALLGATHER:
        prepare_finalize = forward_context.moe_comm_method.prepare_finalize
        prepared_ids = prepare_finalize.all_gather_input_id_with_dp_group(prepared_ids)
    else:
        prepared_ids = forward_context.moe_comm_method.pad_and_split_input_ids(prepared_ids)

    if forward_context.flash_comm_v1_enabled and forward_context.moe_comm_type != MoECommType.ALLGATHER:
        tp_group = get_tp_group()
        prepared_ids = split_tensor_along_first_dim(prepared_ids, num_partitions=tp_group.world_size)[
            tp_group.rank_in_group
        ].contiguous()
    return prepared_ids


class AscendGroupedTopKRouter310(AscendGroupedTopKRouter):
    """310P router with chunked softmax top-k routing."""

    MAX_TOKENS_PER_GATING_CALL = 1024

    def __init__(self, *args, tid2eid: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tid2eid = tid2eid

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tid2eid is not None:
            if input_ids is None:
                raise ValueError("Hash-routed MoE requires current input_ids.")

            token_ids = _prepare_hash_input_ids(input_ids).reshape(-1)
            if token_ids.numel() != router_logits.shape[0]:
                raise ValueError(
                    f"Hash-routed input_ids and router rows differ: {token_ids.numel()} vs {router_logits.shape[0]}."
                )
            token_ids = torch.where(token_ids < 0, torch.zeros_like(token_ids), token_ids)
            topk_ids = self.tid2eid.index_select(0, token_ids).to(torch.int32)
            if topk_ids.shape[-1] != self.top_k:
                raise ValueError(f"Hash table returns {topk_ids.shape[-1]} experts, expected top_k={self.top_k}.")

            if self.scoring_func == "softmax":
                scores = router_logits.softmax(dim=-1)
            elif self.scoring_func == "sigmoid":
                scores = router_logits.sigmoid()
            elif self.scoring_func == "sqrtsoftplus":
                scores = F.softplus(router_logits).sqrt()
            else:
                raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

            topk_weights = scores.gather(1, topk_ids.to(torch.int64))
            topk_weights = self._renormalize_topk_weights(topk_weights)
            topk_weights = topk_weights * self.routed_scaling_factor
            return topk_weights.to(hidden_states.dtype), topk_ids

        if self.scoring_func != "softmax" or self.use_grouped_topk or self.e_score_correction_bias is not None:
            return super()._compute_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                indices_type=indices_type,
                input_ids=input_ids,
            )

        if router_logits.shape[0] > self.MAX_TOKENS_PER_GATING_CALL:
            topk_results = [
                torch_npu.npu_moe_gating_top_k_softmax(router_logits_chunk, k=self.top_k)
                for router_logits_chunk in router_logits.split(self.MAX_TOKENS_PER_GATING_CALL, dim=0)
            ]
            topk_weights = torch.cat([result[0] for result in topk_results], dim=0)
            topk_ids = torch.cat([result[1] for result in topk_results], dim=0)
        else:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=self.top_k)

        topk_weights = self._renormalize_topk_weights(topk_weights)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids.to(torch.int32)
