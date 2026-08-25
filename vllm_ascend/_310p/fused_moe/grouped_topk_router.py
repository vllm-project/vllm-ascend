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
import torch_npu

from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.utils import vllm_version_is


class AscendGroupedTopKRouter310(AscendGroupedTopKRouter):
    """310P router with chunked softmax top-k routing."""

    MAX_TOKENS_PER_GATING_CALL = 1024

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func != "softmax" or self.use_grouped_topk or self.e_score_correction_bias is not None:
            return super()._compute_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                indices_type=indices_type,
                input_ids=input_ids,
            )

        if not vllm_version_is("0.27.1"):
            # vLLM main recomputes router_logits as fp32 in _forward_impl,
            # but the resulting tensor may be in FRACTAL_NZ format on 310P.
            # npu_moe_gating_top_k_softmax only supports DT_FLOAT16 + ND,
            # so cast the router_logits accordingly.
            router_logits = router_logits.to(torch.float16)

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
