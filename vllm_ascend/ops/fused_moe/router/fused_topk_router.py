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
from collections.abc import Callable

import torch
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter

DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID = 129257
DEEPSEEK_V4_IMAGE_SENTINEL_COUNT = 5


def select_deepseek_v4_vision_experts(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor | None,
    bias_vl: torch.Tensor,
    text_bias: torch.Tensor | None,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    image_sentinel_lo: int = DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select text experts and apply the vision route to image rows.

    DeepSeek-V4 vision checkpoints borrow five consecutive in-vocabulary
    sentinel ids for IMAGE_START..IMAGE_END. Text rows retain the deterministic
    ``tid2eid`` lookup used by the text-only model, while image rows use the
    checkpoint's ``bias_vl`` with the sqrt-softplus router scores.
    """
    scores = torch.nn.functional.softplus(router_logits).sqrt()
    image_hi = image_sentinel_lo + DEEPSEEK_V4_IMAGE_SENTINEL_COUNT
    image_mask = (input_ids >= image_sentinel_lo) & (input_ids < image_hi)
    row_bias = torch.where(
        image_mask.unsqueeze(-1),
        bias_vl.to(scores.dtype).unsqueeze(0),
        (text_bias.to(scores.dtype).unsqueeze(0) if text_bias is not None else torch.zeros_like(scores)),
    )
    dynamic_ids = torch.topk(
        scores + row_bias,
        k=top_k,
        dim=-1,
        sorted=True,
    ).indices
    if tid2eid is None:
        topk_ids = dynamic_ids
    else:
        lookup_ids = torch.where(image_mask, 0, input_ids)
        text_ids = tid2eid[lookup_ids].to(torch.int64)
        topk_ids = torch.where(image_mask.unsqueeze(-1), dynamic_ids, text_ids)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


class AscendFusedTopKRouter(AscendGroupedTopKRouter):
    """Router adapter that uses Ascend's existing expert-selection path."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        eplb_state: EplbLayerState | None = None,
        num_logical_experts: int | None = None,
        tid2eid: torch.Tensor | None = None,
        bias_vl: torch.Tensor | None = None,
        image_sentinel_lo: int = DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID,
        select_experts_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_logical_experts = num_logical_experts if num_logical_experts is not None else global_num_experts
        self.tid2eid = tid2eid
        self.bias_vl = bias_vl
        self.image_sentinel_lo = image_sentinel_lo

    def is_fused_supported(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        topk_group = self.topk_group if self.topk_group is not None else 1
        num_expert_group = self.num_expert_group if self.num_expert_group is not None else 1
        if not (
            num_expert_group > 0
            and hidden_states.shape[-1] % num_expert_group == 0
            and hidden_states.shape[-1] // num_expert_group > 2
        ):
            return False
        if self.top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
            return False
        if topk_group * hidden_states.shape[-1] / num_expert_group < self.top_k:  # noqa: SIM103
            return False
        return True

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bias_vl is None and not self.is_fused_supported(hidden_states):
            return super()._compute_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                indices_type=indices_type,
                input_ids=input_ids,
            )

        topk_group = self.topk_group if self.topk_group is not None else 1
        num_expert_group = self.num_expert_group if self.num_expert_group is not None else 1
        renorm = int(self.renormalize)
        if self.scoring_func == "sqrtsoftplus":
            if self.tid2eid is not None or self.bias_vl is not None:
                if input_ids is None:
                    raise ValueError("DeepSeek V4 vision/hash MoE routing requires input_ids.")
                input_ids = input_ids.to(torch.int64)
                tid2eid_ones = self.tid2eid.to(torch.int32) if self.tid2eid is not None else None
                if _EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER:
                    prepare_finalize = _EXTRA_CTX.moe_comm_method.prepare_finalize
                    input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
                else:
                    input_ids = _EXTRA_CTX.moe_comm_method.pad_and_split_input_ids(input_ids)
                if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER and input_ids.numel() != router_logits.shape[0]:
                    # Native MoE SP chunks hidden states before MC2/All2All,
                    # while their replace-allreduce paths retain full token
                    # ids. Apply the identical TP chunk only when communication
                    # has not already aligned ids with local router rows.
                    input_ids = sequence_parallel_chunk(input_ids.reshape(-1, 1)).reshape(-1)
                input_ids = torch.where(input_ids == -1, 0, input_ids)
            else:
                input_ids = None
                tid2eid_ones = None
            if self.bias_vl is not None and input_ids is not None:
                topk_weights, topk_ids = select_deepseek_v4_vision_experts(
                    router_logits=router_logits,
                    input_ids=input_ids,
                    tid2eid=tid2eid_ones,
                    bias_vl=self.bias_vl,
                    text_bias=self.e_score_correction_bias,
                    top_k=self.top_k,
                    renormalize=self.renormalize,
                    routed_scaling_factor=self.routed_scaling_factor,
                    image_sentinel_lo=self.image_sentinel_lo,
                )
                return topk_weights.to(torch.float32), topk_ids.to(
                    torch.int32 if indices_type is None else indices_type
                )
            topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
                x=router_logits,
                k=self.top_k,
                bias=self.e_score_correction_bias,
                input_ids=input_ids,
                tid2eid=tid2eid_ones,
                k_group=topk_group,
                group_count=num_expert_group,
                routed_scaling_factor=self.routed_scaling_factor,
                eps=1e-20,
                group_select_mode=1,
                # The hash custom op currently rejects renorm != 0. Apply
                # norm_topk_prob in Python below before returning to MoE compute.
                renorm=0,
                norm_type=2,
                out_flag=False,
            )
            return topk_weights, topk_ids
        norm_type = 0 if self.scoring_func == "softmax" else 1
        if self.e_score_correction_bias is not None and self.e_score_correction_bias.dtype != router_logits.dtype:
            self.e_score_correction_bias = self.e_score_correction_bias.to(router_logits.dtype)
        topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
            router_logits,
            k=self.top_k,
            k_group=topk_group,
            group_count=num_expert_group,
            group_select_mode=1,
            renorm=renorm,
            norm_type=norm_type,  # 0: softmax; 1: sigmoid
            out_flag=False,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
            bias_opt=self.e_score_correction_bias,
        )

        return topk_weights, topk_ids
