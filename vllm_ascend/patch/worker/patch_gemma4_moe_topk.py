#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import torch

try:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        MoERunner,
        SharedExpertsOrder,
        _unpack,
    )
except ImportError:
    MoERunner = None
    SharedExpertsOrder = None
    _unpack = None


def _is_precomputed_topk(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    topk_weights, topk_ids = value
    return isinstance(topk_weights, torch.Tensor) and isinstance(topk_ids, torch.Tensor)


if MoERunner is not None and not hasattr(MoERunner, "_ascend_dgemma_original_forward"):
    MoERunner._ascend_dgemma_original_apply_quant_method = MoERunner._apply_quant_method
    MoERunner._ascend_dgemma_original_forward = MoERunner.forward
    MoERunner._ascend_dgemma_original_forward_impl = MoERunner._forward_impl

    def _apply_quant_method(
        self: MoERunner,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not _is_precomputed_topk(router_logits):
            original = self.__class__._ascend_dgemma_original_apply_quant_method
            return original(self, hidden_states, router_logits, shared_experts_input, input_ids)

        assert SharedExpertsOrder is not None
        self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.NO_OVERLAP)

        assert not self.routed_experts.quant_method.is_monolithic
        topk_weights, topk_ids = router_logits
        fused_out = self.routed_experts.forward_modular(
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=self._shared_experts,
            shared_experts_input=shared_experts_input,
        )

        self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _forward(
        self: MoERunner,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _is_precomputed_topk(router_logits):
            original = self.__class__._ascend_dgemma_original_forward
            return original(self, hidden_states, router_logits, input_ids)

        hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        result = self._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )

        assert _unpack is not None
        shared_output, fused_output = _unpack(result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        shared_output = self._maybe_reduce_shared_expert_output(shared_output)
        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform)
        return self._maybe_add_zero_expert_output(result)

    def _forward_impl(
        self: MoERunner,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not _is_precomputed_topk(router_logits):
            original = self.__class__._ascend_dgemma_original_forward_impl
            return original(self, hidden_states, router_logits, shared_experts_input, input_ids)

        self.routed_experts._ensure_moe_quant_config_init()
        self._maybe_sync_shared_experts_stream(shared_experts_input)
        assert self.gate is None

        with self._sequence_parallel_context():
            assert not self.do_naive_dispatch_combine
            assert self.moe_config.pcp_size == 1

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    MoERunner._apply_quant_method = _apply_quant_method
    MoERunner.forward = _forward
    MoERunner._forward_impl = _forward_impl
