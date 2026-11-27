# SPDX-License-Identifier: Apache-2.0
"""MoE quantization extension owned entirely by expert offload."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

from .expert_offload_manager import DecodeSplitPlan
from .expert_offload_utils import ExpertWeightView

if TYPE_CHECKING:
    from vllm_ascend.ascend_config import ExpertOffloadConfig
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    from .expert_offload_manager import LayerOffloadBinding


class ExpertOffloadMoEMethod(AscendFusedMoEMethod):
    """Reuse native weight creation and execution with bounded expert storage."""

    # Prevent the loader from moving the complete CPU layer to the NPU.
    requires_device_loading: bool = False

    def __init__(self, native_method, moe_config, expert_offload_config: ExpertOffloadConfig) -> None:
        if not isinstance(native_method, AscendFusedMoEMethod):
            raise ValueError(
                "selected quantization method does not support expert offload; "
                "a supported Ascend quantized routed-expert scheme is required"
            )
        if native_method.quant_method.quant_type not in (QuantType.W4A8MXFP, QuantType.W8A8MXFP):
            raise ValueError(
                f"expert offload supports only W4A8-MXFP and W8A8-MXFP; got {native_method.quant_method.quant_type}"
            )
        super().__init__(native_method.quant_method, moe_config)
        self.expert_offload_config = expert_offload_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer._expert_offload_binding.initialize_storage(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> FusedExpertsResult:
        original_comm = _EXTRA_CTX.moe_comm_method
        binding = layer._expert_offload_binding
        try:
            if binding.device.type == "npu" and torch.npu.is_current_stream_capturing():
                _EXTRA_CTX.moe_comm_method = binding.manager.get_comm(binding, layer.moe_config, binding.num_slots)
                if topk_ids.numel() > binding.num_slots:
                    return self._apply_decode_graph(binding, layer, x, topk_weights, topk_ids)
                execution_layer = ExpertWeightView(layer, binding.resident)
                physical_topk_ids = binding.prepare_graph(topk_ids)
            else:
                prepared = binding.prepare(layer, topk_ids)
                if isinstance(prepared, DecodeSplitPlan):
                    return self._apply_eager_split(binding, layer, x, topk_weights, topk_ids, prepared)
                execution_layer, physical_topk_ids, execution_comm = prepared
                _EXTRA_CTX.moe_comm_method = execution_comm
            return super().apply(
                layer=execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=physical_topk_ids,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        finally:
            _EXTRA_CTX.moe_comm_method = original_comm

    def _apply_resident_group(self, binding, layer, x, topk_weights, physical_ids, rows: slice) -> FusedExpertsResult:
        view = ExpertWeightView(layer, binding.resident)
        if layer.ascend_pertoken_scale is not None:
            view.ascend_pertoken_scale = layer.ascend_pertoken_scale[rows]
        if layer.ascend_mc2_mask is not None:
            view.ascend_mc2_mask = layer.ascend_mc2_mask[rows]
        return super().apply(view, x[rows], topk_weights[rows], physical_ids)

    def _apply_eager_split(
        self, binding, layer, x, topk_weights, topk_ids, plan: DecodeSplitPlan
    ) -> FusedExpertsResult:
        _EXTRA_CTX.moe_comm_method = binding.manager.get_comm(binding, layer.moe_config, binding.num_slots)
        first_rows, second_rows = slice(0, plan.cut), slice(plan.cut, None)

        binding.load_resident(plan.first_slots, plan.first_loads)
        first = self._apply_resident_group(
            binding, layer, x, topk_weights, binding.mapping.remap(topk_ids[first_rows]), first_rows
        )
        next_slots, next_loads = binding.plan_expert_loads(plan.second_experts)
        if next_loads:
            # The H2D copies run on the current stream and fence the first
            # group's use of resident weights and the pinned host map.
            binding.load_resident(next_slots, next_loads)
        second = self._apply_resident_group(
            binding, layer, x, topk_weights, binding.mapping.remap(topk_ids[second_rows]), second_rows
        )
        return _join_groups(first, second)

    def _apply_decode_graph(self, binding, layer, x, topk_weights, topk_ids) -> FusedExpertsResult:
        cut = x.shape[0] // 2
        cut_ids = cut * binding.top_k
        num_topk_ids = binding.stage_graph_routes(topk_ids)
        stream = torch.npu.current_stream(binding.device)

        conditional = binding.manager.conditional_graph
        # MXFP experts can return BF16 from an FP16 input. Only capture the
        # conditional branches when the parent output can use the input dtype.
        if conditional is None or x.dtype != torch.bfloat16:
            torch.npu._launch_host_func(stream, binding.update_first_graph_group, (num_topk_ids, cut_ids))
            first, second = self._apply_graph_groups(binding, layer, x, topk_weights, topk_ids, cut)
            return _join_groups(first, second)

        # A parent-owned buffer gives dependent kernels one fixed output address.
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        fit, overflow = conditional.register(binding, num_topk_ids, cut_ids)
        with conditional.capture_branch(fit):
            view = ExpertWeightView(layer, binding.resident)
            fit_result = super().apply(view, x, topk_weights, binding.mapping.remap(topk_ids))
            if fit_result.routed_out.dtype != output.dtype:
                raise RuntimeError("Conditional expert graph requires the routed output dtype to match its buffer")
            output.copy_(fit_result.routed_out)
            output_result = replace(fit_result, routed_out=output, expert_tokens=None)
            del fit_result
        with conditional.capture_branch(overflow):
            first, second = self._apply_graph_groups(binding, layer, x, topk_weights, topk_ids, cut)
            output[:cut].copy_(first.routed_out)
            output[cut:].copy_(second.routed_out)
            del first, second
        # Expert offload disables dynamic EPLB and shared-expert event overlap.
        return output_result

    def _apply_graph_groups(
        self,
        binding: LayerOffloadBinding,
        layer,
        x,
        topk_weights,
        topk_ids,
        cut: int,
    ) -> tuple[FusedExpertsResult, FusedExpertsResult]:
        first_rows, second_rows = slice(0, cut), slice(cut, None)
        first = self._apply_resident_group(
            binding, layer, x, topk_weights, binding.mapping.remap(topk_ids[first_rows]), first_rows
        )
        # The second callback runs after all first-group reads of resident
        # weights. It returns immediately if the first load covered both groups.
        torch.npu._launch_host_func(torch.npu.current_stream(binding.device), binding.update_second_graph_group, ())
        second = self._apply_resident_group(
            binding, layer, x, topk_weights, binding.mapping.remap(topk_ids[second_rows]), second_rows
        )
        return first, second


def _join_groups(first: FusedExpertsResult, second: FusedExpertsResult) -> FusedExpertsResult:
    return replace(
        second,
        routed_out=torch.cat((first.routed_out, second.routed_out), dim=0),
        before_dispatch_evt=first.before_dispatch_evt,
        shared_activation_overlap_start_evt=first.shared_activation_overlap_start_evt,
    )
