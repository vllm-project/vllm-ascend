# SPDX-License-Identifier: Apache-2.0
"""MoE quantization extension owned entirely by expert offload."""

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

from .config import ExpertOffloadConfig
from .expert_offload_manager import LayerOffloadBinding
from .quantization import MXFP4ExpertAdapter


class ExpertOffloadMoEMethod(AscendFusedMoEMethod):
    """Reuse native weight creation and execution with bounded expert storage."""

    def __init__(self, method, moe_config, config: ExpertOffloadConfig) -> None:
        if not isinstance(method, AscendFusedMoEMethod):
            raise ValueError(
                "selected quantization method does not support expert offload; "
                "W4A8 MXFP quantized routed experts are required"
            )
        if method.quant_method.quant_type != QuantType.W4A8MXFP:
            raise ValueError(
                "expert offload requires W4A8 MXFP routed expert weights; no adapter exists for this format"
            )
        MXFP4ExpertAdapter.validate_scheme(method.quant_method)
        super().__init__(method.quant_method, moe_config, tid2eid=method.tid2eid)
        self.expert_offload_config = config
        # Prevent the loader from moving the complete CPU layer to the NPU.
        self.requires_device_loading = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.expert_offload_config.validate_layer(num_experts, layer.top_k)
        with torch.device("cpu"):
            super().create_weights(
                layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
            )
        layer._expert_offload_binding = LayerOffloadBinding(layer, self.quant_method, self.expert_offload_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer._expert_offload_binding.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_comm = _EXTRA_CTX.moe_comm_method
        try:
            execution_layer, physical_ids, execution_comm = layer._expert_offload_binding.prepare(layer, topk_ids)
            _EXTRA_CTX.moe_comm_method = execution_comm
            return super().apply(
                layer=execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=physical_ids,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        finally:
            _EXTRA_CTX.moe_comm_method = original_comm
