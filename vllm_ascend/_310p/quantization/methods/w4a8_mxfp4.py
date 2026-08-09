# SPDX-License-Identifier: Apache-2.0
"""DeepSeek MXFP4 checkpoint adapter for Ascend 310P W8A8 execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend._310p.quantization.methods.mxfp4_to_w8a8 import requantize_mxfp4_to_int8
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310
from vllm_ascend.utils import maybe_trans_nz


class AscendMXFP4ToW8A8DynamicFusedMoEMethod310(AscendW8A8DynamicFusedMoEMethod310):
    """Load DeepSeek packed MXFP4 experts and execute through 310P W8A8 MoE.

    The checkpoint-facing parameter shapes remain packed MXFP4. After loading,
    each rank converts only its local expert shard to symmetric per-row INT8.
    """

    group_size = 32

    def __init__(self, quant_config: dict[str, Any], tid2eid=None):
        super().__init__()
        configured_group_size = quant_config.get("group_size", self.group_size)
        if configured_group_size != self.group_size:
            raise ValueError(
                f"DeepSeek V4 MXFP4 on 310P requires group_size={self.group_size}, got {configured_group_size}."
            )
        self.tid2eid = tid2eid

    @staticmethod
    def get_weight(
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // 2,
                dtype=torch.uint8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.float8_e8m0fnu,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e8m0fnu,
            ),
        }

    def process_weights_after_loading(self, layer) -> None:
        logger.info_once(
            "Converting local DeepSeek V4 MXFP4 expert shard to 310P W8A8: w13=%s, w2=%s, experts=%d.",
            tuple(layer.w13_weight.shape),
            tuple(layer.w2_weight.shape),
            layer.w13_weight.shape[0],
        )
        w13_weight, w13_scale = requantize_mxfp4_to_int8(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2_weight, w2_scale = requantize_mxfp4_to_int8(layer.w2_weight.data, layer.w2_weight_scale.data)

        layer.w13_weight.data = maybe_trans_nz(w13_weight)
        layer.w2_weight.data = maybe_trans_nz(w2_weight)
        layer.w13_weight_scale.data = w13_scale.view(w13_scale.shape[0], -1)
        layer.w2_weight_scale.data = w2_scale.view(w2_scale.shape[0], -1)
