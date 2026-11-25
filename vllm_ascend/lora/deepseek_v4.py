# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA boundaries for the fused DeepSeek-V4 compressor and grouped wo_a."""

from typing import TYPE_CHECKING, cast

import torch
from vllm.lora.layers import DeepseekV4GroupedWoAWithLoRA, ReplicatedLinearWithLoRA

if TYPE_CHECKING:
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


class AscendCompressorLinearWithLoRA(ReplicatedLinearWithLoRA):
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer(x)
        # The compressor accumulates its base projection in FP32, whereas
        # Punica expands in the adapter dtype. Add the two only after expand.
        # DSA serializes compressor adapters, so all layers can reuse this
        # scratch after adding it to their independently owned FP32 output.
        punica = cast("PunicaWrapperNPU", self.punica_wrapper)
        delta = punica.get_lora_delta_buffer(
            output.shape[0], output.shape[1], self.lora_b_stacked[0].dtype, output.device
        )
        return output.add_(self._apply_lora_to_output(x, delta))

    @classmethod
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None) -> bool:
        return type(source_layer).__name__ == "AscendCompressorLinear"


class AscendDeepseekV4GroupedWoAWithLoRA(DeepseekV4GroupedWoAWithLoRA):
    """Reuse upstream adapter loading with Ascend's already inverse-rotated input."""

    lora_b_stacked: tuple[torch.Tensor, ...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The upstream grouped entry point also prepares the inverse RoPE.
        # Ascend DSA has already done that before dispatching this forward.
        output = self.base_layer.grouped_forward(x)
        # Reuse one pair of contiguous group buffers. Packing all groups at
        # once makes the temporary memory scale with the number of groups.
        single_group = len(self.lora_b_stacked) == 1
        if single_group:
            group_input = x[:, 0].contiguous()
            group_output = output[:, 0].contiguous()
        else:
            group_input = torch.empty((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
            group_output = torch.empty((output.shape[0], output.shape[-1]), dtype=output.dtype, device=output.device)
        for group, lora_b in enumerate(self.lora_b_stacked):
            if not single_group:
                group_input.copy_(x[:, group])
                group_output.copy_(output[:, group])
            self.punica_wrapper.add_lora_linear(
                group_output,
                group_input,
                self.lora_a_stacked,
                (lora_b,),
                1.0,
                (group_output.shape[-1],),
            )
            output[:, group].copy_(group_output)
        return output

    @classmethod
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None) -> bool:
        return type(source_layer).__name__ == "AscendDeepseekV4GroupedWoA"
