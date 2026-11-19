# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek-V4 grouped output projection with an NPU module boundary."""

import torch
import torch_npu
from vllm.config import get_current_vllm_config

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.utils import oproj_tp_enable


class AscendDeepseekV4GroupedWoA(AscendColumnParallelLinear):
    """Project already inverse-rotated attention groups independently.

    The parameter keeps the Ascend BMM layout ``[group, input, output]``.
    LoRA-A is shared across groups; LoRA-B follows the output-row TP shard.
    """

    def __init__(self, *args, **kwargs) -> None:
        config = get_current_vllm_config()
        parallel_config = config.parallel_config
        if config.lora_config is not None and (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
            or oproj_tp_enable()
        ):
            raise NotImplementedError(
                "DeepSeek-V4 LoRA requires prefill_context_parallel_size=1, "
                "decode_context_parallel_size=1 and oproj_tensor_parallel_size=1; "
                "these token exchanges do not exchange adapter mappings."
            )
        super().__init__(*args, **kwargs)
        self.support_fp8_attention = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)

    def grouped_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.support_fp8_attention and getattr(self, "weight_scale", None) is not None:
            quantized, act_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            return torch_npu.npu_transpose_quant_batchmatmul(
                quantized,
                self.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=act_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
        return torch_npu.npu_transpose_batchmatmul(
            x,
            self.weight,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
