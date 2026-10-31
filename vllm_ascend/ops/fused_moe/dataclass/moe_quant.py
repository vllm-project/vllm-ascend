#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.quantization.quant_type import QuantType


@dataclass(frozen=True, slots=True)
class MoEMxfpParams:
    """Internal MXFP-only precision settings used by fused_moe runtime."""

    act_quant_type: torch.dtype | None = None
    weight_quant_type: torch.dtype | None = None
    scale_dtype: torch.dtype | None = None
    per_token_scale_dtype: torch.dtype | None = None
    use_bf16: bool = True


@dataclass(frozen=True, slots=True)
class MoEQuantParams:
    """Quant mode, backend override, and optional internal MXFP leaf config."""

    quant_type: QuantType = QuantType.NONE
    comm_quant_mode: int | None = None
    mxfp: MoEMxfpParams | None = None
    is_per_channel_weight: bool = False

    @property
    def is_quant(self) -> bool:
        return self.quant_type != QuantType.NONE

    @property
    def is_mxfp(self) -> bool:
        return self.quant_type in (QuantType.W8A8MXFP, QuantType.W4A4MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP)

    @property
    def is_w4a4_mxfp(self) -> bool:
        return self.quant_type == QuantType.W4A4MXFP

    @property
    def is_int_quant(self) -> bool:
        return self.quant_type in (QuantType.W8A8, QuantType.W4A8)

    @property
    def is_fp8(self) -> bool:
        return self.quant_type == QuantType.W8A8FP

    @property
    def use_w4a8_per_channel_gmm_swiglu(self) -> bool:
        return self.quant_type == QuantType.W4A8 and self.is_per_channel_weight

    @property
    def dispatch_with_quant(self) -> bool:
        return self.quant_type in (
            QuantType.W8A8,
            QuantType.W4A8,
            QuantType.W8A8MXFP,
            QuantType.W4A4MXFP,
            QuantType.W4A8MXFP,
            QuantType.W8A8FP,
        )

    @property
    def get_dst_type(self):
        if self.is_w4a4_mxfp:
            return torch_npu.float4_e2m1fn_x2
        elif self.is_mxfp or self.is_fp8:
            return torch.float8_e4m3fn
        elif self.dispatch_with_quant:
            return torch.int8
        else:
            return None

    @property
    def get_scale_type(self):
        if self.is_mxfp:
            return torch.float8_e8m0fnu
        elif self.dispatch_with_quant:
            return torch.float32
        else:
            return None


def _build_mxfp_params(
    *,
    quant_type: QuantType,
    mxfp_act_quant_type: torch.dtype | None = None,
    mxfp_weight_quant_type: torch.dtype | None = None,
    mxfp_scale_dtype: torch.dtype | None = None,
    mxfp_per_token_scale_dtype: torch.dtype | None = None,
    mxfp_use_bf16: bool | None = None,
) -> MoEMxfpParams | None:
    if quant_type not in [QuantType.W8A8MXFP, QuantType.W4A4MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP]:
        return None

    has_explicit_mxfp_args = any(
        value is not None
        for value in (
            mxfp_act_quant_type,
            mxfp_weight_quant_type,
            mxfp_scale_dtype,
            mxfp_per_token_scale_dtype,
            mxfp_use_bf16,
        )
    )
    if not has_explicit_mxfp_args:
        raise ValueError("primitive MXFP params are required when quant_type is an MXFP quant type.")

    return MoEMxfpParams(
        act_quant_type=mxfp_act_quant_type,
        weight_quant_type=mxfp_weight_quant_type,
        scale_dtype=mxfp_scale_dtype,
        per_token_scale_dtype=mxfp_per_token_scale_dtype,
        use_bf16=True if mxfp_use_bf16 is None else mxfp_use_bf16,
    )


def build_quant_params(
    quant_type,
    comm_quant_mode,
    mxfp_act_quant_type,
    mxfp_weight_quant_type,
    mxfp_scale_dtype,
    mxfp_per_token_scale_dtype,
    mxfp_use_bf16,
    is_per_channel_weight,
):
    return MoEQuantParams(
        quant_type=quant_type,
        comm_quant_mode=comm_quant_mode,
        mxfp=_build_mxfp_params(
            quant_type=quant_type,
            mxfp_act_quant_type=mxfp_act_quant_type,
            mxfp_weight_quant_type=mxfp_weight_quant_type,
            mxfp_scale_dtype=mxfp_scale_dtype,
            mxfp_per_token_scale_dtype=mxfp_per_token_scale_dtype,
            mxfp_use_bf16=mxfp_use_bf16,
        ),
        is_per_channel_weight=is_per_channel_weight,
    )
