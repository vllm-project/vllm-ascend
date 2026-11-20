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
"""Shared quantization enum definitions.

Keep this module lightweight and side-effect free so core runtime modules can
import QuantType without triggering heavy quantization package initialization.
"""

from enum import Enum


class QuantType(Enum):
    """Quantization type enum for MoE schemes."""

    NONE = 0  # No quantization
    W8A8 = 1  # W and A are INT8
    W4A8 = 2  # W is INT4, A is INT8
    W8A8MXFP = 3  # W and A are MXFP8
    W4A16 = 4  # W is INT4, A is BF16 or FP16
    W4A4MXFP = 5  # W and A are MXFP4
    W4A8MXFP = 6  # W is MXFP4, A is MXFP8
    W8A8FP = 7  # W and A are FP8
    W4A16MXFP = 8  # W is MXFP4, A is BF16 or FP16


# Quant types whose weight layouts the A5 MegaMoe (FUSED_MC2) operator supports.
A5_SUPPORT_MEGA_MOE_QUANT_TYPES = frozenset(
    {
        QuantType.W4A4MXFP,
        QuantType.W4A8MXFP,
        QuantType.W8A8MXFP,
    }
)

# Quant types whose weight layouts the A2 dispatch_ffn_combine (FUSED_MC2)
# operator supports. On A2 the W4A8 variant and the BF16 variant are built for
# ascend910b; QuantType.NONE is the unquantized (BF16/FP16) weight layout that
# the BF16 variant consumes, which is also the MTP/draft layer's layout. MoE
# layers with another quant type must stay on the non-fused path instead of
# selecting FUSED_MC2.
A2_SUPPORT_DISPATCH_FFN_COMBINE_QUANT_TYPES = frozenset({QuantType.NONE, QuantType.W4A8})
