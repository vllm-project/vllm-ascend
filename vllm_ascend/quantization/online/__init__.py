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
"""Online (load-time) quantization subsystem.

Converts dense BF16/FP32 checkpoint weights into the layouts the W8A8
production schemes consume, so a checkpoint can be served quantized without
an offline calibration tool. This package only provides the numerical
kernels and accuracy tooling; the config and adapters that drive them are
delivered separately.
"""

from .accuracy import compare_layers_cosine, render_cosine_report
from .weight_ops import (
    dequantize_weight_int8_per_channel,
    dequantize_weight_mx_fp8,
    quantize_weight_int8_per_channel,
    quantize_weight_int8_per_channel_reference,
    quantize_weight_mx_fp4,
    quantize_weight_mx_fp8,
    quantize_weight_mx_fp8_reference,
)

__all__ = [
    "compare_layers_cosine",
    "dequantize_weight_int8_per_channel",
    "dequantize_weight_mx_fp8",
    "quantize_weight_int8_per_channel",
    "quantize_weight_int8_per_channel_reference",
    "quantize_weight_mx_fp8",
    "quantize_weight_mx_fp8_reference",
    "quantize_weight_mx_fp4",
    "render_cosine_report",
]
