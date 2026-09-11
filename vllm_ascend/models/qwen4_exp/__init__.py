# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Qwen3.8-Flash-Next support for Ascend NPU."""

from .model import (
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
    AscendQwen4ExpModel,
)
from .mtp import AscendQwen4ExpMTP, AscendQwen4ExpMultiTokenPredictor

__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpMTP",
    "AscendQwen4ExpModel",
    "AscendQwen4ExpMultiTokenPredictor",
]
