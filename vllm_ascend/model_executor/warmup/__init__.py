# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton kernel warmup for Ascend NPU."""

from vllm_ascend.model_executor.warmup.kernel_warmup import kernel_warmup

__all__ = [
    "kernel_warmup",
]
