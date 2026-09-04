# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up the KV-block zeroing Triton kernel on Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def kv_block_zeroer_triton_warmup(worker: NPUWorker) -> None:
    """Compile the KV-block zeroing Triton kernel before the first request."""
    model_runner = worker.model_runner
    if worker.use_v2_model_runner:
        zeroer = getattr(model_runner, "kv_block_zeroer", None)
    else:
        zeroer = getattr(model_runner, "_kv_block_zeroer", None)
    if zeroer is not None:
        zeroer.warmup(model_runner.kv_cache_config.num_blocks)
