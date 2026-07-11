# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import functools

from vllm.logger import logger
from vllm.model_executor.layers.mamba.ops import mamba_ssm

_ORIGINAL_TRY_GET_OPTIMAL_SSM_CONFIG = mamba_ssm.try_get_optimal_ssm_config

# Tuned with vLLM's selective-state-update benchmark on Ascend 910B3. The
# keys are effective batch sizes (batch * TP-local heads).
_ASCEND_910B3_FP32_CONFIGS = {
    16: (64, 1),
    32: (64, 4),
    64: (64, 1),
}


@functools.cache
def _get_optimal_ssm_config_npu_cached(
    headdim: int,
    dstate: int,
    batch: int,
    nheads: int,
    cache_dtype: str,
    is_blackwell: bool,
) -> tuple[int, int]:
    if (
        headdim == 64
        and dstate == 128
        and cache_dtype == "float32"
        and mamba_ssm.get_ssm_device_name() == "Ascend910B3"
    ):
        logger.info_once(
            "Using tuned Ascend 910B3 Mamba SSU config for headdim=64, dstate=128, and float32 state cache.",
            scope="global",
        )
        effective_batch = batch * nheads
        closest = min(
            _ASCEND_910B3_FP32_CONFIGS,
            key=lambda candidate: abs(candidate - effective_batch),
        )
        return _ASCEND_910B3_FP32_CONFIGS[closest]

    return _ORIGINAL_TRY_GET_OPTIMAL_SSM_CONFIG(
        headdim,
        dstate,
        batch,
        nheads,
        cache_dtype,
        is_blackwell,
    )


def try_get_optimal_ssm_config_npu(
    headdim: int,
    dstate: int,
    batch: int,
    nheads: int,
    cache_dtype: str,
    is_blackwell: bool,
) -> tuple[int, int]:
    # Keep vLLM's benchmark override authoritative even after a production
    # config for the same shape has been cached.
    if getattr(mamba_ssm, "_ssm_config_override", None) is not None:
        return _ORIGINAL_TRY_GET_OPTIMAL_SSM_CONFIG(
            headdim,
            dstate,
            batch,
            nheads,
            cache_dtype,
            is_blackwell,
        )
    return _get_optimal_ssm_config_npu_cached(
        headdim,
        dstate,
        batch,
        nheads,
        cache_dtype,
        is_blackwell,
    )
