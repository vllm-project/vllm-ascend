# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

TURBOQUANT_CACHE_DTYPE = "turboquant_4bit_nc"
HEAD_DIM = 512
SLOT_BYTES = HEAD_DIM // 2 + 2


def is_turboquant(vllm_config) -> bool:
    return getattr(getattr(vllm_config, "cache_config", None), "cache_dtype", None) == TURBOQUANT_CACHE_DTYPE
