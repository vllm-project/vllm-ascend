# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np


def prompt_tail_request_mask(
    is_prefilling: np.ndarray,
    num_computed_prefill_tokens: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    prefill_len: np.ndarray,
    decode_query_len: int,
) -> np.ndarray:
    """Requests whose query is the last decode-shaped chunk of the prompt.

    Async KV recv schedules one such token (``prompt_len - 1`` already
    computed). Treating it as prefill drops FULL_DECODE_ONLY onto eager, and
    one data-parallel rank in eager forces every rank eager for that step.
    """
    return (
        is_prefilling
        & (num_computed_prefill_tokens > 0)
        & (num_scheduled_tokens == decode_query_len)
        & (num_computed_prefill_tokens + num_scheduled_tokens >= prefill_len)
    )
