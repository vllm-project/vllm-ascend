# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport vLLM #58542's prefill-only PP broadcast gate to v0.30.0.

Keep #54436's protection for decoding rows, including speculative steps that
temporarily overrun the request's length cap. Remove with the upstream backport.
"""

import numpy as np
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.input_batch import InputBatch


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return rows whose sampled outputs may be consumed by a later step."""
    prefill_len = input_batch.prefill_len_np
    need_sampled_mask = input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens >= prefill_len
    # v0.30.0's base InputBatch (dummy, PCP and ubatch paths) has no length caps.
    max_seq_len = getattr(input_batch, "max_seq_len_np", None)
    if max_seq_len is not None:
        finished_prefill = prefill_len + 1 >= max_seq_len
        finished_prefill &= input_batch.is_prefilling_np
        need_sampled_mask &= ~finished_prefill
    return need_sampled_mask if need_sampled_mask.any() else None


pp_utils.compute_need_sampled_mask = compute_need_sampled_mask
