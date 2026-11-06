# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors
"""Triton-free MRV2 sample ops for 310P.

Reuses the shared ``apply_top_k_top_p`` hardware-capability dispatch, just
like other Ascend devices.
"""

from vllm_ascend._310p.worker.v2.sample.logprob import gather_logprobs
from vllm_ascend._310p.worker.v2.sample.min_p import apply_min_p
from vllm_ascend._310p.worker.v2.sample.temperature import apply_temperature
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p

__all__ = [
    "apply_min_p",
    "apply_temperature",
    "apply_top_k_top_p",
    "gather_logprobs",
]
