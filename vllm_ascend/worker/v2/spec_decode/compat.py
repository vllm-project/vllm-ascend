# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import ModuleType
from typing import Any


def resolve_rejection_sampler_helpers(
    upstream_module: ModuleType,
) -> tuple[Any, Any, Any]:
    """Resolve private rejection-sampler helpers across vLLM revisions."""
    if hasattr(upstream_module, "_compute_global_logsumexp"):
        compute_global_lse = upstream_module._compute_global_logsumexp
        compute_block_stats = upstream_module._compute_local_logits_stats_kernel
    else:
        compute_global_lse = upstream_module._compute_global_lse
        compute_block_stats = upstream_module._compute_block_stats_kernel

    return (
        compute_global_lse,
        compute_block_stats,
        upstream_module._insert_resampled_kernel,
    )
