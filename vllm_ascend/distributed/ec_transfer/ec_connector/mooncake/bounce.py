# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy bounce capacity configuration, now added to the fixed staging pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.config import VllmConfig
ASCEND_DIRECT_MEMORY_ALIGNMENT = 2 * 1024 * 1024
_BOUNCE_ARENA_CONFIG_KEY = "ascend_mooncake_bounce_arena_size"
_DEFAULT_BOUNCE_LIMIT = 128


@dataclass(frozen=True)
class _TransferFragmentPlan:
    """One physical Mooncake write fragment of a logical source."""

    source_index: int
    source_address: int
    destination_offset: int
    nbytes: int


def _resolve_bounce_arena_size(vllm_config: VllmConfig) -> int:
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None

    extra_config = ec_config.ec_connector_extra_config

    if _BOUNCE_ARENA_CONFIG_KEY not in extra_config:
        scheduler_config = vllm_config.scheduler_config
        max_num_seqs = scheduler_config.max_num_seqs
        max_num_active_seqs = getattr(
            scheduler_config,
            "max_num_active_seqs",
            None,
        )
        # Follow the scheduler's effective active-sequence capacity. Missing
        # or None preserves the pre-active-cap default exactly.
        effective_max_num_seqs = max_num_seqs if max_num_active_seqs is None else max_num_active_seqs
        quantum_count = min(
            _DEFAULT_BOUNCE_LIMIT,
            max_num_seqs,
            effective_max_num_seqs,
        )
        return ASCEND_DIRECT_MEMORY_ALIGNMENT * quantum_count

    value = extra_config[_BOUNCE_ARENA_CONFIG_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_BOUNCE_ARENA_CONFIG_KEY} must be an integer")
    if value == 0:
        return 0
    if value < ASCEND_DIRECT_MEMORY_ALIGNMENT:
        raise ValueError(f"{_BOUNCE_ARENA_CONFIG_KEY} must be 0 or at least {ASCEND_DIRECT_MEMORY_ALIGNMENT} bytes")

    return round_up(value, ASCEND_DIRECT_MEMORY_ALIGNMENT)
