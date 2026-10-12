# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend attention metadata context for PARD-2's parallel query pass."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import ParallelConfig


@contextmanager
def pard2_attn_metadata_context(
    *,
    positions: torch.Tensor,
    num_tokens: int,
    is_prefilling: torch.Tensor,
    parallel_config: ParallelConfig,
) -> Iterator[None]:
    """Apply PARD-2's Ascend state while the upstream builder runs.

    Keep attention imports here: importing the speculator class must not load
    the attention backend and trigger device/ops registration. The two wrappers
    restore the previous metadata builder even if construction raises.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper, build_draft_attn_metadata_factory

    with (
        build_attn_metadata_wrapper(),
        build_draft_attn_metadata_factory(
            positions,
            num_tokens,
            is_prefilling,
            attn_state=AscendAttentionState.ChunkedPrefill,
            parallel_config=parallel_config,
        ),
    ):
        yield
