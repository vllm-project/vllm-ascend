# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for Ascend DFlash/DSpark speculators."""

from contextlib import contextmanager
from typing import Any

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend


# TODO: Remove this helper and build_draft_attn_metadatas once tiling
# eliminates replay-time metadata rebuilding. The rebuild currently needs
# this count because it cannot access the effective num_reqs from upstream propose().
@contextmanager
def draft_metadata_request_count(speculator: Any, num_reqs: int):
    previous = getattr(speculator, "_draft_metadata_num_reqs", None)
    speculator._draft_metadata_num_reqs = num_reqs
    try:
        yield
    finally:
        speculator._draft_metadata_num_reqs = previous


@contextmanager
def proposal_host_lengths(speculator: Any, num_reqs: int):
    """Scope the request count and the host-length cache to one propose() call."""
    with draft_metadata_request_count(speculator, num_reqs):
        previous = getattr(speculator, "_proposal_host_lengths", None)
        speculator._proposal_host_lengths = {}
        try:
            yield
        finally:
            speculator._proposal_host_lengths = previous


def prepare_draft_host_lengths(speculator: Any, num_reqs: int, num_reqs_padded: int, step: int) -> torch.Tensor:
    """Zero-padded host copy of the per-step device seq_lens, for FIA drafts.

    The input-prep kernel rewrites input_buffers.seq_lens every step with
    min(last_valid_pos + 1 + step, max_model_len): the accepted context plus
    this step's queries. build_attn_metadata_factory feeds these exact lengths
    as seq_lens_np so the builder emits them as the FIA seq_lens_list; the
    optimistic CPU upper bound would drag up to step stale rows into the
    right-aligned sliding window after a rejection.
    """
    cache = getattr(speculator, "_proposal_host_lengths", None)
    shape = (num_reqs, num_reqs_padded)
    if cache is not None and shape in cache:
        return cache[shape]
    seq_lens_cpu = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
    seq_lens_cpu[:num_reqs].copy_(speculator.input_buffers.seq_lens[:num_reqs])
    if cache is not None:
        cache[shape] = seq_lens_cpu
    return seq_lens_cpu


def detect_draft_attn_architecture(attn_groups: Any) -> tuple[str | None, bool]:
    """Classify the draft attention backend from the initialized attention groups.

    Returns (attn_architecture, use_flashmla): "MLA" or "GQA" when the draft
    runs on the corresponding Ascend backend, None otherwise, plus whether
    every MLA group resolved to FlashMLA. FIA-backed MLA drafts need the
    rejection-adjusted host lengths; FlashMLA keeps consuming device lengths.
    """
    backend = _get_graph_update_backend(attn_groups)
    if issubclass(backend, AscendMLABackend):
        mla_groups = [
            group for groups in attn_groups for group in groups if issubclass(group.backend, AscendMLABackend)
        ]
        use_flashmla = bool(mla_groups) and all(
            getattr(group.get_metadata_builder(0), "flashmla_state", None) is not None for group in mla_groups
        )
        return "MLA", use_flashmla
    if issubclass(backend, AscendAttentionBackend):
        return "GQA", False
    return None, False
