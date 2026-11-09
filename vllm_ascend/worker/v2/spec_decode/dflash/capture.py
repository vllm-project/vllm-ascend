# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bind Window reads through the original DFlash capture metadata builder."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as cudagraph_module

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.draft_window.graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphBucket,
)


@contextmanager
def draft_window_dflash_capture_scope(owner: Any) -> Iterator[DraftWindowGraphBindingProvider | None]:
    """Add one read callback while preserving the fixed dummy/slot preparation.

    Worker Host capture is serialized, like the existing metadata factory.
    Restore this independent capture-module alias on normal or failed exit.
    """
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        yield None
        return
    raw = cudagraph_module.build_attn_metadata
    if raw is not build_attn_metadata:
        raise RuntimeError("Draft Window capture metadata builder is already scoped or not installed")

    def build(**kwargs):
        if kwargs.get("common_metadata_callback") is not None:
            raise ValueError("Draft Window capture metadata already has a read callback")
        bank = provider.capture_banks_for(num_reqs=kwargs["num_reqs"], num_tokens=kwargs["num_tokens"])[0]
        _seed_capture_workspace(bank, max_model_len=kwargs["max_seq_len"])

        def bind(gid, common):
            if gid == provider.group.gid:
                common.block_table_tensor = bank.window_block_table
                common._draft_window_exact_cpu = bank.cpu_window_exact

        kwargs["common_metadata_callback"] = bind
        return raw(**kwargs)

    try:
        cudagraph_module.build_attn_metadata = build
        yield provider
    finally:
        cudagraph_module.build_attn_metadata = raw


def _seed_capture_workspace(
    workspace: DraftWindowGraphBucket,
    *,
    max_model_len: int,
) -> None:
    # The actual provider borrows this same canonical BlockTables owner.
    # Capture prepares its Full dummy tables before seeding the private bank.
    provider = workspace.owner
    source = provider.table[: workspace.descriptor.num_reqs]
    destination = workspace.window_block_table

    capture_upper = min(
        max_model_len,
        provider.window_size + provider.group.table_column_span - 1,
    )
    valid_num_blocks = (capture_upper + provider.group.table_column_span - 1) // provider.group.table_column_span
    if valid_num_blocks > source.shape[1] or valid_num_blocks > destination.shape[1]:
        raise RuntimeError("Draft Window capture table cannot cover the Window capacity")

    destination.zero_()
    destination[:, :valid_num_blocks].copy_(source[:, :valid_num_blocks])
    workspace.cpu_window_exact.fill_(capture_upper)
