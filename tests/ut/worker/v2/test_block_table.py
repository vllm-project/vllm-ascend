# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from vllm_ascend.patch.worker.patch_v2 import patch_uva
from vllm_ascend.worker.v2.block_table import AscendBlockTables


def test_encoder_only_zero_cache_groups_initialize_and_compute(monkeypatch) -> None:
    """Exercise the real block-table construction used by an encoder producer."""
    original_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    # CPU CI cannot allocate Ascend pinned memory. Only substitute that device
    # facility; the production AscendBlockTables constructor remains real.
    monkeypatch.setattr(torch, "zeros", zeros_without_pinning)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)

    block_tables = AscendBlockTables(
        block_sizes=[],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[],
        device=torch.device("cpu"),
        kernel_block_sizes=[],
    )

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        positions=torch.tensor([0], dtype=torch.int64),
        num_tokens_padded=1,
    )

    assert block_tables.num_kv_cache_groups == 0
    assert block_tables._block_table_pad_size == 1
    assert slot_mappings.shape == (0, 1)
