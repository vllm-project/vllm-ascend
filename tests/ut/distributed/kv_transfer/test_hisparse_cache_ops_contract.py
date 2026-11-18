# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.v1.hisparse.runtime import HiSparseRuntime

from vllm_ascend.hisparse import cache_ops


def test_current_core_resolver_call_preserves_native_argument_roles(monkeypatch):
    # Exercise the actual Core caller so positional ABI drift cannot pass silently.
    rows, topk, hot_size, block = 2, 2, 4, 2
    indices = torch.zeros((rows, topk), dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    valid = torch.tensor(rows, dtype=torch.int32)
    table = torch.ones((rows, 2), dtype=torch.int32)
    hot = torch.zeros((4, block, 4), dtype=torch.uint8)
    native = MagicMock()
    workspace = SimpleNamespace(
        max_rows=rows,
        max_topk=topk,
        hot_size=hot_size,
        max_union_rows=hot_size,
        first_refs=torch.empty(rows * topk, dtype=torch.int32),
        errors=torch.zeros((rows, 8), dtype=torch.int32),
        native=native,
    )
    ops = cache_ops.HiSparseNPUOps(None, workspace)
    monkeypatch.setattr(cache_ops, "_cache_layout", lambda *args: None)
    monkeypatch.setattr(cache_ops, "_check_tensor", lambda *args, **kwargs: None)
    shared = SimpleNamespace(
        device_topk_rows=indices.clone(),
        valid_topk_counts=counts.clone(),
        physical_topk_indices=indices.clone(),
        swap_host_physical_rows=indices.clone(),
        swap_device_physical_rows=indices.clone(),
        swap_counts=counts.clone(),
    )
    stats = torch.zeros(2, dtype=torch.int64)
    group = SimpleNamespace(
        shared_topk=shared,
        device_global_indices=torch.zeros((rows, hot_size), dtype=torch.int32),
        lru_slots=torch.zeros((rows, hot_size), dtype=torch.int16),
        swap_stats=stats,
    )
    runtime = SimpleNamespace(
        hot=SimpleNamespace(cache=hot, block_size=block, attention_block_stride=block),
        host_cache=torch.zeros((16, 4), dtype=torch.uint8),
        request_state_indices=torch.arange(rows, dtype=torch.int32),
        index_group=group,
        hot_block_table=table,
        region_stride=hot_size,
        max_union_rows=hot_size,
        cache_ops=ops,
    )
    resident = SimpleNamespace(block_table=table, view=SimpleNamespace(block_size=block))
    HiSparseRuntime._resolve_residency(
        runtime,
        resident=resident,
        req_id_per_token=torch.arange(rows, dtype=torch.int32),
        num_valid_rows=valid,
        block_table=table,
        topk_indices=indices,
        block_size=block,
        return_valid_counts=True,
        shared_rows=slice(0, rows),
    )
    args = native.hisparse_resolve_residency.call_args.args
    assert len(args) == 27
    assert args[6] is valid
    assert args[17] is None  # Core uses compact outputs, not a dense miss mask.
    assert args[18] is stats
    assert args[21:24] == (block, block, 0)
    assert args[13].data_ptr() == shared.valid_topk_counts.data_ptr()
    assert args[14].data_ptr() == shared.swap_host_physical_rows.data_ptr()
    assert args[15].data_ptr() == shared.swap_device_physical_rows.data_ptr()
    assert args[16].data_ptr() == shared.swap_counts.data_ptr()
