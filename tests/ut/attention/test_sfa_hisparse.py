# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for HiSparse metadata feeding SparseKvPlan."""

from types import SimpleNamespace

import torch

from vllm_ascend.attention.sfa_hisparse import AscendHiSparseMetadataBuilder


def _builder():
    builder = object.__new__(AscendHiSparseMetadataBuilder)
    builder.device = torch.device("cpu")
    builder._row_capacity = 8
    builder._sparse_plan_inputs = {}
    builder._sparse_plan_max_token = 100
    return builder


def _metadata():
    metadata = object.__new__(type("Metadata", (), {}))
    metadata.num_decode_tokens = 8
    metadata.max_query_len = 4
    metadata.num_reqs = 2
    metadata.seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    metadata.query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    metadata.plan_request_ids = None
    metadata.plan_token_to_req = None
    metadata.plan_stable_prefix_lens = None
    metadata.plan_visible_seq_lens = None
    metadata.plan_active_rows = None
    metadata.plan_max_token = 0
    return metadata


def test_uniform_sparse_plan_groups_accept_fixed_width_decode():
    builder = _builder()
    metadata = SimpleNamespace(num_reqs=2, max_query_len=4, num_decode_tokens=8)
    common = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 4, 8]))
    assert builder._has_uniform_sparse_plan_groups(common, metadata)


def test_nonuniform_sparse_plan_groups_are_rejected():
    builder = _builder()
    metadata = SimpleNamespace(num_reqs=2, max_query_len=4, num_decode_tokens=8)
    common = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 4, 6]))
    assert not builder._has_uniform_sparse_plan_groups(common, metadata)


def test_sparse_plan_inputs_use_durable_request_ids():
    builder = _builder()
    metadata = _metadata()
    request_ids = torch.tensor([101, 102], dtype=torch.int64)

    builder._build_sparse_kv_plan_inputs(
        metadata,
        builder._get_sparse_kv_plan_buffers(metadata, draft_index=None),
        request_ids,
    )

    assert torch.equal(metadata.plan_request_ids, request_ids)
    assert torch.equal(metadata.plan_token_to_req, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    assert torch.equal(metadata.plan_stable_prefix_lens, torch.tensor([6, 6, 16, 16], dtype=torch.int32))
    assert torch.equal(metadata.plan_visible_seq_lens, torch.tensor([7, 8, 17, 18], dtype=torch.int32))
    assert torch.equal(metadata.plan_active_rows, torch.tensor([8], dtype=torch.int32))
    assert metadata.plan_max_token == 100


def test_sparse_plan_inputs_require_durable_request_ids():
    builder = _builder()
    metadata = _metadata()

    builder._build_sparse_kv_plan_inputs(
        metadata,
        builder._get_sparse_kv_plan_buffers(metadata, draft_index=None),
        None,
    )

    assert metadata.plan_request_ids is None
    assert metadata.plan_token_to_req is None
    assert metadata.plan_stable_prefix_lens is None
    assert metadata.plan_visible_seq_lens is None
    assert metadata.plan_active_rows is None
    assert metadata.plan_max_token == 0
