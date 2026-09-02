# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention import (
    build_dflash_hybrid_draft_paged_view_310,
    create_dflash_hybrid_draft_attention_inputs_310,
    dflash_hybrid_draft_graph_safe_attention_310,
    update_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend._310p.attention.metadata_builder import (
    AscendAttentionMetadataBuilder310,
    dflash_hybrid_draft_capture_scope_310,
    get_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState


def _prepare(
    *,
    capacity_reqs: int,
    capacity_tokens: int,
    query_lens: Sequence[int],
    seq_lens: Sequence[int],
):
    inputs = create_dflash_hybrid_draft_attention_inputs_310(
        capacity_reqs=capacity_reqs,
        capacity_tokens=capacity_tokens,
        max_blocks=4,
        device=torch.device("cpu"),
    )
    block_table = torch.arange(
        capacity_reqs * 4,
        dtype=torch.int32,
    ).reshape(capacity_reqs, 4)
    update_dflash_hybrid_draft_attention_inputs_310(
        inputs,
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=block_table,
        valid_num_reqs=len(query_lens),
        valid_num_tokens=sum(query_lens),
    )
    return inputs


@pytest.mark.parametrize(
    ("capacity_reqs", "capacity_tokens", "query_lens", "seq_lens"),
    [
        (16, 64, [1], [24]),
        (16, 64, [1, 1, 1, 1, 1, 1, 1], [24, 18, 31, 9, 12, 22, 27]),
        (16, 160, [15], [31]),
        (20, 400, [8, 15, 20], [18, 29, 35]),
    ],
)
def test_device_view_separates_runtime_requests_from_physical_descriptor(
    capacity_reqs,
    capacity_tokens,
    query_lens,
    seq_lens,
):
    inputs = _prepare(
        capacity_reqs=capacity_reqs,
        capacity_tokens=capacity_tokens,
        query_lens=query_lens,
        seq_lens=seq_lens,
    )

    view = build_dflash_hybrid_draft_paged_view_310(inputs)

    expected_ids = []
    for request_id, query_len in enumerate(query_lens):
        expected_ids.extend([request_id] * query_len)
    expected_ids.extend([0] * (capacity_tokens - sum(query_lens)))
    assert view.request_ids.tolist() == expected_ids
    assert view.valid_token_mask.tolist() == [True] * sum(query_lens) + [
        False
    ] * (capacity_tokens - sum(query_lens))
    assert view.context_lens[: sum(query_lens)].tolist() == [
        seq_lens[request_id] for request_id in expected_ids[: sum(query_lens)]
    ]
    assert view.context_lens[sum(query_lens) :].tolist() == [1] * (
        capacity_tokens - sum(query_lens)
    )


def test_same_physical_inputs_can_be_updated_for_a_different_runtime_topology():
    inputs = _prepare(
        capacity_reqs=16,
        capacity_tokens=64,
        query_lens=[1],
        seq_lens=[24],
    )
    first_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            inputs.valid_num_reqs,
            inputs.valid_num_tokens,
            inputs.query_lens,
            inputs.query_starts,
            inputs.query_ends,
            inputs.seq_lens,
            inputs.block_table,
        )
    )
    block_table = torch.arange(64, dtype=torch.int32).reshape(16, 4)

    update_dflash_hybrid_draft_attention_inputs_310(
        inputs,
        query_lens=torch.tensor([1, 1, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([25, 19, 32, 10], dtype=torch.int32),
        block_table=block_table,
        valid_num_reqs=4,
        valid_num_tokens=4,
    )

    assert first_ptrs == tuple(
        tensor.data_ptr()
        for tensor in (
            inputs.valid_num_reqs,
            inputs.valid_num_tokens,
            inputs.query_lens,
            inputs.query_starts,
            inputs.query_ends,
            inputs.seq_lens,
            inputs.block_table,
        )
    )
    view = build_dflash_hybrid_draft_paged_view_310(inputs)
    assert view.request_ids[:4].tolist() == [0, 1, 2, 3]
    assert view.context_lens[:4].tolist() == [25, 19, 32, 10]
    assert not view.valid_token_mask[4:].any()


def test_hybrid_builder_owns_config_and_attaches_private_inputs_for_drafting():
    builder = object.__new__(AscendAttentionMetadataBuilder310)
    builder._vllm_config_310 = object()
    private_inputs = object()
    builder._prepare_dflash_hybrid_draft_attention_inputs_310 = Mock(
        return_value=private_inputs
    )
    common = SimpleNamespace(
        causal=False,
        num_reqs=1,
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
    )
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with (
        patch(
            "vllm_ascend._310p.attention.metadata_builder."
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.attention.metadata_builder."
            "AscendAttentionMetadataBuilder.build",
            return_value=metadata,
        ),
    ):
        actual = AscendAttentionMetadataBuilder310.build(
            builder,
            0,
            common,
            is_drafting=True,
            dflash_hybrid_draft_step=1,
        )

    assert actual is metadata
    assert get_dflash_hybrid_draft_attention_inputs_310(actual) is private_inputs
    builder._prepare_dflash_hybrid_draft_attention_inputs_310.assert_called_once_with(
        common,
        draft_step=1,
        real_num_reqs_override=None,
        capacity_tokens_override=None,
    )


def test_dummy_capture_scope_attaches_contract_without_marking_runtime_drafting():
    builder = object.__new__(AscendAttentionMetadataBuilder310)
    builder._vllm_config_310 = object()
    private_inputs = object()
    builder._prepare_dflash_hybrid_draft_attention_inputs_310 = Mock(
        return_value=private_inputs
    )
    common = SimpleNamespace(
        causal=False,
        num_reqs=16,
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
    )
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with (
        patch(
            "vllm_ascend._310p.attention.metadata_builder."
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.attention.metadata_builder."
            "AscendAttentionMetadataBuilder.build",
            return_value=metadata,
        ),
        dflash_hybrid_draft_capture_scope_310(
            real_num_reqs=16,
            capacity_tokens=64,
        ),
    ):
        actual = AscendAttentionMetadataBuilder310.build(
            builder,
            0,
            common,
            is_drafting=False,
        )

    assert get_dflash_hybrid_draft_attention_inputs_310(actual) is private_inputs
    builder._prepare_dflash_hybrid_draft_attention_inputs_310.assert_called_once_with(
        common,
        draft_step=-1,
        real_num_reqs_override=16,
        capacity_tokens_override=64,
    )


def test_private_entry_calls_paged_attention_and_zeros_dummy_rows():
    inputs = _prepare(
        capacity_reqs=4,
        capacity_tokens=8,
        query_lens=[1, 1],
        seq_lens=[4, 3],
    )
    query = torch.randn(8, 2, 4, dtype=torch.float16)
    key_cache = torch.randn(3, 4, 1, 4, dtype=torch.float16)
    value_cache = torch.randn(3, 4, 1, 4, dtype=torch.float16)
    output = torch.empty_like(query)
    calls = []

    def fake_paged_attention(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(2)

    with patch(
        "vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention.torch_npu._npu_paged_attention",
        fake_paged_attention,
    ):
        actual = dflash_hybrid_draft_graph_safe_attention_310(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            inputs=inputs,
            num_kv_heads=1,
            num_heads=2,
            scale=0.5,
            output=output,
        )

    assert len(calls) == 1
    assert calls[0]["context_lens"].shape == (8,)
    assert calls[0]["block_table"].shape == (8, 4)
    torch.testing.assert_close(actual[:2], torch.full_like(actual[:2], 2))
    torch.testing.assert_close(actual[2:], torch.zeros_like(actual[2:]))


@pytest.mark.parametrize(
    ("valid_num_reqs", "valid_num_tokens"),
    [(17, 16), (16, 65)],
)
def test_update_rejects_runtime_workload_beyond_descriptor(
    valid_num_reqs,
    valid_num_tokens,
):
    inputs = create_dflash_hybrid_draft_attention_inputs_310(
        capacity_reqs=16,
        capacity_tokens=64,
        max_blocks=4,
        device=torch.device("cpu"),
    )
    query_lens = torch.ones(valid_num_reqs, dtype=torch.int32)
    seq_lens = torch.ones(valid_num_reqs, dtype=torch.int32)
    block_table = torch.zeros(valid_num_reqs, 4, dtype=torch.int32)

    with pytest.raises(ValueError, match="exceeds FULL descriptor capacity"):
        update_dflash_hybrid_draft_attention_inputs_310(
            inputs,
            query_lens=query_lens,
            seq_lens=seq_lens,
            block_table=block_table,
            valid_num_reqs=valid_num_reqs,
            valid_num_tokens=valid_num_tokens,
        )
