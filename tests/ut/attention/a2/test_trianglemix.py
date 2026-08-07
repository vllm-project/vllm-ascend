#
# Copyright (c) 2026 TriangleMix contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for TriangleMix configuration and scheduler-step routing."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.trianglemix import (
    TriangleMixConfig,
    TriangleMixFallbackReason,
    TriangleMixRequestPlan,
    build_trianglemix_plan,
    parse_layer_indices,
    run_trianglemix,
    trianglemix_dispatch_reason,
)


def test_parse_layer_indices() -> None:
    assert parse_layer_indices("5,7,10,15-17") == frozenset({5, 7, 10, 15, 16, 17})


@pytest.mark.parametrize("value", ["-1", "7-5", "x"])
def test_parse_layer_indices_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_layer_indices(value)


def test_config_is_opt_in_and_requires_layers() -> None:
    assert not TriangleMixConfig.from_mapping({}).enabled
    with pytest.raises(ValueError, match="requires at least one"):
        TriangleMixConfig.from_mapping({"enabled": True})


def test_additional_config_controls_selected_layers() -> None:
    config = TriangleMixConfig.from_mapping(
        {"enabled": False, "layers": "3"},
    )
    assert not config.enabled
    assert config.layer_indices == frozenset({3})


def test_long_single_prefill_uses_direct_path() -> None:
    config = TriangleMixConfig.from_mapping(
        {
            "enabled": True,
            "layers": "5",
            "min_sparse_rows": 1,
            "min_saved_qk": 1,
        }
    )
    plan = build_trianglemix_plan(
        state_name="ChunkedPrefill",
        cumulative_query_ends=[2048],
        seq_lens=[8320],
        prompt_lens=[8320],
        num_decodes=0,
        num_prefills=1,
        config=config,
    )
    assert plan.direct
    assert plan.query_start == 6272
    assert plan.sparse_end == 8192
    assert plan.saved_qk > 0


def test_prompt_length_controls_dense_tail_across_chunks() -> None:
    config = TriangleMixConfig.from_mapping(
        {
            "enabled": True,
            "layers": "5",
            "min_sparse_rows": 0,
            "min_saved_qk": 0,
        }
    )
    plan = build_trianglemix_plan(
        state_name="ChunkedPrefill",
        cumulative_query_ends=[2048],
        seq_lens=[4096],
        prompt_lens=[8320],
        num_decodes=0,
        num_prefills=1,
        config=config,
    )
    assert plan.sparse_end == 4096
    assert plan.prompt_len == 8320


@pytest.mark.parametrize(
    ("state", "query_ends", "seq_lens", "prompt_lens", "decodes", "prefills", "reason"),
    [
        (
            "DecodeOnly",
            [1],
            [8321],
            [8320],
            1,
            0,
            TriangleMixFallbackReason.STATE_UNSUPPORTED,
        ),
        (
            "ChunkedPrefill",
            [1024, 2048],
            [1024, 1024],
            [1024, 1024],
            0,
            2,
            TriangleMixFallbackReason.BATCH_UNSUPPORTED,
        ),
        (
            "ChunkedPrefill",
            None,
            None,
            None,
            0,
            0,
            TriangleMixFallbackReason.MISSING_METADATA,
        ),
    ],
)
def test_unsupported_requests_fall_back(
    state,
    query_ends,
    seq_lens,
    prompt_lens,
    decodes,
    prefills,
    reason,
) -> None:
    config = TriangleMixConfig.from_mapping({"enabled": True, "layers": "5"})
    plan = build_trianglemix_plan(
        state_name=state,
        cumulative_query_ends=query_ends,
        seq_lens=seq_lens,
        prompt_lens=prompt_lens,
        num_decodes=decodes,
        num_prefills=prefills,
        config=config,
    )
    assert not plan.direct
    assert plan.reason is reason


def _direct_plan(query_len: int = 2048) -> TriangleMixRequestPlan:
    return TriangleMixRequestPlan(
        query_len=query_len,
        seq_len=8320,
        prompt_len=8320,
        query_start=8320 - query_len,
        sparse_start=8320 - query_len,
        sparse_end=8192,
        saved_qk=1_500_000,
        reason=TriangleMixFallbackReason.NONE,
    )


def test_runtime_contract_rejects_unsupported_geometry() -> None:
    config = TriangleMixConfig.from_mapping({"enabled": True, "layers": "5"})
    query = torch.empty(2048, 16, 128, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    key_cache = torch.empty(65, 128, 8, 128, dtype=torch.bfloat16)
    value_cache = torch.empty_like(key_cache)
    block_table = torch.arange(65, dtype=torch.int32).view(1, -1)

    reason = trianglemix_dispatch_reason(
        config=config,
        plan=_direct_plan(),
        layer_name="model.layers.5.self_attn.attn",
        query=query,
        output=output,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        causal=True,
        capturing=False,
        tensor_parallel_size=1,
        context_parallel_enabled=False,
        sliding_window=None,
        sinks=None,
        alibi_slopes=None,
        enable_c8_quant=False,
    )

    assert reason is TriangleMixFallbackReason.QUERY_UNSUPPORTED


def test_run_trianglemix_uses_native_out_operator() -> None:
    query = torch.empty(2048, 32, 128, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    key_cache = torch.empty(65, 128, 8, 128, dtype=torch.bfloat16)
    value_cache = torch.empty_like(key_cache)
    block_table = torch.arange(65, dtype=torch.int32).view(1, -1)

    with patch.object(
        torch.ops._C_ascend,
        "npu_triangle_paged_sparse_attention",
        create=True,
        return_value=output,
    ) as op:
        result = run_trianglemix(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            plan=_direct_plan(),
            scale=0.0883883476,
            output=output,
        )

    assert result is output
    args = op.call_args.args
    assert args[0] is query
    assert args[1] is key_cache
    assert args[2] is value_cache
    assert args[3] is block_table
    assert args[4:7] == (6272, 8320, 8320)
    assert args[8].data_ptr() == output.data_ptr()
    assert args[8].shape == query.shape
