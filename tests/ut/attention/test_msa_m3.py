# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for MiniMax M3 sparse attention layer wiring in ``msa_m3``."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.msa_m3 import (
    AscendMiniMaxM3IndexerMetadataBuilder,
    AscendMiniMaxM3SparseBackend,
    AscendMiniMaxM3SparseMetadataBuilder,
    minimax_m3_sparse_forward,
)


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


def _create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(batch_spec.query_lens)

    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [
        batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        for i in range(batch_spec.batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def _make_vllm_config(
    *,
    max_num_batched_tokens: int = 8192,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )


def _make_sparse_builder(device: torch.device) -> AscendMiniMaxM3SparseMetadataBuilder:
    vllm_config = _make_vllm_config()
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    return AscendMiniMaxM3SparseMetadataBuilder(
        spec,
        ["layer0.attn"],
        vllm_config,
        device,
    )


def _make_indexer_builder(device: torch.device) -> AscendMiniMaxM3IndexerMetadataBuilder:
    vllm_config = _make_vllm_config()
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    return AscendMiniMaxM3IndexerMetadataBuilder(
        spec,
        ["layer0.attn.index_cache"],
        vllm_config,
        device,
    )


def test_minimax_m3_sparse_custom_op_registered() -> None:
    assert hasattr(torch.ops.vllm, "minimax_m3_sparse_forward")


def test_sparse_backend_get_name() -> None:
    assert AscendMiniMaxM3SparseBackend.get_name() == "ASCEND_MINIMAX_M3_SPARSE"
    assert AscendMiniMaxM3SparseBackend.is_sparse() is True


@patch("vllm_ascend.attention.msa_m3.get_forward_context")
def test_minimax_m3_sparse_forward_dispatches_to_layer(
    mock_get_forward_context: MagicMock,
) -> None:
    layer = MagicMock()
    layer._run_sparse_attention = MagicMock()
    mock_get_forward_context.return_value = MagicMock(
        attn_metadata={"layer.attn": object()},
        no_compile_layers={"layer.attn": layer},
    )

    q = torch.randn(2, 32, 128)
    k = torch.randn(2, 2, 128)
    v = torch.randn(2, 2, 128)
    index_q = torch.randn(2, 2, 128)
    index_k = torch.randn(2, 128)
    out = torch.empty(2, 32 * 128)

    minimax_m3_sparse_forward(q, k, v, index_q, index_k, out, "layer.attn")

    layer._run_sparse_attention.assert_called_once_with(
        q, k, v, index_q, index_k, out
    )


@patch("vllm_ascend.attention.msa_m3.get_forward_context")
def test_minimax_m3_sparse_forward_zeros_output_without_dict_metadata(
    mock_get_forward_context: MagicMock,
) -> None:
    mock_get_forward_context.return_value = MagicMock(attn_metadata=None)
    out = torch.ones(2, 32 * 128)
    minimax_m3_sparse_forward(
        torch.randn(2, 32, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 128),
        out,
        "layer.attn",
    )
    assert torch.all(out == 0)


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[129, 257], query_lens=[129, 257], name="prefill_only"),
        BatchSpec(seq_lens=[130, 131], query_lens=[1, 1], name="decode_only"),
    ],
    ids=lambda case: case.name,
)
def test_sparse_metadata_builder(batch_spec: BatchSpec) -> None:
    device = torch.device("cpu")
    builder = _make_sparse_builder(device)
    common = _create_common_attn_metadata(batch_spec, block_size=128, device=device)
    metadata = builder.build(0, common)

    assert metadata.num_actual_tokens == sum(batch_spec.query_lens)
    assert metadata.num_decodes + metadata.num_prefills == batch_spec.batch_size
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
        assert metadata.decode.decode_query_len == 1
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert metadata.prefill.cu_seqlens_k.shape[0] == batch_spec.batch_size + 1


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[129, 257], query_lens=[129, 257], name="prefill_only"),
        BatchSpec(seq_lens=[130, 131], query_lens=[1, 1], name="decode_only"),
    ],
    ids=lambda case: case.name,
)
def test_indexer_metadata_builder(batch_spec: BatchSpec) -> None:
    device = torch.device("cpu")
    builder = _make_indexer_builder(device)
    common = _create_common_attn_metadata(batch_spec, block_size=128, device=device)
    metadata = builder.build(0, common)

    assert metadata.num_actual_tokens == sum(batch_spec.query_lens)
    assert metadata.num_decodes + metadata.num_prefills == batch_spec.batch_size
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
