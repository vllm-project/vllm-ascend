# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
import vllm_ascend.attention.dsa_v1 as dsa_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_window import (
    build_dspark_swa_indices,
    build_dspark_swa_metadata_for_drafting,
)


def _vllm_config(window_size: int = 7, query_block_size: int = 5) -> SimpleNamespace:
    draft_hf_config = SimpleNamespace(dspark_block_size=query_block_size)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=window_size, num_attention_heads=8),
            get_head_size=lambda: 16,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=query_block_size,
            draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        ),
    )


def _metadata(*, causal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([15], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([15], dtype=torch.int32),
        seq_lens_cpu=None,
        positions=torch.arange(10, 15, dtype=torch.int32),
        block_table_tensor=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=5,
        num_actual_tokens=5,
        causal=causal,
        attn_state=AscendAttentionState.SpecDecoding,
    )


def _physical_slots(block_table: torch.Tensor, req_idx: int, start: int, end: int) -> torch.Tensor:
    return torch.tensor(
        [int(block_table[req_idx, position // 64]) * 64 + position % 64 for position in range(start, end)],
        dtype=torch.int32,
    )


def test_noncausal_draft_uses_explicit_full_block_visibility():
    common_metadata = _metadata()
    win_left, win_right, indices, lens = build_dspark_swa_metadata_for_drafting(
        _vllm_config(),
        common_metadata,
        torch.arange(5, dtype=torch.int32),
        cache_block_size=64,
    )

    assert (win_left, win_right) == (11, 0)
    assert indices is not None and lens is not None
    assert indices.shape == (5, 1, 128)
    torch.testing.assert_close(lens, torch.full((5,), 12, dtype=torch.int32))
    expected = _physical_slots(common_metadata.block_table_tensor, 0, 3, 15)
    for row in range(5):
        torch.testing.assert_close(indices[row, 0, :12], expected)
        assert torch.all(indices[row, 0, 12:] == -1)


def test_causal_draft_preserves_standard_window():
    win_left, win_right, indices, lens = build_dspark_swa_metadata_for_drafting(
        _vllm_config(),
        _metadata(causal=True),
        torch.arange(5, dtype=torch.int32),
        cache_block_size=64,
    )

    assert (win_left, win_right) == (6, 0)
    assert indices is None
    assert lens is None


def test_sparse_indices_cover_token_mapping_zero_length_and_padding():
    block_table = torch.tensor([[10], [-1], [20]], dtype=torch.int32)
    indices, lens = build_dspark_swa_indices(
        block_table=block_table,
        slot_mapping=torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([12, 0, 22], dtype=torch.int32),
        query_block_size=2,
        window_size=4,
        cache_block_size=64,
        num_query_tokens=5,
        token_to_req_indices=torch.tensor([0, 2, 0, 2, -1], dtype=torch.int32),
    )

    torch.testing.assert_close(lens, torch.tensor([6, 6, 6, 6, 0], dtype=torch.int32))
    expected_req0 = _physical_slots(block_table, 0, 6, 12)
    expected_req2 = _physical_slots(block_table, 2, 16, 22)
    for row in (0, 2):
        torch.testing.assert_close(indices[row, 0, :6], expected_req0)
    for row in (1, 3):
        torch.testing.assert_close(indices[row, 0, :6], expected_req2)
    assert torch.all(indices[4] == -1)


def test_sparse_indices_map_request_boundary_to_next_request():
    block_table = torch.tensor([[10], [20]], dtype=torch.int32)
    indices, lens = build_dspark_swa_indices(
        block_table=block_table,
        slot_mapping=torch.arange(4, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([12, 22], dtype=torch.int32),
        query_block_size=2,
        window_size=4,
        cache_block_size=64,
        num_query_tokens=4,
    )

    torch.testing.assert_close(lens, torch.full((4,), 6, dtype=torch.int32))
    expected_req0 = _physical_slots(block_table, 0, 6, 12)
    expected_req1 = _physical_slots(block_table, 1, 16, 22)
    for row in (0, 1):
        torch.testing.assert_close(indices[row, 0, :6], expected_req0)
    for row in (2, 3):
        torch.testing.assert_close(indices[row, 0, :6], expected_req1)


def test_sparse_indices_reject_incomplete_token_mapping():
    with pytest.raises(ValueError, match="token_to_req_indices"):
        build_dspark_swa_indices(
            block_table=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=None,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            query_block_size=2,
            window_size=4,
            cache_block_size=64,
            num_query_tokens=2,
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        )


def _patch_metadata_ops(monkeypatch, module, captured):
    monkeypatch.setattr(
        module,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache=False, draft_index=None: (
            torch.zeros(positions.numel(), 1),
            torch.zeros(positions.numel(), 1),
        ),
    )

    def metadata_op(**kwargs):
        captured.update(kwargs)
        return torch.ones(1024, dtype=torch.int32)

    monkeypatch.setattr(module.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: metadata_op)
    monkeypatch.setattr(module.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _device: {})


def test_dsa_builder_propagates_shared_metadata(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_metadata_ops(monkeypatch, dsa_v1, captured)
    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    config = _vllm_config()
    builder = SimpleNamespace(
        model_config=config.model_config,
        vllm_config=config,
        spec_slot_mapping=[torch.arange(5, dtype=torch.int32)],
        spec_sas_metadata=[torch.zeros(1024, dtype=torch.int32)],
        block_size=64,
        seqused_q=torch.empty(0, dtype=torch.int32),
        cu_seqlens_ori_kv=torch.empty(0, dtype=torch.int32),
        cu_seqlens_cmp_kv=torch.empty(0, dtype=torch.int32),
    )

    result = dsa_v1.AscendDSAMetadataBuilder.build_decode_metadata_for_drafting(
        cast(Any, builder),
        draft_index=1,
        common_attn_metadata=_metadata(),
        num_decodes=1,
        num_decode_tokens=5,
    )

    assert captured["ori_win_left"] == result.ori_win_left == 11
    assert captured["ori_win_right"] == result.ori_win_right == 0
    assert result.dspark_swa_indices is not None
    torch.testing.assert_close(result.dspark_swa_lens, torch.full((5,), 12, dtype=torch.int32))


def test_dsa_cp_builder_propagates_shared_metadata(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_metadata_ops(monkeypatch, dsa_cp, captured)
    monkeypatch.setattr(
        dsa_cp.DeviceOperator,
        "get_dsa_decode_cu_seqlens_ori_kv",
        lambda *args, **kwargs: torch.tensor([0, 5], dtype=torch.int32),
    )
    monkeypatch.setattr(
        dsa_cp.DeviceOperator,
        "get_dsa_decode_cu_seqlens_cmp_kv",
        lambda *args, **kwargs: torch.empty(0, dtype=torch.int32),
    )
    config = _vllm_config()
    common_metadata = _metadata()

    def local_metadata(**kwargs):
        num_reqs = kwargs["num_reqs"]
        return (
            3,
            6,
            3,
            6,
            torch.tensor([0, 2], dtype=torch.int32),
            kwargs["seq_lens"][:num_reqs],
            torch.zeros(3, 1),
            torch.zeros(3, 1),
        )

    builder = SimpleNamespace(
        model_config=config.model_config,
        vllm_config=config,
        seq_lens=common_metadata.seq_lens,
        seq_lens_cpu=common_metadata._seq_lens_cpu,
        num_actual_tokens=5,
        spec_slot_mapping=[torch.arange(5, dtype=torch.int32)],
        block_table=common_metadata.block_table_tensor,
        block_size=64,
        spec_local_query_start_loc=[torch.zeros(2, dtype=torch.int32)],
        spec_local_seq_lens=[torch.zeros(1, dtype=torch.int32)],
        seqused_q=torch.empty(0, dtype=torch.int32),
        cu_seqlens_ori_kv=torch.empty(0, dtype=torch.int32),
        cu_seqlens_cmp_kv=torch.empty(0, dtype=torch.int32),
        _zero_i32=torch.tensor([0], dtype=torch.int32),
        _build_local_token_metadata=local_metadata,
    )

    result = dsa_cp.AscendDSACPMetadataBuilder.build_req_metadata_for_drafting(
        cast(Any, builder),
        draft_index=1,
        common_attn_metadata=common_metadata,
        input_positions=common_metadata.positions.long(),
        num_input_tokens=5,
    )

    assert captured["ori_win_left"] == result.ori_win_left == 11
    assert captured["ori_win_right"] == result.ori_win_right == 0
    assert result.dspark_swa_indices is not None
    assert result.dspark_swa_indices.shape == (3, 1, 128)
    torch.testing.assert_close(result.dspark_swa_lens, torch.tensor([12, 12, 0], dtype=torch.int32))
    assert torch.all(result.dspark_swa_indices[-1] == -1)
