# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
import vllm_ascend.attention.dsa_v1 as dsa_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_window import get_draft_swa_window


def _fake_vllm_config(window_size: int = 7, block_size: int = 5) -> SimpleNamespace:
    draft_hf_config = SimpleNamespace(dspark_block_size=block_size)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=block_size,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
    )
    hf_config = SimpleNamespace(
        sliding_window=window_size,
        num_attention_heads=8,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            get_head_size=lambda: 16,
        ),
        speculative_config=speculative_config,
    )


def _fake_common_metadata(causal: bool) -> SimpleNamespace:
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([15], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([15], dtype=torch.int32),
        seq_lens_cpu=None,
        positions=torch.arange(5, dtype=torch.int32),
        block_table_tensor=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=5,
        num_actual_tokens=5,
        causal=causal,
        attn_state=AscendAttentionState.SpecDecoding,
    )


def _expected_slot_ids(block_table: torch.Tensor, req_idx: int, cache_block_size: int, start: int, end: int):
    return torch.tensor(
        [
            int(block_table[req_idx, pos // cache_block_size]) * cache_block_size + pos % cache_block_size
            for pos in range(start, end)
        ],
        dtype=torch.int32,
    )


def test_dspark_dsa_window_does_not_encode_dspark_full_block():
    vllm_config = _fake_vllm_config(window_size=7, block_size=5)

    assert get_draft_swa_window(vllm_config, _fake_common_metadata(causal=False)) == (6, 0)
    assert get_draft_swa_window(vllm_config, _fake_common_metadata(causal=True)) == (6, 0)

    vllm_config.speculative_config.draft_model_config.hf_config.dspark_block_size = 0
    assert get_draft_swa_window(vllm_config, _fake_common_metadata(causal=False)) == (6, 0)


def test_dspark_dsa_decode_metadata_uses_noncausal_window(monkeypatch):
    captured = {}
    ori_cu_seqlens = torch.tensor([0, 15], dtype=torch.int32)
    cmp_cu_seqlens = torch.tensor([0], dtype=torch.int32)
    fallback_ori_cu_seqlens = torch.tensor([-1], dtype=torch.int32)
    fallback_cmp_cu_seqlens = torch.tensor([-2], dtype=torch.int32)

    def fake_metadata_op(**kwargs):
        captured.update(kwargs)
        return torch.tensor([123], dtype=torch.int32)

    def fake_cu_seqlens_ori_kv(cache, cache_key, seq_lens, num_decodes, zero_i32, fallback):
        assert cache is None
        assert cache_key == "draft_cu_seqlens_ori_kv"
        torch.testing.assert_close(seq_lens, torch.tensor([15], dtype=torch.int32))
        assert num_decodes == 1
        torch.testing.assert_close(zero_i32, torch.tensor([0], dtype=torch.int32))
        assert fallback is fallback_ori_cu_seqlens
        return ori_cu_seqlens

    def fake_cu_seqlens_cmp_kv(fallback):
        assert fallback is fallback_cmp_cu_seqlens
        return cmp_cu_seqlens

    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        dsa_v1,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache=False: (
            torch.zeros(positions.numel(), 1),
            torch.zeros(positions.numel(), 1),
        ),
    )
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: fake_metadata_op)
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _device: {})
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_decode_cu_seqlens_ori_kv", fake_cu_seqlens_ori_kv)
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", fake_cu_seqlens_cmp_kv)

    vllm_config = _fake_vllm_config()
    builder = SimpleNamespace(
        model_config=vllm_config.model_config,
        vllm_config=vllm_config,
        spec_slot_mapping=[torch.arange(5, dtype=torch.int32)],
        block_size=64,
        cu_seqlens_ori_kv=fallback_ori_cu_seqlens,
        cu_seqlens_cmp_kv=fallback_cmp_cu_seqlens,
        seqused_q=torch.empty(0, dtype=torch.int32),
        _zero_i32=torch.tensor([0], dtype=torch.int32),
    )

    metadata = dsa_v1.AscendDSAMetadataBuilder.build_decode_metadata_for_drafting(
        builder,
        draft_index=1,
        common_attn_metadata=_fake_common_metadata(causal=False),
        num_decodes=1,
        num_decode_tokens=5,
    )

    assert captured["ori_win_left"] == 11
    assert captured["ori_win_right"] == 0
    assert captured["cu_seqlens_ori_kv"] is ori_cu_seqlens
    assert captured["cu_seqlens_cmp_kv"] is cmp_cu_seqlens
    assert metadata.ori_win_left == 11
    assert metadata.ori_win_right == 0
    assert metadata.dspark_swa_indices.shape == (5, 1, 128)
    torch.testing.assert_close(metadata.dspark_swa_lens, torch.full((5,), 12, dtype=torch.int32))
    expected_slots = _expected_slot_ids(
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
        req_idx=0,
        cache_block_size=64,
        start=3,
        end=15,
    )
    for row in range(5):
        torch.testing.assert_close(metadata.dspark_swa_indices[row, 0, :12], expected_slots)
        assert torch.all(metadata.dspark_swa_indices[row, 0, 12:] == -1)


def test_dspark_dsa_metadata_uses_token_to_req_indices(monkeypatch):
    def fake_metadata_op(**kwargs):
        return torch.tensor([123], dtype=torch.int32)

    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        dsa_v1,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache=False: (
            torch.zeros(positions.numel(), 1),
            torch.zeros(positions.numel(), 1),
        ),
    )
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: fake_metadata_op)
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _device: {})
    monkeypatch.setattr(
        dsa_v1.DeviceOperator,
        "get_dsa_decode_cu_seqlens_ori_kv",
        lambda *args, **kwargs: torch.tensor([0, 12, 22], dtype=torch.int32),
    )
    monkeypatch.setattr(
        dsa_v1.DeviceOperator,
        "get_dsa_decode_cu_seqlens_cmp_kv",
        lambda *args, **kwargs: torch.empty(0, dtype=torch.int32),
    )

    vllm_config = _fake_vllm_config(window_size=4, block_size=2)
    common_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([12, 22], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([12, 22], dtype=torch.int32),
        seq_lens_cpu=None,
        positions=torch.tensor([10, 20, 11, 21], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        block_table_tensor=torch.tensor([[10], [20]], dtype=torch.int32),
        num_reqs=2,
        num_input_tokens=4,
        num_actual_tokens=4,
        causal=False,
        attn_state=AscendAttentionState.SpecDecoding,
    )
    builder = SimpleNamespace(
        model_config=vllm_config.model_config,
        vllm_config=vllm_config,
        spec_slot_mapping=[torch.arange(4, dtype=torch.int32)],
        block_size=64,
        cu_seqlens_ori_kv=torch.empty(0, dtype=torch.int32),
        cu_seqlens_cmp_kv=torch.empty(0, dtype=torch.int32),
        seqused_q=torch.empty(0, dtype=torch.int32),
        _zero_i32=torch.tensor([0], dtype=torch.int32),
    )

    metadata = dsa_v1.AscendDSAMetadataBuilder.build_decode_metadata_for_drafting(
        builder,
        draft_index=1,
        common_attn_metadata=common_metadata,
        num_decodes=2,
        num_decode_tokens=4,
    )

    torch.testing.assert_close(metadata.dspark_swa_lens, torch.full((4,), 6, dtype=torch.int32))
    expected_req0 = _expected_slot_ids(common_metadata.block_table_tensor, 0, 64, 6, 12)
    expected_req1 = _expected_slot_ids(common_metadata.block_table_tensor, 1, 64, 16, 22)
    for row in (0, 2):
        torch.testing.assert_close(metadata.dspark_swa_indices[row, 0, :6], expected_req0)
    for row in (1, 3):
        torch.testing.assert_close(metadata.dspark_swa_indices[row, 0, :6], expected_req1)


def test_dspark_dsa_prefill_metadata_slices_slot_mapping_from_token_start(monkeypatch):
    captured = {}

    def fake_metadata_op(**kwargs):
        captured.update(kwargs)
        return torch.tensor([789], dtype=torch.int32)

    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        dsa_v1,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache=False: (
            torch.zeros(positions.numel(), 1),
            torch.zeros(positions.numel(), 1),
        ),
    )
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: fake_metadata_op)
    monkeypatch.setattr(dsa_v1.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _device: {})

    vllm_config = _fake_vllm_config()
    common_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([7, 12], dtype=torch.int32),
        positions=torch.arange(5, dtype=torch.int32),
        block_table_tensor=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        causal=False,
    )
    builder = SimpleNamespace(
        model_config=vllm_config.model_config,
        vllm_config=vllm_config,
        spec_slot_mapping=[torch.arange(10, dtype=torch.int32)],
        block_size=64,
        seqused_q=torch.empty(0, dtype=torch.int32),
    )

    metadata = dsa_v1.AscendDSAMetadataBuilder.build_prefill_metadata_for_drafting(
        builder,
        draft_index=1,
        common_attn_metadata=common_metadata,
        reqs_start=1,
        tokens_start=2,
        num_prefill_tokens=3,
    )

    torch.testing.assert_close(metadata.slot_mapping, torch.tensor([2, 3, 4], dtype=torch.int32))
    assert captured["ori_win_left"] == 11
    assert captured["ori_win_right"] == 0
    assert metadata.dspark_swa_indices.shape == (3, 1, 128)
    torch.testing.assert_close(metadata.dspark_swa_lens, torch.full((3,), 10, dtype=torch.int32))
    expected_slots = _expected_slot_ids(
        common_metadata.block_table_tensor,
        req_idx=1,
        cache_block_size=64,
        start=2,
        end=12,
    )
    for row in range(3):
        torch.testing.assert_close(metadata.dspark_swa_indices[row, 0, :10], expected_slots)
        assert torch.all(metadata.dspark_swa_indices[row, 0, 10:] == -1)


def test_dspark_dsa_cp_req_metadata_uses_noncausal_window(monkeypatch):
    captured = {}

    def fake_metadata_op(**kwargs):
        captured.update(kwargs)
        return torch.tensor([456], dtype=torch.int32)

    def fake_local_token_metadata(
        *,
        num_reqs,
        num_input_tokens,
        input_positions,
        query_start_loc,
        seq_lens,
        use_cache,
        local_query_start_loc=None,
        local_seq_lens=None,
    ):
        del input_positions, use_cache, local_query_start_loc, local_seq_lens
        return (
            0,
            num_input_tokens,
            num_input_tokens,
            num_input_tokens,
            query_start_loc[: num_reqs + 1],
            seq_lens[:num_reqs],
            torch.zeros(num_input_tokens, 1),
            torch.zeros(num_input_tokens, 1),
        )

    monkeypatch.setattr(
        dsa_cp,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache=False: (
            torch.zeros(positions.numel(), 1),
            torch.zeros(positions.numel(), 1),
        ),
    )
    monkeypatch.setattr(dsa_cp.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: fake_metadata_op)
    monkeypatch.setattr(dsa_cp.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _device: {})
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

    vllm_config = _fake_vllm_config()
    common_metadata = _fake_common_metadata(causal=False)
    builder = SimpleNamespace(
        model_config=vllm_config.model_config,
        vllm_config=vllm_config,
        seq_lens=common_metadata.seq_lens,
        seq_lens_cpu=common_metadata._seq_lens_cpu,
        num_actual_tokens=common_metadata.num_actual_tokens,
        spec_slot_mapping=[torch.arange(5, dtype=torch.int32)],
        block_table=common_metadata.block_table_tensor,
        block_size=64,
        spec_local_query_start_loc=[torch.zeros(2, dtype=torch.int32)],
        spec_local_seq_lens=[torch.zeros(1, dtype=torch.int32)],
        seqused_q=torch.empty(0, dtype=torch.int32),
        cu_seqlens_ori_kv=torch.empty(0, dtype=torch.int32),
        cu_seqlens_cmp_kv=torch.empty(0, dtype=torch.int32),
        _zero_i32=torch.tensor([0], dtype=torch.int32),
        _build_local_token_metadata=fake_local_token_metadata,
    )

    metadata = dsa_cp.AscendDSACPMetadataBuilder.build_req_metadata_for_drafting(
        builder,
        draft_index=1,
        common_attn_metadata=common_metadata,
        input_positions=common_metadata.positions.long(),
        num_input_tokens=5,
    )

    assert captured["ori_win_left"] == 11
    assert captured["ori_win_right"] == 0
    assert metadata.ori_win_left == 11
    assert metadata.ori_win_right == 0
    assert metadata.dspark_swa_indices.shape == (5, 1, 128)
    torch.testing.assert_close(metadata.dspark_swa_lens, torch.full((5,), 12, dtype=torch.int32))
    expected_slots = _expected_slot_ids(
        common_metadata.block_table_tensor,
        req_idx=0,
        cache_block_size=64,
        start=3,
        end=15,
    )
    for row in range(5):
        torch.testing.assert_close(metadata.dspark_swa_indices[row, 0, :12], expected_slots)
        assert torch.all(metadata.dspark_swa_indices[row, 0, 12:] == -1)
