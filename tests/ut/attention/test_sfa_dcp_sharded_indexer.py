# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadataBuilder,
    _get_dcp_metadata_max_lens_from_cpu,
    dcp_local_to_global_indices,
    dcp_local_visible_counts,
    mask_dcp_inactive_local_candidates,
    merge_dcp_indexer_candidates,
)
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.utils import enable_sfa_dcp_sharded_indexer, get_sfa_dcp_indexer_cache_factor


def _sfa_config(**overrides):
    parallel_config = SimpleNamespace(
        tensor_parallel_size=16,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=128,
        prefill_context_parallel_size=1,
    )
    for key, value in overrides.pop("parallel", {}).items():
        setattr(parallel_config, key, value)
    return SimpleNamespace(
        additional_config={"enable_sfa_dcp_sharded_indexer": True, **overrides.pop("additional", {})},
        parallel_config=parallel_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=2048, index_head_dim=128),
            hf_config=SimpleNamespace(),
        ),
        kv_transfer_config=overrides.pop("kv_transfer_config", None),
    )


def test_device_operator_default_indexer_select_returns_indices_only(monkeypatch):
    calls = {}
    topk = torch.tensor([[3, 2, 1]], dtype=torch.int32)
    scores = torch.tensor([[0.3, 0.2, 0.1]], dtype=torch.bfloat16)

    def fake_li(**kwargs):
        calls.update(kwargs)
        return topk, scores

    monkeypatch.setattr("vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer", fake_li, raising=False)
    metadata = SimpleNamespace(block_table=torch.tensor([[9]], dtype=torch.int32))

    result = BaseDeviceAdaptor.indexer_select_post_process(
        torch.zeros(1, 1, 128),
        None,
        None,
        torch.ones(1, 1),
        (torch.zeros(1, 128, 1, 128),),
        0,
        1,
        metadata,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        False,
        True,
    )

    assert result is topk
    assert calls["sparse_mode"] == 3
    assert calls["block_table"] is metadata.block_table


def test_device_operator_selected_score_abi_preserves_dtype_and_sparse_mode(monkeypatch):
    custom_block_table = torch.tensor([[7]], dtype=torch.int32)
    scores = torch.tensor([[1.0, 0.5]], dtype=torch.float16)

    def fake_li(**kwargs):
        assert kwargs["sparse_mode"] == 0
        assert kwargs["block_table"] is custom_block_table
        assert kwargs["return_value"] is True
        return torch.tensor([[11, 10]], dtype=torch.int32), scores

    monkeypatch.setattr("vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer", fake_li, raising=False)
    result_indices, result_scores = BaseDeviceAdaptor.indexer_select_post_process(
        torch.zeros(1, 1, 128),
        None,
        None,
        torch.ones(1, 1),
        (torch.zeros(1, 128, 1, 128),),
        0,
        1,
        SimpleNamespace(block_table=torch.tensor([[9]], dtype=torch.int32)),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        False,
        True,
        return_selected_scores=True,
        sparse_mode=0,
        block_table=custom_block_table,
    )

    assert result_indices.dtype == torch.int32
    assert result_scores is scores
    assert result_scores.dtype == torch.float16


def test_device_operator_selected_scores_fail_closed_for_li_c8():
    with pytest.raises(NotImplementedError):
        BaseDeviceAdaptor.indexer_select_post_process(
            torch.zeros(1, 128),
            torch.ones(1),
            (1, 1, 128),
            torch.ones(1, 1),
            (torch.zeros(1, 128, 1, 128), torch.ones(1, 128, 1, 1)),
            0,
            1,
            SimpleNamespace(block_table=torch.tensor([[9]], dtype=torch.int32)),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            True,
            True,
            return_selected_scores=True,
        )


def test_sharded_indexer_selector_default_off_and_fallback(monkeypatch):
    cfg = _sfa_config()
    monkeypatch.setattr("vllm_ascend.utils.enable_dsa_cp", lambda: False)

    def uninitialized_config():
        raise RuntimeError("Ascend config is not initialized")

    monkeypatch.setattr("vllm_ascend.utils.get_ascend_config", uninitialized_config)
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16

    ascend_cfg = SimpleNamespace(
        enable_sfa_dcp_sharded_indexer=False,
        enable_sparse_li_c8=False,
        xlite_graph_config=SimpleNamespace(enabled=False),
        ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=False),
    )
    monkeypatch.setattr("vllm_ascend.utils.get_ascend_config", lambda: ascend_cfg)
    # The validated AscendConfig field is authoritative, not the raw dict.
    cfg.additional_config["enable_sfa_dcp_sharded_indexer"] = True
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16

    ascend_cfg.enable_sfa_dcp_sharded_indexer = True
    cfg.additional_config["enable_sfa_dcp_sharded_indexer"] = False
    assert enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 1

    cfg.kv_transfer_config = SimpleNamespace()
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16
    cfg.kv_transfer_config = None

    ascend_cfg.enable_sparse_li_c8 = True
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16
    ascend_cfg.enable_sparse_li_c8 = False

    ascend_cfg.ascend_compilation_config.enable_npugraph_ex = True
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16
    ascend_cfg.ascend_compilation_config.enable_npugraph_ex = False

    cfg.compilation_config = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL"))
    assert not enable_sfa_dcp_sharded_indexer(cfg)
    assert get_sfa_dcp_indexer_cache_factor(cfg) == 16
    del cfg.compilation_config

    unsupported = _sfa_config(parallel={"decode_context_parallel_size": 8})
    assert not enable_sfa_dcp_sharded_indexer(unsupported)
    assert get_sfa_dcp_indexer_cache_factor(unsupported) == 8


def test_local_visible_counts_for_c64k_q32_rank0():
    visible = torch.arange(65537, 65569, dtype=torch.int32)
    local = dcp_local_visible_counts(visible, dcp_rank=0, dcp_world_size=16, interleave_size=128)
    assert torch.equal(local, torch.arange(4097, 4129, dtype=torch.int32))


def test_local_to_global_map_uses_128_token_interleave():
    local = torch.tensor([0, 127, 128, 129, -1], dtype=torch.int32)
    global_indices = dcp_local_to_global_indices(local, dcp_rank=3, dcp_world_size=16, interleave_size=128)
    assert torch.equal(global_indices, torch.tensor([384, 511, 2432, 2433, -1], dtype=torch.int32))


def test_inactive_local_rows_publish_only_sentinels():
    indices = torch.tensor([[[7, 3, -1]], [[9, 2, -1]]], dtype=torch.int32)
    scores = torch.tensor([[[0.7, 0.3, float("-inf")]], [[0.9, 0.2, float("-inf")]]])
    masked_i, masked_s = mask_dcp_inactive_local_candidates(indices, scores, torch.tensor([4, 0], dtype=torch.int32))
    assert torch.equal(masked_i[0], indices[0])
    assert torch.equal(masked_i[1], torch.full_like(indices[1], -1))
    assert torch.equal(masked_s[0], scores[0])
    assert torch.isneginf(masked_s[1]).all()


def test_merge_unique_scores_matches_global_topk():
    indices = torch.tensor([[[0, 2, 4]], [[1, 3, 5]]], dtype=torch.int32)
    scores = torch.tensor([[[0.9, 0.7, 0.5]], [[0.8, 0.6, 0.4]]], dtype=torch.float16)
    merged = merge_dcp_indexer_candidates(indices, scores, topk=4)
    assert torch.equal(merged, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))


def test_merge_all_equal_scores_use_global_logical_index_order():
    indices = torch.tensor(
        [[[1284, 1287, 1286, 1285, 1282, 1281, 1408, 1283]], [[7, 3, 5, 1, 6, 4, 2, 0]]],
        dtype=torch.int32,
    )
    scores = torch.ones_like(indices, dtype=torch.float32)
    merged = merge_dcp_indexer_candidates(indices, scores, topk=8)
    assert torch.equal(merged, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32))


def test_merge_cutoff_ties_choose_smallest_global_indices():
    indices = torch.tensor([[[8, 2, 4, 6]], [[1, 3, 5, 7]]], dtype=torch.int32)
    scores = torch.tensor([[[0.4, 0.7, 0.5, 0.5]], [[0.7, 0.5, 0.5, 0.5]]], dtype=torch.float32)
    merged = merge_dcp_indexer_candidates(indices, scores, topk=4)
    assert torch.equal(merged, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))


def test_merge_partial_k_and_sentinel_padding_remain_stable():
    indices = torch.tensor([[[5, -1, -1]], [[1, 7, -1]]], dtype=torch.int32)
    scores = torch.tensor([[[0.5, float("-inf"), float("-inf")]], [[0.7, 0.5, float("-inf")]]], dtype=torch.float32)
    merged = merge_dcp_indexer_candidates(indices, scores, topk=5)
    assert torch.equal(merged, torch.tensor([[1, 5, 7, -1, -1]], dtype=torch.int32))


def test_merge_preserves_native_singleton_indexer_head_dimension():
    indices = torch.tensor([[[[0, 2, -1]], [[4, 6, -1]]], [[[1, 3, -1]], [[5, 7, -1]]]], dtype=torch.int32)
    scores = torch.tensor(
        [
            [[[0.9, 0.7, float("-inf")]], [[0.9, 0.7, float("-inf")]]],
            [[[0.8, 0.6, float("-inf")]], [[0.8, 0.6, float("-inf")]]],
        ],
        dtype=torch.float32,
    )
    merged = merge_dcp_indexer_candidates(indices, scores, topk=3)
    assert merged.shape == (2, 1, 3)
    assert torch.equal(merged[:, 0], torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32))


def test_multi_request_pseudo_rows_preserve_every_query_row():
    context_lens = torch.tensor([65536, 64], dtype=torch.int32)
    query_lens = torch.tensor([32, 4], dtype=torch.int32)
    row_offsets = torch.arange(1, int(query_lens.max().item()) + 1, dtype=torch.int32)
    global_visible = context_lens.unsqueeze(1) + row_offsets.unsqueeze(0)
    row_mask = row_offsets.unsqueeze(0) <= query_lens.unsqueeze(1)
    per_query_row = global_visible[row_mask]
    li_cum_query_lens = torch.arange(1, per_query_row.numel() + 1, dtype=torch.int32)
    local_visible = dcp_local_visible_counts(per_query_row, 0, 16, 128)

    assert li_cum_query_lens.numel() == 36
    assert torch.equal(li_cum_query_lens[:32], torch.arange(1, 33, dtype=torch.int32))
    assert torch.equal(local_visible[:32], torch.arange(4097, 4129, dtype=torch.int32))
    assert torch.equal(local_visible[32:], torch.tensor([65, 66, 67, 68], dtype=torch.int32))


def test_cpu_metadata_max_lens_ignores_padded_requests():
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 4, 12, 12, 99], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([20, 33, 99, 99], dtype=torch.int32),
    )

    assert _get_dcp_metadata_max_lens_from_cpu(common, num_reqs=2) == (8, 33)
    assert _get_dcp_metadata_max_lens_from_cpu(SimpleNamespace(), num_reqs=2) is None


def _make_replicated_indexer_builder(
    *,
    world_size: int,
    kernel_block_size: int,
    blocks_per_phys_block: int,
    local_cols: int,
    max_num_tokens: int,
):
    builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.kernel_block_size = kernel_block_size
    builder._pcp_active = False
    builder._dcp_sharded_indexer = False
    builder._dcp_replicated_indexer = True
    builder._dcp_world_size = world_size
    builder._dcp_rank = 0
    builder._dcp_interleave_size = 128
    builder._dcp_replicated_view_block_size = kernel_block_size
    builder._dcp_blocks_per_phys_block = blocks_per_phys_block
    builder._dcp_max_local_block_table_cols = local_cols
    replicated_cols = local_cols * world_size
    builder._dcp_replicated_block_table_buf = torch.empty((1, replicated_cols), dtype=torch.int32)
    builder._dcp_replicated_col_idx = torch.arange(replicated_cols, dtype=torch.int32)
    builder._dcp_replicated_slot_mapping_buf = torch.empty(max_num_tokens, dtype=torch.int32)
    return builder


def _make_sfa_dcp_reference_builder(
    *,
    world_size: int,
    kernel_block_size: int,
    blocks_per_phys_block: int,
    local_cols: int,
    max_num_tokens: int,
):
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = world_size
    builder.dcp_rank = 0
    builder.blocks_per_phys_block = blocks_per_phys_block
    builder.replicated_view_block_size = kernel_block_size
    builder.device = torch.device("cpu")
    replicated_cols = local_cols * world_size
    builder.block_table_replicated_view_buf = torch.empty((1, replicated_cols), dtype=torch.int32)
    builder.arange_buffer = torch.arange(replicated_cols, dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.empty(max_num_tokens, dtype=torch.int32)
    return builder


def test_independent_indexer_build_uses_replicated_dcp_view(monkeypatch):
    builder = _make_replicated_indexer_builder(
        world_size=2,
        kernel_block_size=4,
        blocks_per_phys_block=1,
        local_cols=4,
        max_num_tokens=8,
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=8,
        num_input_tokens=8,
        slot_mapping=torch.full((8,), -777, dtype=torch.int32),
        positions=torch.arange(24, 32, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        seq_lens=torch.tensor([32], dtype=torch.int32),
        block_table_tensor=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        group_len=MagicMock(),
        group_key_idx=MagicMock(),
        group_key_cache_idx=MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.indexer.get_cos_and_sin_mla",
        lambda positions, use_cache=True: (
            torch.zeros(positions.numel(), 1, 1, 8),
            torch.zeros(positions.numel(), 1, 1, 8),
        ),
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.indexer.get_ascend_config",
        lambda: SimpleNamespace(c8_reshape_optim_enabled=False),
    )

    metadata = builder.build(0, common)

    assert torch.equal(
        metadata.block_table,
        torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.slot_mapping,
        torch.tensor([104, 105, 106, 107, 108, 109, 110, 111], dtype=torch.int32),
    )
    assert not torch.equal(metadata.slot_mapping, common.slot_mapping)


def test_independent_replicated_view_matches_sfa_dcp_at_a1r3_24k_to_32k_boundary():
    # A1r3: scheduler block=2048, native LI block=128, DCP16. At the fault
    # boundary the next chunk covers global positions [24576,32768), so LI
    # needs replicated block-table columns 192..255 rather than the DCP-local
    # first 64 columns exposed by the generic common metadata.
    world_size = 16
    kernel_block_size = 128
    blocks_per_phys_block = 16
    local_cols = 64
    num_tokens = 8192
    local_block_table = torch.arange(local_cols, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([32768], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        positions=torch.arange(24576, 32768, dtype=torch.int64),
    )

    independent = _make_replicated_indexer_builder(
        world_size=world_size,
        kernel_block_size=kernel_block_size,
        blocks_per_phys_block=blocks_per_phys_block,
        local_cols=local_cols,
        max_num_tokens=num_tokens,
    )
    reference = _make_sfa_dcp_reference_builder(
        world_size=world_size,
        kernel_block_size=kernel_block_size,
        blocks_per_phys_block=blocks_per_phys_block,
        local_cols=local_cols,
        max_num_tokens=num_tokens,
    )

    independent_table = independent._build_dcp_replicated_block_table(local_block_table, seq_lens, 1)
    reference_table = reference._build_block_table_replicated_view(local_block_table, seq_lens)
    torch.testing.assert_close(independent_table, reference_table, rtol=0, atol=0)
    assert independent_table.shape == (1, 1024)

    independent_slots = independent._build_dcp_replicated_slot_mapping(common, independent_table)
    reference_slots = reference._build_slot_mapping_replicated_view(common, reference_table)
    torch.testing.assert_close(independent_slots, reference_slots, rtol=0, atol=0)
    assert independent_slots.numel() == num_tokens
    assert int(independent_slots.min()) >= 0


@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=True)
@patch("vllm_ascend.attention.indexer.get_dcp_group")
def test_metadata_builder_emits_q32_pseudo_rows_and_rank_local_slots(
    mock_get_dcp_group,
    _mock_enable,
    mock_cos_sin,
):
    mock_get_dcp_group.return_value.rank_in_group = 0
    mock_cos_sin.return_value = (torch.zeros(40, 1, 1, 8), torch.zeros(40, 1, 1, 8))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    cfg = _sfa_config()
    builder = AscendSFAIndexerMetadataBuilder(
        spec, ["model.layers.0.self_attn.indexer.k_cache"], cfg, torch.device("cpu")
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=32,
        num_input_tokens=32,
        slot_mapping=torch.arange(32, dtype=torch.int32),
        positions=torch.arange(65536, 65568, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
        seq_lens=torch.tensor([65568], dtype=torch.int32),
        block_table_tensor=torch.arange(512, dtype=torch.int32).view(1, 512),
        group_len=MagicMock(),
        group_key_idx=MagicMock(),
        group_key_cache_idx=MagicMock(),
    )
    with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_cfg:
        mock_cfg.return_value.c8_reshape_optim_enabled = False
        metadata = builder.build(0, common)

    assert metadata.dcp_sharded_indexer_enabled
    assert torch.equal(metadata.li_cum_query_lens, torch.arange(1, 33, dtype=torch.int32))
    assert torch.equal(metadata.local_visible_by_query, torch.arange(4097, 4129, dtype=torch.int32))
    assert metadata.dcp_local_block_table.shape == (32, 33)
    assert torch.equal(metadata.dcp_local_block_table, common.block_table_tensor[:, :33].expand(32, -1))
    assert torch.equal(metadata.dcp_local_token_mask, torch.ones(32, dtype=torch.bool))
    assert torch.equal(metadata.dcp_local_slot_mapping, torch.arange(4096, 4128, dtype=torch.int32))


@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=True)
@patch("vllm_ascend.attention.indexer.get_dcp_group")
def test_metadata_builder_keeps_eager_padding_out_of_dcp_cache_write(
    mock_get_dcp_group,
    _mock_enable,
    mock_cos_sin,
):
    mock_get_dcp_group.return_value.rank_in_group = 0
    mock_cos_sin.return_value = (torch.zeros(6, 1, 1, 8), torch.zeros(6, 1, 1, 8))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    cfg = _sfa_config(
        parallel={"tensor_parallel_size": 2, "decode_context_parallel_size": 2, "cp_kv_cache_interleave_size": 2}
    )
    builder = AscendSFAIndexerMetadataBuilder(
        spec, ["model.layers.0.self_attn.indexer.k_cache"], cfg, torch.device("cpu")
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=4,
        num_input_tokens=6,
        slot_mapping=torch.tensor([0, 1, 2, 3, -1, -1], dtype=torch.int32),
        positions=torch.arange(6, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([6], dtype=torch.int32),
        is_prefilling=torch.tensor([True], dtype=torch.bool),
        block_table_tensor=torch.arange(8, dtype=torch.int32).view(1, 8),
        group_len=MagicMock(),
        group_key_idx=MagicMock(),
        group_key_cache_idx=MagicMock(),
    )

    with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_cfg:
        mock_cfg.return_value.c8_reshape_optim_enabled = False
        metadata = builder.build(0, common)

    assert torch.equal(
        metadata.dcp_local_token_mask,
        torch.tensor([True, True, False, False, False, False]),
    )
    assert metadata.dcp_local_token_mask.shape == (common.num_input_tokens,)
    assert torch.equal(metadata.dcp_local_slot_mapping, torch.tensor([0, 1], dtype=torch.int32))
    assert torch.all(metadata.dcp_local_slot_mapping >= 0)

    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    k_li = torch.arange(common.num_input_tokens * 2, dtype=torch.float32).view(common.num_input_tokens, 2)
    gathered_k_li, gathered_scale, gathered_slots = backend._gather_cache_inputs(k_li, None, metadata)

    assert gathered_scale is None
    assert torch.equal(gathered_k_li, k_li[:2])
    assert torch.equal(gathered_slots, torch.tensor([0, 1], dtype=torch.int32))


@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=True)
@patch("vllm_ascend.attention.indexer.get_dcp_group")
def test_metadata_builder_uses_cpu_max_lens_without_tensor_item(
    mock_get_dcp_group,
    _mock_enable,
    mock_cos_sin,
    monkeypatch,
):
    mock_get_dcp_group.return_value.rank_in_group = 0
    mock_cos_sin.return_value = (torch.zeros(40, 1, 1, 8), torch.zeros(40, 1, 1, 8))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    cfg = _sfa_config()
    builder = AscendSFAIndexerMetadataBuilder(
        spec, ["model.layers.0.self_attn.indexer.k_cache"], cfg, torch.device("cpu")
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=32,
        num_input_tokens=32,
        slot_mapping=torch.arange(32, dtype=torch.int32),
        positions=torch.arange(65536, 65568, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 32], dtype=torch.int32),
        seq_lens=torch.tensor([65568], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([65568], dtype=torch.int32),
        is_prefilling=torch.tensor([True], dtype=torch.bool),
        block_table_tensor=torch.arange(512, dtype=torch.int32).view(1, 512),
        group_len=MagicMock(),
        group_key_idx=MagicMock(),
        group_key_cache_idx=MagicMock(),
    )

    def fail_item(_self):
        raise AssertionError("CPU metadata path must not depend on Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_cfg:
        mock_cfg.return_value.c8_reshape_optim_enabled = False
        metadata = builder.build(0, common)

    assert metadata.dcp_local_block_table.shape == (32, 33)
    assert torch.equal(metadata.li_cum_query_lens, torch.arange(1, 33, dtype=torch.int32))


@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=True)
@patch("vllm_ascend.attention.indexer.get_dcp_group")
def test_metadata_builder_falls_back_to_device_max_lens_without_cpu_metadata(
    mock_get_dcp_group,
    _mock_enable,
    mock_cos_sin,
):
    mock_get_dcp_group.return_value.rank_in_group = 0
    mock_cos_sin.return_value = (torch.zeros(40, 1, 1, 8), torch.zeros(40, 1, 1, 8))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    cfg = _sfa_config()
    builder = AscendSFAIndexerMetadataBuilder(
        spec, ["model.layers.0.self_attn.indexer.k_cache"], cfg, torch.device("cpu")
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=32,
        num_input_tokens=32,
        slot_mapping=torch.arange(32, dtype=torch.int32),
        positions=torch.arange(65536, 65568, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
        seq_lens=torch.tensor([65568], dtype=torch.int32),
        block_table_tensor=torch.arange(512, dtype=torch.int32).view(1, 512),
        group_len=MagicMock(),
        group_key_idx=MagicMock(),
        group_key_cache_idx=MagicMock(),
    )

    with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_cfg:
        mock_cfg.return_value.c8_reshape_optim_enabled = False
        metadata = builder.build(0, common)

    assert metadata.dcp_local_block_table.shape == (32, 33)
    assert torch.equal(metadata.local_visible_by_query, torch.arange(4097, 4129, dtype=torch.int32))


def test_sfa_cp_does_not_own_lightning_indexer_selection():
    import vllm_ascend.attention.context_parallel.sfa_cp as sfa_cp

    path = Path(sfa_cp.__file__)
    source = path.read_text()
    assert "indexer_select_post_process" not in source
    assert "npu_lightning_indexer" not in source


@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=True)
@patch("vllm_ascend.attention.indexer.get_dcp_group")
def test_r11d_all_local_rows_active_requires_cpu_authoritative_long_prefill(
    mock_get_dcp_group,
    _mock_enable,
    mock_cos_sin,
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    mock_get_dcp_group.return_value.rank_in_group = 0
    mock_cos_sin.return_value = (torch.zeros(40, 1, 1, 8), torch.zeros(40, 1, 1, 8))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    cfg = _sfa_config()
    builder = AscendSFAIndexerMetadataBuilder(
        spec,
        ["model.layers.0.self_attn.indexer.k_cache"],
        cfg,
        torch.device("cpu"),
    )

    def build(context: int, *, is_prefilling: bool):
        q = 32
        seq = context + q
        common = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=q,
            num_input_tokens=q,
            slot_mapping=torch.arange(q, dtype=torch.int32),
            positions=torch.arange(context, context + q, dtype=torch.int64),
            query_start_loc=torch.tensor([0, q], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q], dtype=torch.int32),
            seq_lens=torch.tensor([seq], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([seq], dtype=torch.int32),
            is_prefilling=torch.tensor([is_prefilling], dtype=torch.bool),
            block_table_tensor=torch.arange(512, dtype=torch.int32).view(1, 512),
            group_len=MagicMock(),
            group_key_idx=MagicMock(),
            group_key_cache_idx=MagicMock(),
        )
        with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_cfg:
            mock_cfg.return_value.c8_reshape_optim_enabled = False
            return builder.build(0, common)

    assert build(65536, is_prefilling=True).all_local_rows_active is True
    assert build(1024, is_prefilling=True).all_local_rows_active is False
    assert build(65536, is_prefilling=False).all_local_rows_active is False
