# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)

_KERNEL_BLOCK_SIZE = 128


def _make_builder(pcp_size: int = 1, dcp_size: int = 1) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_topk=2048),
        hf_config=SimpleNamespace(),
        max_model_len=1024,
    )
    vllm_config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=16,
        max_num_seqs=4,
    )
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
    vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 4
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "vllm_ascend.attention.indexer.select_common_block_size",
                return_value=_KERNEL_BLOCK_SIZE,
            )
        )
        if dcp_size > 1:
            stack.enter_context(
                patch(
                    "vllm_ascend.attention.indexer.get_dcp_group",
                    return_value=SimpleNamespace(world_size=dcp_size, rank_in_group=0),
                )
            )
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        block_table_tensor=torch.arange(6).view(3, 2),
        group_len=MagicMock(name="group_len"),
        group_key_idx=MagicMock(name="group_key_idx"),
        group_key_cache_idx=MagicMock(name="group_key_cache_idx"),
    )


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_kernel_metadata(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    cos = torch.zeros(5, 1, 1, 8)
    sin = torch.zeros(5, 1, 1, 8)
    mock_cos_sin.return_value = (cos, sin)

    builder = _make_builder()
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(MagicMock(), MagicMock()) is AttentionCGSupport.UNIFORM_BATCH

    common = _make_common_metadata()
    metadata = builder.build(0, common)

    assert isinstance(metadata, AscendSFAIndexerMetadata)
    assert metadata.num_actual_tokens == 4
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert metadata.group_len is common.group_len
    assert metadata.group_key_idx is common.group_key_idx
    assert metadata.group_key_cache_idx is common.group_key_cache_idx

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.slot_mapping is common.slot_mapping


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_primes_reshape_optim(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder()
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    mock_store_kv_block_metadata.assert_called_once_with(
        metadata.slot_mapping,
        common.group_len,
        common.group_key_idx,
        common.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_skips_local_reshape_groups_under_dcp(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    metadata = _make_builder(dcp_size=2).build(0, _make_common_metadata())

    mock_store_kv_block_metadata.assert_not_called()
    assert metadata.group_len is None
    assert metadata.group_key_idx is None
    assert metadata.group_key_cache_idx is None


def _make_dcp_builder(*, pcp_active: bool = False, dsa_cp_active: bool = False) -> AscendSFAIndexerMetadataBuilder:
    builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    builder.kernel_block_size = 4
    builder._pcp_active = pcp_active
    builder._dcp_active = True
    builder._dsa_cp_active = dsa_cp_active
    builder.dcp_size = 2
    builder.dcp_rank = 0
    builder.blocks_per_phys_block = 1
    builder.replicated_view_block_size = 4
    builder.max_local_block_table_cols = 2
    builder.block_table_replicated_view_buf = torch.empty((4, 4), dtype=torch.int32)
    builder.arange_buffer = torch.arange(4, dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.empty(16, dtype=torch.int32)
    if pcp_active:
        builder.pcp_indexer_slot_mapping_buf = torch.empty(16, dtype=torch.int32)
    return builder


def _make_dcp_common_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=6,
        num_input_tokens=6,
        slot_mapping=torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        block_table_tensor=torch.tensor([[10, 11]], dtype=torch.int32),
        group_len=torch.tensor([1]),
        group_key_idx=torch.tensor([2]),
        group_key_cache_idx=torch.tensor([3]),
    )


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_owns_replicated_dcp_cache_view(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(6, 1, 1, 8), torch.zeros(6, 1, 1, 8))
    builder = _make_dcp_builder()

    metadata = builder.build(0, _make_dcp_common_metadata())

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[20, 21, 22, 23]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([80, 81, 82, 83, 84, 85], dtype=torch.int32),
    )
    assert metadata.group_len is None
    assert metadata.group_key_idx is None
    assert metadata.group_key_cache_idx is None


def test_sfa_indexer_metadata_builder_builds_pcp_ordered_dcp_slots_independently():
    builder = _make_dcp_builder(pcp_active=True)
    common = _make_dcp_common_metadata()
    global_common = _make_dcp_common_metadata()
    common.replace = MagicMock(return_value=global_common)
    global_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=6,
        query_start_loc=global_common.query_start_loc,
        seq_lens=global_common.seq_lens,
        positions=global_common.positions,
        is_prefilling_np=torch.tensor([True]).numpy(),
    )
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(global_common.block_table_tensor,),
        padded_gather_idx=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        gathered_kv_write_mask=torch.tensor([True, True, True, False]),
    )

    slots = builder._build_slot_mapping(common, pcp_context, 0)

    torch.testing.assert_close(
        slots,
        torch.tensor([82, 80, 81, -1], dtype=torch.int32),
    )
    common.replace.assert_called_once_with(
        query_start_loc=global_batch.query_start_loc,
        seq_lens=global_batch.seq_lens[:1],
        num_reqs=1,
        num_actual_tokens=6,
        num_input_tokens=6,
        positions=global_batch.positions,
        block_table_tensor=global_common.block_table_tensor,
    )


@patch("vllm_ascend.attention.indexer.get_tp_group")
def test_sfa_indexer_metadata_builder_pads_dcp_slots_for_dsa_cp(mock_get_tp_group):
    mock_get_tp_group.return_value.world_size = 4
    builder = _make_dcp_builder(dsa_cp_active=True)
    common = _make_dcp_common_metadata()
    common.num_actual_tokens = 3
    common.num_input_tokens = 3
    common.positions = common.positions[:3]
    common.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)

    slots = builder._build_slot_mapping(common, None, None)

    torch.testing.assert_close(
        slots,
        torch.tensor([80, 81, 82, -1], dtype=torch.int32),
    )
