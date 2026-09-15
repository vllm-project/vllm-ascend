# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPMetadataBuilder
from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)

_KERNEL_BLOCK_SIZE = 128


def _make_builder(
    pcp_size: int = 1,
    dcp_size: int = 1,
    dsa_cp: bool = False,
    num_speculative_tokens: int | None = None,
) -> AscendSFAIndexerMetadataBuilder:
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
    vllm_config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=4,
    )
    vllm_config.scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    vllm_config.speculative_config = (
        None if num_speculative_tokens is None else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    with (
        patch(
            "vllm_ascend.attention.indexer.select_common_block_size",
            return_value=_KERNEL_BLOCK_SIZE,
        ),
        patch("vllm_ascend.attention.indexer.enable_dsa_cp", return_value=dsa_cp),
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        query_start_loc_cpu=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        max_query_len=2,
        is_prefilling=torch.tensor([True, True]),
        context_parallel_metadata=None,
        block_table_tensor=torch.arange(6).view(3, 2),
        group_len=MagicMock(name="group_len"),
        group_key_idx=MagicMock(name="group_key_idx"),
        group_key_cache_idx=MagicMock(name="group_key_cache_idx"),
    )

    def replace(**changes):
        values = vars(metadata).copy()
        values.pop("replace", None)
        values.update(changes)
        return SimpleNamespace(**values)

    metadata.replace = replace
    return metadata


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "query_starts,seq_lens,num_actual_tokens",
    [([0, 3, 5], [13, 15], 5), ([0, 4, 8, 12], [12, 13, 0], 8), ([0], [], 0)],
)
def test_sfa_indexer_dsa_cp_matches_sfa_with_independent_buffers(world_size, query_starts, seq_lens, num_actual_tokens):
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor(query_starts, dtype=torch.int32)
    common.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    common.num_reqs = len(seq_lens)
    common.num_input_tokens = query_starts[-1]
    common.num_actual_tokens = num_actual_tokens
    cos = torch.arange(common.num_input_tokens * 2, dtype=torch.float32).view(-1, 1, 1, 2)
    sin = cos + 100
    slots = torch.arange(common.num_input_tokens, dtype=torch.int32)
    for rank in range(world_size):
        sfa_builder = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
        sfa_builder.dsa_cp_actual_seq_lengths_query = torch.zeros(4, dtype=torch.int32)
        sfa_builder.dsa_cp_actual_seq_lengths_key = torch.zeros(4, dtype=torch.int32)
        indexer_builder = _make_builder(dsa_cp=True)
        indexer_builder.dsa_cp_world_size = world_size
        group = SimpleNamespace(world_size=world_size, rank_in_group=rank)
        with (
            patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group),
            patch("vllm_ascend.attention.indexer.get_tp_group", return_value=group),
        ):
            sfa_cos, sfa_sin, _, extra = sfa_builder._prepare_parallel_metadata(
                common, cos, sin, slots, common.query_start_loc[1:], common.seq_lens, draft_index=None
            )
            local_cos, local_sin, query, key = indexer_builder._build_dsa_cp_parallel_metadata(common, cos, sin, "test")
        context = extra["dsa_cp_context"]
        torch.testing.assert_close(local_cos, sfa_cos)
        torch.testing.assert_close(local_sin, sfa_sin)
        torch.testing.assert_close(query, context.actual_seq_lengths_query)
        torch.testing.assert_close(key, context.actual_seq_lengths_key)
        if common.num_reqs:
            assert query.data_ptr() != context.actual_seq_lengths_query.data_ptr()
            assert key.data_ptr() != context.actual_seq_lengths_key.data_ptr()
            query.fill_(-1)
            key.fill_(-1)
            assert torch.all(context.actual_seq_lengths_query >= 0)
            assert torch.all(context.actual_seq_lengths_key >= 0)


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
    assert metadata.group_len is None
    assert metadata.group_key_idx is None
    assert metadata.group_key_cache_idx is None
    assert torch.equal(metadata.actual_seq_lengths_query, common.query_start_loc[1:3])
    assert torch.equal(metadata.actual_seq_lengths_key, common.seq_lens[:2])
    assert metadata.num_decode_tokens == 0

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_uses_graph_shape_for_dspark_adaptive(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (
        torch.zeros(5, 1, 1, 8),
        torch.zeros(5, 1, 1, 8),
    )
    common = _make_common_metadata()

    builder = _make_builder(num_speculative_tokens=7)
    builder.speculative_config.method = "dspark"
    builder.speculative_config.enable_adaptive_verification = True
    metadata = builder.build(0, common)

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions)
    assert metadata.cos.shape[0] == common.positions.shape[0]
    assert metadata.sin.shape[0] == common.positions.shape[0]


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


@pytest.mark.parametrize("dcp_size,expected_decode_tokens", [(1, 0), (2, 1)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_pcp_decode_boundary(
    mock_cos_sin, mock_get_ascend_config, dcp_size, expected_decode_tokens
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(4, 1, 1, 8), torch.zeros(4, 1, 1, 8))
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 3
    common.is_prefilling = torch.tensor([False, True])

    metadata = _make_builder(pcp_size=2, dcp_size=dcp_size).build(0, common)

    # Preserve the ordinary SFA default and the DCP leading-decode count.
    assert metadata.num_decode_tokens == expected_decode_tokens


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
        metadata.group_len,
        metadata.group_key_idx,
        metadata.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )
    assert metadata.group_len is not common.group_len
    assert metadata.group_len.numel() == metadata.slot_mapping.numel()


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_owns_replicated_dcp_addresses(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    common = _make_common_metadata()

    metadata = _make_builder(dcp_size=2).build(0, common)

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([0, 1, 512, 513], dtype=torch.int32),
    )
    # The builder derives the replicated view without changing the local view
    # consumed by the SFA cache backend.
    torch.testing.assert_close(common.slot_mapping, torch.tensor([1, 2, 3, 4, 5]))
    torch.testing.assert_close(common.block_table_tensor, torch.arange(6).view(3, 2))


@pytest.mark.parametrize("dcp_size,expected_slots", [(1, [1, 2, 3, -1]), (2, [0, 1, 512, -1])])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.get_tp_group")
def test_sfa_indexer_metadata_builder_pads_dsa_slots(
    mock_get_tp_group, mock_cos_sin, mock_get_ascend_config, dcp_size, expected_slots
):
    mock_get_tp_group.return_value.world_size = 4
    mock_get_tp_group.return_value.rank_in_group = 0
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(3, 1, 1, 8), torch.zeros(3, 1, 1, 8))
    common = _make_common_metadata()
    common.num_actual_tokens = 3
    common.num_input_tokens = 3
    common.query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.positions = torch.tensor([0, 1, 0], dtype=torch.int64)
    builder = _make_builder(dcp_size=dcp_size, dsa_cp=True)

    first_metadata = builder.build(0, common)
    second_metadata = builder.build(0, common)

    torch.testing.assert_close(second_metadata.slot_mapping, torch.tensor(expected_slots, dtype=torch.int32))
    assert first_metadata.slot_mapping.data_ptr() == second_metadata.slot_mapping.data_ptr()
    torch.testing.assert_close(second_metadata.actual_seq_lengths_query, torch.tensor([1, 1], dtype=torch.int32))
    torch.testing.assert_close(second_metadata.actual_seq_lengths_key, torch.tensor([4, 0], dtype=torch.int32))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.get_tp_group")
def test_sfa_indexer_draft_metadata_owns_per_step_dsa_buffers(
    mock_get_tp_group,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_tp_group.return_value.rank_in_group = 0
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (
        torch.zeros(3, 1, 1, 8),
        torch.zeros(3, 1, 1, 8),
    )
    builder = _make_builder(dsa_cp=True, num_speculative_tokens=3)
    common = _make_common_metadata()
    common.num_actual_tokens = 3
    common.num_input_tokens = 3
    common.query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.positions = torch.tensor([0, 1, 0], dtype=torch.int64)

    step_one = builder.build_for_drafting(common, 1)
    step_one_slots = step_one.slot_mapping.clone()
    common.slot_mapping = torch.tensor([9, 10, 11], dtype=torch.int32)
    step_two = builder.build_for_drafting(common, 2)

    assert step_one.slot_mapping.data_ptr() != step_two.slot_mapping.data_ptr()
    assert step_one.cos.data_ptr() != step_two.cos.data_ptr()
    assert step_one.actual_seq_lengths_query.data_ptr() != step_two.actual_seq_lengths_query.data_ptr()
    torch.testing.assert_close(step_one.slot_mapping, step_one_slots)
    torch.testing.assert_close(
        step_two.slot_mapping,
        torch.tensor([9, 10, 11, -1], dtype=torch.int32),
    )


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_draft_metadata_owns_per_step_dcp_buffers(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (
        torch.zeros(5, 1, 1, 8),
        torch.zeros(5, 1, 1, 8),
    )
    builder = _make_builder(dcp_size=2, num_speculative_tokens=3)
    common = _make_common_metadata()

    step_one = builder.build_for_drafting(common, 1)
    step_one_slots = step_one.slot_mapping.clone()
    step_one_blocks = step_one.block_table.clone()
    # The proposer owns one persistent slot tensor per logical draft step;
    # its address is also the key for every derived metadata buffer.
    common.slot_mapping = common.slot_mapping.clone()
    common.positions = torch.tensor([2, 3, 2, 3, 9])
    step_two = builder.build_for_drafting(common, 2)

    assert step_one.slot_mapping.data_ptr() != step_two.slot_mapping.data_ptr()
    assert step_one.block_table.data_ptr() != step_two.block_table.data_ptr()
    torch.testing.assert_close(step_one.slot_mapping, step_one_slots)
    torch.testing.assert_close(step_one.block_table, step_one_blocks)


@pytest.mark.parametrize("dsa_cp", [False, True])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.get_tp_group")
def test_sfa_indexer_graph_capture_owns_stable_per_step_buffers(
    mock_get_tp_group, mock_cos_sin, mock_get_ascend_config, dsa_cp
):
    mock_get_tp_group.return_value.rank_in_group = 0
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.side_effect = [
        (torch.full((4, 1, 1, 8), value), torch.full((4, 1, 1, 8), -value)) for value in range(4)
    ]
    builder = _make_builder(dsa_cp=dsa_cp, num_speculative_tokens=3)
    common = _make_common_metadata()

    first = builder.build_for_graph_capture(common)
    first_runtime = builder.build(common_prefix_len=0, common_attn_metadata=common)
    common.slot_mapping = common.slot_mapping.clone()
    second = builder.build_for_graph_capture(common)
    second_runtime = builder.build_for_drafting(common, draft_index=1)

    for field in ("slot_mapping", "cos", "sin"):
        assert getattr(first, field).data_ptr() != getattr(second, field).data_ptr()
        assert getattr(first, field).data_ptr() == getattr(first_runtime, field).data_ptr()
        assert getattr(second, field).data_ptr() == getattr(second_runtime, field).data_ptr()
    torch.testing.assert_close(first.cos, torch.ones_like(first.cos))
    torch.testing.assert_close(first.sin, -torch.ones_like(first.sin))
    torch.testing.assert_close(second.cos, torch.full_like(second.cos, 3))
    torch.testing.assert_close(second.sin, torch.full_like(second.sin, -3))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_builds_pcp_dcp_slots_and_c8_groups(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    common = _make_common_metadata()
    global_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([384], dtype=torch.int32),
        positions=torch.tensor([0, 128, 256], dtype=torch.int64),
        is_prefilling_np=torch.tensor([True]),
    )
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(torch.tensor([[10, 11]], dtype=torch.int32),),
        padded_gather_idx=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        gathered_kv_write_mask=torch.tensor([True, True, True, False]),
    )

    metadata = _make_builder(pcp_size=2, dcp_size=2).build(
        0,
        common,
        pcp_context=pcp_context,
        pcp_cache_group_idx=0,
    )

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([2816, 2560, 2688, -1], dtype=torch.int32),
    )
    mock_store_kv_block_metadata.assert_called_once_with(
        metadata.slot_mapping,
        metadata.group_len,
        metadata.group_key_idx,
        metadata.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )
    assert metadata.group_len is not common.group_len
    assert metadata.group_len.numel() == metadata.slot_mapping.numel()
