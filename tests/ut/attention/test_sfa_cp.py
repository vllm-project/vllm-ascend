# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
    AscendSFADSACPImpl,
    AscendSFADSACPMetadata,
    AscendSFADSACPMetadataBuilder,
    AscendSFADSADCPImpl,
    AscendSFADSADCPMetadata,
    AscendSFADSADCPMetadataBuilder,
    AscendSFAPCPDCPImpl,
    AscendSFAPCPDCPMetadataBuilder,
    AscendSFAPCPImpl,
    resolve_sfa_impl,
    resolve_sfa_metadata_builder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)


def test_sfa_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendSFADCPImpl, AscendSFAImpl)
    assert AscendSFADCPImpl.supports_mtp_with_cp_non_trivial_interleave_size
    assert AscendSFADCPImpl.can_return_lse_for_decode
    assert issubclass(
        AscendSFADCPMetadataBuilder,
        AscendSFAMetadataBuilder,
    )
    assert "dcp_context" not in {field.name for field in fields(AscendSFAMetadata)}
    assert "dcp_context" in {field.name for field in fields(AscendSFADCPMetadata)}
    assert "dsa_cp_context" not in {field.name for field in fields(AscendSFAMetadata)}
    assert "dsa_cp_context" in {field.name for field in fields(AscendSFADSACPMetadata)}
    assert issubclass(AscendSFADSADCPImpl, AscendSFADCPImpl)
    assert issubclass(AscendSFADSADCPImpl, AscendSFADSACPImpl)
    assert issubclass(AscendSFADSADCPMetadataBuilder, AscendSFADCPMetadataBuilder)
    assert issubclass(AscendSFADSADCPMetadataBuilder, AscendSFADSACPMetadataBuilder)
    assert issubclass(AscendSFADSADCPMetadata, AscendSFADCPMetadata)
    impl_mro = AscendSFADSADCPImpl.__mro__
    builder_mro = AscendSFADSADCPMetadataBuilder.__mro__
    assert impl_mro.index(AscendSFADCPImpl) < impl_mro.index(AscendSFADSACPImpl)
    assert builder_mro.index(AscendSFADCPMetadataBuilder) < builder_mro.index(AscendSFADSACPMetadataBuilder)
    assert issubclass(AscendSFAPCPDCPImpl, AscendSFADCPImpl)
    assert issubclass(AscendSFAPCPDCPImpl, AscendSFAPCPImpl)
    assert issubclass(AscendSFAPCPDCPMetadataBuilder, AscendSFADCPMetadataBuilder)
    pcp_dcp_mro = AscendSFAPCPDCPImpl.__mro__
    assert pcp_dcp_mro.index(AscendSFADCPImpl) < pcp_dcp_mro.index(AscendSFAPCPImpl)


def test_sfa_cp_four_mode_resolution() -> None:
    expected = {
        (False, False): (AscendSFAMetadataBuilder, AscendSFAImpl),
        (True, False): (AscendSFADSACPMetadataBuilder, AscendSFADSACPImpl),
        (False, True): (AscendSFADCPMetadataBuilder, AscendSFADCPImpl),
        (True, True): (AscendSFADSADCPMetadataBuilder, AscendSFADSADCPImpl),
    }
    for flags, classes in expected.items():
        with (
            patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_dsa_cp", return_value=flags[0]),
            patch(
                "vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_replicated_indexer",
                return_value=flags[1],
            ),
        ):
            assert resolve_sfa_metadata_builder() is classes[0]
            assert resolve_sfa_impl() is classes[1]


def test_sfa_pcp_resolution_for_mrv2_config() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
    )
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_dsa_cp", return_value=False),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_replicated_indexer",
            return_value=False,
        ),
    ):
        assert resolve_sfa_impl(vllm_config) is AscendSFAPCPImpl


def test_sfa_pcp_dcp_builds_pcp_ordered_indexer_slots_with_receiver_local_blocks() -> None:
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    builder.pcp_indexer_slot_mapping_buf = torch.empty(8, dtype=torch.int32)
    local_block_table = torch.tensor([[11, 12]], dtype=torch.int32)
    replicated_block_table = torch.tensor([[22, 23, 24, 25]], dtype=torch.int32)
    global_slot_mapping = torch.tensor([100, 101, 102], dtype=torch.int32)
    builder._get_dcp_local_block_table = Mock(return_value=local_block_table)
    builder._build_block_table_replicated_view = Mock(return_value=replicated_block_table)
    builder._build_slot_mapping_replicated_view = Mock(return_value=global_slot_mapping)
    global_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(local_block_table,),
        padded_gather_idx=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        gathered_kv_write_mask=torch.tensor([True, True, True, False]),
    )
    global_common = SimpleNamespace(seq_lens=global_batch.seq_lens)

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.replace",
        return_value=global_common,
    ) as replace_metadata:
        result = builder._build_pcp_ordered_indexer_slot_mapping(
            object(),
            pcp_context,
            0,
        )

    torch.testing.assert_close(
        result,
        torch.tensor([102, 100, 101, -1], dtype=torch.int32),
    )
    replace_metadata.assert_called_once()
    builder._get_dcp_local_block_table.assert_called_once_with(
        local_block_table,
        1,
    )
    builder._build_block_table_replicated_view.assert_called_once_with(
        local_block_table,
        global_batch.seq_lens,
    )


def test_sfa_dcp_compact_kv_table_uses_logical_dcp_rank_order() -> None:
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = 8
    builder.dcp_collective_rank_order = torch.tensor(
        [0, 4, 1, 5, 2, 6, 3, 7],
        dtype=torch.int32,
    )
    dcp_block_table = torch.tensor([[5, 9]], dtype=torch.int32)

    valid_block_ids, block_table = builder._build_compact_kv_gather_metadata(dcp_block_table)

    torch.testing.assert_close(
        valid_block_ids,
        torch.tensor([5, 9], dtype=torch.int32),
    )
    torch.testing.assert_close(
        block_table,
        torch.tensor(
            [[0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15]],
            dtype=torch.int32,
        ),
    )


def test_sfa_pcp_dcp_builder_allows_decode_graph_metadata_without_pcp_context() -> None:
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    common_attn_metadata = SimpleNamespace()
    expected = object()

    with patch.object(
        AscendSFADCPMetadataBuilder,
        "build",
        autospec=True,
        return_value=expected,
    ) as dcp_build:
        result = builder.build(0, common_attn_metadata)

    assert result is expected
    dcp_build.assert_called_once_with(builder, 0, common_attn_metadata, False)


def test_sfa_pcp_dcp_only_overrides_main_cache_slot_mapping() -> None:
    impl = AscendSFAPCPDCPImpl.__new__(AscendSFAPCPDCPImpl)
    attn_metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    attn_metadata.num_prefills = 1
    attn_metadata.num_decode_tokens = 0
    attn_metadata.num_input_tokens = 2
    main_slots = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    attn_metadata.dcp_context = SimpleNamespace(
        slot_mapping=main_slots,
    )
    kv_no_split = torch.zeros(2, 3)
    cos = torch.zeros(2, 1)
    sin = torch.zeros(2, 1)
    kv_cache = (torch.empty(1), torch.empty(1))

    with patch.object(
        AscendSFAPCPImpl,
        "exec_kv",
        autospec=True,
        return_value="written",
    ) as pcp_exec_kv:
        result = impl.exec_kv(
            kv_no_split,
            cos,
            sin,
            kv_cache,
            torch.tensor([-1, -1]),
            attn_metadata,
        )

    assert result == "written"
    pcp_exec_kv.assert_called_once_with(
        impl,
        kv_no_split,
        cos,
        sin,
        kv_cache,
        main_slots,
        attn_metadata,
    )


def test_sfa_pcp_gathers_main_kv_before_base_cache_write() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    attn_metadata = SimpleNamespace(num_decode_tokens=1)
    kv_no_split = torch.arange(6, dtype=torch.float32).view(2, 3)
    cos = torch.arange(2, dtype=torch.float32).view(2, 1)
    sin = cos + 10
    slots = torch.tensor([4, 5], dtype=torch.int64)
    gathered_kv = torch.arange(12, dtype=torch.float32).view(4, 3)
    gathered_cos = torch.arange(4, dtype=torch.float32).view(4, 1)
    gathered_sin = gathered_cos + 10
    gathered_slots = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    kv_cache = (torch.empty(1), torch.empty(1))

    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs",
            return_value=((gathered_kv, gathered_cos, gathered_sin), gathered_slots),
        ) as gather,
        patch.object(AscendSFAImpl, "exec_kv", autospec=True, return_value="written") as base_exec_kv,
    ):
        result = impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)

    assert result == "written"
    gather.assert_called_once_with((kv_no_split, cos, sin), slots, 1)
    base_exec_kv.assert_called_once_with(
        impl,
        gathered_kv,
        gathered_cos,
        gathered_sin,
        kv_cache,
        gathered_slots,
        attn_metadata,
    )


def test_sfa_pcp_gathers_indexer_kv_with_its_slot_mapping() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    attn_metadata = SimpleNamespace(num_decode_tokens=1)
    k_li = torch.arange(8, dtype=torch.float32).view(2, 4)
    k_li_scale = torch.ones(2, 1, dtype=torch.float32)
    slots = torch.tensor([7, 8], dtype=torch.int64)
    gathered_k_li = torch.arange(16, dtype=torch.float32).view(4, 4)
    gathered_scale = torch.full((4, 1), 2.0)
    gathered_slots = torch.tensor([1, 2, 7, 8], dtype=torch.int64)
    kv_cache = (torch.empty(1), torch.empty(1), torch.empty(1))

    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs",
            return_value=((gathered_k_li, gathered_scale), gathered_slots),
        ) as gather,
        patch.object(AscendSFAImpl, "_write_indexer_cache", autospec=True) as base_write,
    ):
        impl._write_indexer_cache(k_li, k_li_scale, slots, kv_cache, attn_metadata)

    gather.assert_called_once_with((k_li, k_li_scale), slots, 1)
    base_write.assert_called_once_with(
        impl,
        gathered_k_li,
        gathered_scale,
        gathered_slots,
        kv_cache,
        attn_metadata,
    )


def test_sfa_cp_query_gather_axis_follows_composed_layout() -> None:
    dcp_impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    combined_impl = AscendSFADSADCPImpl.__new__(AscendSFADSADCPImpl)
    assert dcp_impl._parallel_query_gather_dim() == 1
    assert combined_impl._parallel_query_gather_dim() == 0


def test_sfa_dsa_cp_builder_shards_tokens_and_sequence_lengths() -> None:
    builder = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
    builder.actual_seq_lengths_query = torch.tensor([3, 5, 0], dtype=torch.int32)
    builder.actual_seq_lengths_key = torch.tensor([3, 5, 0], dtype=torch.int32)
    builder.dsa_cp_actual_seq_lengths_query = torch.zeros(3, dtype=torch.int32)
    builder.dsa_cp_actual_seq_lengths_key = torch.zeros(3, dtype=torch.int32)
    builder.dsa_cp_spec_actual_seq_lengths_query = None
    builder.dsa_cp_spec_actual_seq_lengths_key = None
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=5,
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    tp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group):
        cos, sin, slot_mapping, extra = builder._prepare_parallel_metadata(
            common,
            torch.arange(10, dtype=torch.float32).view(5, 1, 1, 2),
            torch.arange(10, dtype=torch.float32).view(5, 1, 1, 2),
            torch.arange(5, dtype=torch.int32),
            torch.tensor([3, 5], dtype=torch.int32),
            torch.tensor([3, 5], dtype=torch.int32),
            draft_index=None,
        )

    assert cos.shape[0] == sin.shape[0] == 3
    torch.testing.assert_close(slot_mapping, torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32))
    context = extra["dsa_cp_context"]
    torch.testing.assert_close(context.slot_mapping_cp, torch.tensor([3, 4, -1], dtype=torch.int32))
    torch.testing.assert_close(context.actual_seq_lengths_query, torch.tensor([0, 2], dtype=torch.int32))
    torch.testing.assert_close(context.actual_seq_lengths_key, torch.tensor([0, 5], dtype=torch.int32))
    torch.testing.assert_close(builder.actual_seq_lengths_query, torch.tensor([3, 5, 0], dtype=torch.int32))
    torch.testing.assert_close(builder.actual_seq_lengths_key, torch.tensor([3, 5, 0], dtype=torch.int32))


def test_sfa_dsa_cp_metadata_builder_masks_graph_padding() -> None:
    # TP8, graph size 80 and MTP3 produce 20 four-token request slots. With
    # nine real requests, rank 6 splits a padded slot at its local boundary.
    builder = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
    builder.dsa_cp_actual_seq_lengths_query = torch.zeros(21, dtype=torch.int32)
    builder.dsa_cp_actual_seq_lengths_key = torch.zeros(21, dtype=torch.int32)
    builder.dsa_cp_spec_actual_seq_lengths_query = None
    builder.dsa_cp_spec_actual_seq_lengths_key = None
    query_start_loc = torch.arange(0, 81, 4, dtype=torch.int32)
    seq_lens = torch.zeros(20, dtype=torch.int32)
    seq_lens[:9] = torch.arange(128, 137, dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=20,
        num_input_tokens=80,
        num_actual_tokens=36,
        query_start_loc=query_start_loc,
    )
    tp_group = SimpleNamespace(world_size=8, rank_in_group=6)

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
        return_value=tp_group,
    ):
        _, _, _, extra = builder._prepare_parallel_metadata(
            common,
            torch.zeros(80, 1, 1, 64),
            torch.zeros(80, 1, 1, 64),
            torch.arange(80, dtype=torch.int64),
            query_start_loc[1:],
            seq_lens,
            draft_index=None,
        )

    local_seq_lens = extra["dsa_cp_context"].actual_seq_lengths_key
    assert local_seq_lens[17].item() == 0
    assert torch.all(local_seq_lens >= 0)


def test_sfa_dcp_builder_sizes_replicated_view_from_padded_block_table() -> None:
    def fake_base_init(self, *args, **kwargs) -> None:
        self.dcp_size = 2
        self.kernel_block_size = 128

    kv_cache_spec = SimpleNamespace(block_size=128)
    for pcp_size, expected_num_reqs in ((1, 5), (2, 9)):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                cp_kv_cache_interleave_size=1,
                prefill_context_parallel_size=pcp_size,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=4,
                max_num_batched_tokens=1024,
            ),
            model_config=SimpleNamespace(max_model_len=1024),
        )

        with (
            patch.object(
                DCPMetadataBuilderMixin,
                "__init__",
                new=fake_base_init,
            ),
            patch(
                "vllm_ascend.attention.context_parallel.sfa_cp.get_dcp_group",
                return_value=SimpleNamespace(ranks=[0, 1]),
            ),
        ):
            builder = AscendSFADCPMetadataBuilder(
                kv_cache_spec,
                [],
                vllm_config,
                torch.device("cpu"),
            )

        assert builder.dcp_local_seq_lens_buf.shape == (expected_num_reqs,)
        assert builder.block_table_replicated_view_buf.shape == (
            expected_num_reqs,
            8,
        )
        assert builder.arange_buffer.shape == (8,)


def _make_builder(rank: int = 0) -> AscendSFADCPMetadataBuilder:
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = 2
    builder.dcp_rank = rank
    builder.cp_kv_cache_interleave_size = 4
    builder.blocks_per_phys_block = 1
    builder.replicated_view_block_size = 4
    builder.device = torch.device("cpu")
    builder.block_table_replicated_view_buf = torch.empty(
        (4, 8),
        dtype=torch.int32,
    )
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.empty(32, dtype=torch.int32)
    return builder


def test_sfa_dcp_local_sequence_lengths_follow_interleave_layout() -> None:
    seq_lens = torch.tensor([0, 3, 4, 5, 8, 9, 12], dtype=torch.int32)

    rank0 = _make_builder(rank=0)._get_dcp_local_seq_lens(seq_lens)
    rank1 = _make_builder(rank=1)._get_dcp_local_seq_lens(seq_lens)

    torch.testing.assert_close(rank0, torch.tensor([0, 3, 4, 4, 4, 5, 8], dtype=torch.int32))
    torch.testing.assert_close(rank1, torch.tensor([0, 0, 0, 1, 4, 4, 4], dtype=torch.int32))


def test_sfa_dcp_builds_replicated_block_table_view() -> None:
    builder = _make_builder()
    local_block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)

    replicated = builder._build_block_table_replicated_view(
        local_block_table,
        seq_lens,
    )

    torch.testing.assert_close(
        replicated,
        torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32),
    )


def test_sfa_dcp_updates_dsa_cp_local_slot_mapping_with_padding() -> None:
    builder = AscendSFADSADCPMetadataBuilder.__new__(AscendSFADSADCPMetadataBuilder)
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=6,
        local_start=2,
        local_end_with_pad=5,
        slot_mapping_cp=None,
    )
    metadata = SimpleNamespace(dsa_cp_context=dsa_cp_context)

    builder._update_parallel_slot_mapping(
        metadata,
        slot_mapping=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        num_input_tokens=4,
    )

    torch.testing.assert_close(
        dsa_cp_context.slot_mapping_cp,
        torch.tensor([12, 13, -1], dtype=torch.int32),
    )
