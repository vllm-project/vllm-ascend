# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _make_tq_dcp_decode_case() -> SimpleNamespace:
    impl = MagicMock(spec=AscendSFADCPImpl)
    impl._has_prefill.return_value = False
    impl.kv_lora_rank = 512
    impl.dcp_group = MagicMock()

    gather_context = object()
    ql_nope = torch.randn(2, 1, 4)
    q_pe = torch.randn(2, 1, 2)
    gathered_ql_nope = torch.randn_like(ql_nope)
    gathered_q_pe = torch.randn_like(q_pe)
    topk_indices = torch.tensor([[[0, 1]], [[2, 3]]], dtype=torch.int32)
    gathered_topk = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]], dtype=torch.int32)
    remapped_topk = torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.int32)
    rotated_query = torch.randn(2, 1, 6)
    attn_out = torch.randn(2, 1, 4, dtype=torch.bfloat16)
    softmax_max = torch.zeros(1, 2, 1)
    softmax_sum = torch.ones(1, 2, 1)
    merged_output = torch.randn_like(attn_out)

    impl._finish_dcp_gather.return_value = (gathered_ql_nope, gathered_q_pe)
    impl._remap_sparse_indices.return_value = remapped_topk
    impl._tq_rotate_query.return_value = rotated_query
    impl._turboquant_sfa.return_value = (attn_out, softmax_max, softmax_sum)
    impl._merge_dcp_outputs.return_value = merged_output

    return SimpleNamespace(
        impl=impl,
        gather_context=gather_context,
        ql_nope=ql_nope,
        q_pe=q_pe,
        kv_cache=(torch.empty(1),),
        topk_indices=topk_indices,
        gathered_topk=gathered_topk,
        remapped_topk=remapped_topk,
        merged_output=merged_output,
        block_table=torch.tensor([[0], [0]], dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1, 1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([8, 8], dtype=torch.int32),
    )


def test_tq_dcp_decode_without_dsa_cp_context() -> None:
    case = _make_tq_dcp_decode_case()
    dcp_context = SimpleNamespace(
        gather_context=case.gather_context,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    metadata = SimpleNamespace(dcp_context=dcp_context)

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.tq_latent_store.had_inv",
        side_effect=lambda tensor, **_: tensor,
    ) as mock_had_inv:
        result = AscendSFADCPImpl._execute_tq_dcp_sfa(
            case.impl,
            case.ql_nope,
            case.q_pe,
            case.kv_cache,
            case.topk_indices,
            metadata,
            case.actual_seq_lengths_query,
            case.actual_seq_lengths_key,
        )

    case.impl.dcp_group.all_gather.assert_not_called()
    assert case.impl._remap_sparse_indices.call_args.args[0] is case.topk_indices
    tq_args = case.impl._turboquant_sfa.call_args.args
    assert tq_args[2] is case.remapped_topk
    assert tq_args[4] is case.actual_seq_lengths_query
    assert tq_args[5] is case.seq_lens
    assert case.impl._turboquant_sfa.call_args.kwargs == {
        "sparse_mode": 0,
        "return_softmax_lse": True,
    }
    assert case.impl._merge_dcp_outputs.call_args.args[2] is None
    assert dcp_context.gather_context is None
    assert mock_had_inv.call_args.args[0] is case.merged_output
    assert mock_had_inv.call_args.kwargs == {"head_dim": 512}
    torch.testing.assert_close(result, case.merged_output)


def test_tq_dcp_decode_with_dsa_cp_context() -> None:
    case = _make_tq_dcp_decode_case()
    case.impl.dcp_group.all_gather.return_value = case.gathered_topk
    dsa_cp_context = SimpleNamespace()
    cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    dcp_context = SimpleNamespace(
        gather_context=case.gather_context,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    metadata = SimpleNamespace(
        dcp_context=dcp_context,
        dsa_cp_context=dsa_cp_context,
        cum_query_lens=cum_query_lens,
    )

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.tq_latent_store.had_inv",
        side_effect=lambda tensor, **_: tensor,
    ):
        result = AscendSFADCPImpl._execute_tq_dcp_sfa(
            case.impl,
            case.ql_nope,
            case.q_pe,
            case.kv_cache,
            case.topk_indices,
            metadata,
            case.actual_seq_lengths_query,
            case.actual_seq_lengths_key,
        )

    all_gather_call = case.impl.dcp_group.all_gather.call_args
    torch.testing.assert_close(all_gather_call.args[0], case.topk_indices)
    assert all_gather_call.kwargs == {"dim": 0}
    assert case.impl._remap_sparse_indices.call_args.args[0] is case.gathered_topk
    tq_args = case.impl._turboquant_sfa.call_args.args
    assert tq_args[2] is case.remapped_topk
    assert tq_args[4] is cum_query_lens
    assert tq_args[5] is case.seq_lens
    assert case.impl._merge_dcp_outputs.call_args.args[2] is dsa_cp_context
    assert dcp_context.gather_context is None
    torch.testing.assert_close(result, case.merged_output)


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
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
        ),
        model_config=SimpleNamespace(max_model_len=1024),
    )

    with patch.object(DCPMetadataBuilderMixin, "__init__", new=fake_base_init):
        builder = AscendSFADCPMetadataBuilder(
            kv_cache_spec,
            [],
            vllm_config,
            torch.device("cpu"),
        )

    assert builder.block_table_replicated_view_buf.shape == (5, 8)
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
