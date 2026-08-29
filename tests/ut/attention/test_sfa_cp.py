# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.attention_v1 import AscendAttentionState
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
    SFAForwardContext,
)
from vllm_ascend.weight_switch import (
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchLoadState,
    WeightSwitchMixin,
)


class _PCPOProjLinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


def _make_pcp_o_proj_impl():
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl._o_proj_weight_switch_enabled = False
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    impl.o_proj_weight_switch_config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
    impl.o_proj_weight_load_state = WeightSwitchLoadState(
        input_size_per_partition_before=4,
        input_size_per_partition_after=2,
    )
    impl.o_proj = SimpleNamespace(
        input_size=8,
        input_size_per_partition=2,
        output_size=3,
        output_size_per_partition=3,
        weight=torch.nn.Parameter(torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]), requires_grad=False),
        bias=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]), requires_grad=False),
        quant_method=_PCPOProjLinearMethod(),
        reduce_results=True,
        tp_size=2,
        tp_rank=0,
        skip_bias_add=False,
    )
    return impl


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


def test_sfa_pcp_o_proj_switch_slices_the_tp_local_weight_by_pcp_rank() -> None:
    AscendSFAPCPImpl.o_proj_full_pools.clear()
    impl = _make_pcp_o_proj_impl()

    impl._enable_o_proj_weight_switch()

    assert impl._o_proj_weight_switch_enabled
    torch.testing.assert_close(
        impl.o_proj.weight,
        torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]),
    )
    assert impl.o_proj_weight_state.gather_parts["weight"].full_tensor.shape == (3, 4)


def test_sfa_pcp_prefill_gathers_weight_and_restores_local_view() -> None:
    impl = _make_pcp_o_proj_impl()
    impl._enable_o_proj_weight_switch()

    local_weight_ptr = impl.o_proj.weight.data_ptr()
    full_weight = impl.o_proj_weight_state.gather_parts["weight"].full_tensor
    full_weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))

    def fake_finalize(_self, _attn_output, output, _gather_full_o_proj):
        assert impl.o_proj.weight.data_ptr() == full_weight.data_ptr()
        output.fill_(7)
        return output

    with patch.object(AscendSFAImpl, "_finalize_o_proj", new=fake_finalize):
        result = impl._finalize_o_proj(torch.empty(1, 4), torch.empty(1, 3), gather_full_o_proj=True)

    assert result.tolist() == [[7.0, 7.0, 7.0]]
    assert impl.o_proj.weight.data_ptr() == local_weight_ptr


def test_sfa_pcp_decode_projects_local_weight_then_reduces_pcp_and_tp() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl = _make_pcp_o_proj_impl()
    impl.o_proj_weight_switch_config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
    impl._enable_o_proj_weight_switch()

    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    input_ = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(input_, full_weight, impl.o_proj.bias)
    pcp_group.all_reduce = lambda _: torch.nn.functional.linear(input_, full_weight, bias=None)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group):
        result = impl._finalize_o_proj(input_, torch.empty_like(expected), gather_full_o_proj=False)

    torch.testing.assert_close(result, expected)


def test_sfa_pcp_prefill_context_starts_weight_gather_but_decode_does_not() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl._o_proj_weight_switch_enabled = True
    impl._all_gather_o_proj_full_weight = MagicMock()
    base_context = SFAForwardContext(
        actual_seq_lengths_query=torch.empty(0),
        actual_seq_lengths_key=torch.empty(0),
        kv_slot_mapping=torch.empty(0),
        topk_num_tokens=0,
    )

    with patch.object(AscendSFAImpl, "_get_parallel_forward_context", return_value=base_context):
        prefill = impl._get_parallel_forward_context(
            SimpleNamespace(attn_state=AscendAttentionState.ChunkedPrefill),
            1,
            torch.empty(1),
        )
    assert prefill.gather_full_o_proj
    impl._all_gather_o_proj_full_weight.assert_called_once_with()

    base_context.gather_full_o_proj = False
    with patch.object(AscendSFAImpl, "_get_parallel_forward_context", return_value=base_context):
        decode = impl._get_parallel_forward_context(
            SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly),
            1,
            torch.empty(1),
        )
    assert not decode.gather_full_o_proj
    impl._all_gather_o_proj_full_weight.assert_called_once()


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
