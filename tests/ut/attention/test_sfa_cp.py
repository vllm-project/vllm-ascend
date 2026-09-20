# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
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
    resolve_sfa_impl,
    resolve_sfa_metadata_builder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    PreprocessType,
)
from vllm_ascend.weight_switch import (
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchLoadState,
    WeightSwitchMixin,
)


@pytest.mark.parametrize("first_block_id", [0, 3])
def test_sfa_pcp_dcp_compact_kv_selects_only_allocated_blocks(first_block_id):
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.max_local_block_table_cols = 3
    builder.arange_buffer = torch.arange(3, dtype=torch.int32)
    builder.dcp_size = 2
    builder.dcp_collective_rank_order = torch.tensor([0, 1], dtype=torch.int32)
    global_table = torch.tensor([[first_block_id, 91, 92], [4, 5, 93]], dtype=torch.int32)
    original_table = global_table.clone()
    context = SimpleNamespace(
        global_batch=SimpleNamespace(num_reqs=2, is_prefilling_np=torch.tensor([True, True])),
        global_block_tables=(torch.full_like(global_table, 77), global_table),
        global_block_table_num_blocks=torch.tensor([[0, 0], [1, 2]], dtype=torch.int32),
    )
    metadata = MagicMock(spec=AscendSFADCPMetadata)
    common_metadata = SimpleNamespace()
    with (
        patch.object(builder, "_build_pcp_ordered_indexer_slot_mapping", return_value=None),
        patch.object(builder, "_build_with_metadata_view", return_value=metadata) as build_view,
    ):
        assert builder.build(0, common_metadata, pcp_context=context, pcp_cache_group_idx=1) is metadata

    compact_source = build_view.call_args.kwargs["global_dcp_block_table"]
    compact_num_blocks = build_view.call_args.kwargs["global_dcp_num_blocks"]
    torch.testing.assert_close(compact_num_blocks, torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(global_table, original_table)
    # Local tails may still contain stale IDs; attention consumes only the
    # allocated columns, whose indices must match the canonical dictionary.
    valid_ids, remapped = builder._build_compact_kv_gather_metadata(
        global_table, global_dcp_block_table=compact_source, global_dcp_num_blocks=compact_num_blocks
    )
    torch.testing.assert_close(valid_ids, torch.tensor([first_block_id, 4, 5], dtype=torch.int32))
    torch.testing.assert_close(remapped[0, :2], torch.tensor([0, 3], dtype=torch.int32))
    torch.testing.assert_close(remapped[1, :4], torch.tensor([1, 4, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(global_table, original_table)


@pytest.mark.parametrize("num_input_tokens", [12, 16])
def test_sfa_dcp_replicated_slots_exclude_input_padding(num_input_tokens):
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.dcp_size = 2
    builder.replicated_view_block_size = 4
    builder.arange_buffer = torch.arange(2, dtype=torch.int32)
    builder.block_table_replicated_view_buf = torch.empty((3, 2), dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.full((16,), 999, dtype=torch.int32)
    offsets = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=3,
        num_input_tokens=num_input_tokens,
        num_actual_tokens=12,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        positions=torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 0, 3, 4, 7] + [9999] * 4),
    )
    block_table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    slots = builder._build_slot_mapping_replicated_view(metadata, block_table)
    expected = torch.tensor(
        [40, 41, 44, 45, 82, 83, 86, 87, 120, 123, 124, 127] + [-1] * (num_input_tokens - 12),
        dtype=torch.int32,
    )
    torch.testing.assert_close(slots, expected)


@pytest.mark.parametrize("has_indexer", [False, True])
def test_sfa_dcp_indexer_metadata_preserves_independent_cache_view(has_indexer):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.has_indexer = has_indexer
    impl.layer_name = "model.layers.0.self_attn.attn"
    prefix = "model.layers.0.self_attn.indexer.k_cache"
    impl.indexer = SimpleNamespace(k_cache=SimpleNamespace(prefix=prefix))
    indexer_metadata = SimpleNamespace(slot_mapping=torch.tensor([90]), block_table=torch.tensor([[91]]), block_size=1)
    with patch("vllm_ascend.attention.sfa_v1.get_forward_context") as get_context:
        get_context.return_value.attn_metadata = {prefix: indexer_metadata}
        result = impl._get_indexer_attn_metadata()
    if not has_indexer:
        assert result is None
        get_context.assert_not_called()
        return
    assert result is indexer_metadata
    torch.testing.assert_close(result.slot_mapping, torch.tensor([90]))
    torch.testing.assert_close(result.block_table, torch.tensor([[91]]))
    assert result.block_size == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_sfa_dcp_local_seq_lens_uses_configured_rank_and_interleave(rank):
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = 2
    builder.dcp_rank = rank
    builder.cp_kv_cache_interleave_size = 128
    lengths = torch.tensor([[0, 1, 127, 128], [129, 255, 256, 257]], dtype=torch.int64)
    expected = [[0, 1, 127, 128], [128, 128, 128, 129]] if rank == 0 else [[0, 0, 0, 0], [1, 127, 128, 128]]
    torch.testing.assert_close(builder._get_dcp_local_seq_lens(lengths), torch.tensor(expected, dtype=torch.int32))


class _PCPOProjLinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _PCPOProjLayer:
    def __init__(self) -> None:
        self.input_size = 8
        self.input_size_per_partition = 2
        self.output_size = 3
        self.output_size_per_partition = 3
        self.weight = torch.nn.Parameter(
            torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]),
            requires_grad=False,
        )
        self.bias = torch.nn.Parameter(
            torch.tensor([1.0, 2.0, 3.0]),
            requires_grad=False,
        )
        self.quant_method = _PCPOProjLinearMethod()
        self.reduce_results = True
        self.tp_size = 2
        self.tp_rank = 0
        self.skip_bias_add = False

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias), None


def _make_pcp_o_proj_impl():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sfa_pcp = True
    impl._o_proj_weight_switch_enabled = False
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    impl.o_proj_weight_switch_config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
    impl.o_proj_weight_load_state = WeightSwitchLoadState(
        input_size_per_partition_before=4,
        input_size_per_partition_after=2,
    )
    impl.o_proj = _PCPOProjLayer()
    return impl


def test_sfa_pcp_weight_switch_does_not_install_loader_when_disabled() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
    )
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.enable_pcp_o_proj_weight_sharding",
            return_value=False,
        ),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=pcp_group),
        patch.object(AscendSFAImpl, "_get_o_proj_weight_switch_method") as get_method,
    ):
        impl._initialize_pcp_weight_switch()

    assert impl.enable_sfa_pcp
    assert not impl.enable_pcp_o_proj_weight_sharding
    assert impl.o_proj_weight_switch_config.group is pcp_group
    assert not hasattr(impl, "o_proj_weight_load_state")
    get_method.assert_not_called()


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
        assert resolve_sfa_impl(vllm_config) is AscendSFAImpl
        assert AscendSFAImpl.supports_mtp_with_cp_non_trivial_interleave_size


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
        query_start_loc_np=torch.tensor([0, 3], dtype=torch.int32).numpy(),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(local_block_table,),
        padded_gather_idx=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        gathered_kv_write_mask=torch.tensor([True, True, True, False]),
    )
    global_common = SimpleNamespace(seq_lens=global_batch.seq_lens, block_table_tensor=local_block_table, num_reqs=1)
    common_attn_metadata = SimpleNamespace(
        replace=Mock(return_value=global_common),
    )

    result = builder._build_pcp_ordered_indexer_slot_mapping(
        common_attn_metadata,
        pcp_context,
        0,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([102, 100, 101, -1], dtype=torch.int32),
    )
    query_start_loc_cpu = common_attn_metadata.replace.call_args.kwargs["query_start_loc_cpu"]
    torch.testing.assert_close(query_start_loc_cpu, global_batch.query_start_loc)
    common_attn_metadata.replace.assert_called_once_with(
        query_start_loc=global_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=global_batch.seq_lens,
        num_reqs=1,
        num_actual_tokens=3,
        num_input_tokens=3,
        positions=global_batch.positions,
        block_table_tensor=local_block_table,
    )
    builder._get_dcp_local_block_table.assert_called_once_with(
        local_block_table,
        1,
    )
    builder._build_block_table_replicated_view.assert_called_once_with(
        local_block_table,
        global_batch.seq_lens,
    )


@pytest.mark.parametrize("with_global_view", [False, True])
def test_sfa_dcp_compact_kv_table_uses_logical_dcp_rank_order(with_global_view) -> None:
    builder_cls = AscendSFAPCPDCPMetadataBuilder if with_global_view else AscendSFADCPMetadataBuilder
    builder = builder_cls.__new__(builder_cls)
    builder.device = torch.device("cpu")
    builder.arange_buffer = torch.arange(2, dtype=torch.int32)
    builder.dcp_size = 8
    builder.dcp_collective_rank_order = torch.tensor(
        [0, 4, 1, 5, 2, 6, 3, 7],
        dtype=torch.int32,
    )
    dcp_block_table = torch.tensor([[5, 9]], dtype=torch.int32)

    if with_global_view:
        valid_block_ids, block_table = builder._build_compact_kv_gather_metadata(
            dcp_block_table,
            global_dcp_block_table=torch.tensor([[5, 99], [9, 88]], dtype=torch.int32),
            global_dcp_num_blocks=torch.tensor([1, 1], dtype=torch.int32),
        )
    else:
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


@pytest.mark.parametrize(
    "local_rows,expected_rows",
    [
        ([[4, 0], [3, 0], [4, 0]], [[1, 3], [0, 2], [1, 3]]),
        ([[4, 0], [4, 0]], [[1, 3], [1, 3]]),
    ],
)
def test_sfa_pcp_dcp_compact_kv_uses_global_request_blocks(local_rows, expected_rows) -> None:
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.arange_buffer = torch.arange(2, dtype=torch.int32)
    builder.dcp_size = 2
    builder.dcp_collective_rank_order = torch.arange(2, dtype=torch.int32)
    # Actual [8, 1] prefill layout: the second PCP rank has no row for
    # the one-token request (block 3), despite both tables having zero tails.
    global_dcp_block_table = torch.tensor([[3, 0], [4, 0]], dtype=torch.int32)

    valid_block_ids, block_table = builder._build_compact_kv_gather_metadata(
        torch.tensor(local_rows, dtype=torch.int32),
        global_dcp_block_table=global_dcp_block_table,
        global_dcp_num_blocks=torch.tensor([1, 1], dtype=torch.int32),
    )

    torch.testing.assert_close(valid_block_ids, torch.tensor([3, 4], dtype=torch.int32))
    # Only the first logical block is allocated for each request.
    torch.testing.assert_close(block_table[:, :2], torch.tensor(expected_rows, dtype=torch.int32))


def test_sfa_pcp_dcp_compact_kv_requires_global_block_counts():
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    block_table = torch.tensor([[3]], dtype=torch.int32)
    with pytest.raises(ValueError, match="requires valid block counts"):
        builder._build_compact_kv_gather_metadata(block_table, global_dcp_block_table=block_table)


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
        AscendSFAImpl,
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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sfa_pcp = True
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = 3
    impl.kv_a_layernorm = lambda x: x
    attn_metadata = SimpleNamespace(num_decode_tokens=1)
    kv_no_split = torch.arange(6, dtype=torch.float32).view(2, 3)
    cos = torch.arange(2, dtype=torch.float32).view(2, 1)
    sin = cos + 10
    slots = torch.tensor([4, 5], dtype=torch.int64)
    gathered_kv = torch.arange(12, dtype=torch.float32).view(4, 3)
    gathered_cos = torch.arange(4, dtype=torch.float32).view(4, 1)
    gathered_sin = gathered_cos + 10
    gathered_slots = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    kv_cache = (torch.empty(4, 3),)

    with (
        patch(
            "vllm_ascend.attention.sfa_v1._gather_prefill_cache_inputs",
            return_value=((gathered_kv, gathered_cos, gathered_sin), gathered_slots),
        ) as gather,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_") as scatter,
    ):
        result = impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)

    assert result == (None, None)
    gather.assert_called_once_with((kv_no_split, cos, sin), slots, 1)
    scatter.assert_called_once()
    torch.testing.assert_close(scatter.call_args.args[1], gathered_slots.view(-1, 1))
    torch.testing.assert_close(scatter.call_args.args[2], gathered_kv)


def test_sfa_pcp_o_proj_switch_slices_the_tp_local_weight_by_pcp_rank() -> None:
    AscendSFAImpl.o_proj_full_pools.clear()
    impl = _make_pcp_o_proj_impl()

    impl._enable_o_proj_full_weight_switch()

    assert impl._o_proj_weight_switch_enabled
    torch.testing.assert_close(
        impl.o_proj.weight,
        torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]),
    )
    assert impl.o_proj_weight_state.gather_parts["weight"].full_tensor.shape == (3, 4)


def test_sfa_pcp_prefill_gathers_weight_and_restores_local_view() -> None:
    impl = _make_pcp_o_proj_impl()
    impl._enable_o_proj_full_weight_switch()

    local_weight_ptr = impl.o_proj.weight.data_ptr()
    full_weight = impl.o_proj_weight_state.gather_parts["weight"].full_tensor
    full_weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))
    input_ = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(input_, full_weight, impl.o_proj.bias)

    result = impl._finalize_o_proj(input_, torch.empty(1, 3), gather_full_o_proj=True)

    torch.testing.assert_close(result, expected)
    assert impl.o_proj.weight.data_ptr() == local_weight_ptr


def test_sfa_pcp_decode_projects_local_weight_then_reduces_pcp_and_tp() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl = _make_pcp_o_proj_impl()
    impl.o_proj_weight_switch_config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
    impl._enable_o_proj_full_weight_switch()

    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    input_ = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(input_, full_weight, impl.o_proj.bias)
    pcp_group.all_reduce = lambda _: torch.nn.functional.linear(input_, full_weight, bias=None)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    with patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=tp_group):
        result = impl._finalize_o_proj(input_, torch.empty_like(expected), gather_full_o_proj=False)

    torch.testing.assert_close(result, expected)


def test_sfa_pcp_prefill_context_starts_weight_gather_but_decode_does_not() -> None:
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sfa_pcp = True
    impl._o_proj_weight_switch_enabled = True
    impl._all_gather_o_proj_full_weight = MagicMock()
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.ChunkedPrefill,
        cum_query_lens=torch.empty(0),
        seq_lens=torch.empty(0),
        pcp_slot_mapping=torch.empty(0),
        slot_mapping=torch.empty(0),
    )

    prefill = impl._get_parallel_forward_context(metadata, 1, torch.empty(1))
    assert prefill.gather_full_o_proj
    impl._all_gather_o_proj_full_weight.assert_called_once_with()

    metadata.attn_state = AscendAttentionState.DecodeOnly
    decode = impl._get_parallel_forward_context(metadata, 1, torch.empty(1))
    assert not decode.gather_full_o_proj
    impl._all_gather_o_proj_full_weight.assert_called_once()


def test_sfa_cp_query_gather_axis_follows_composed_layout() -> None:
    dcp_impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    combined_impl = AscendSFADSADCPImpl.__new__(AscendSFADSADCPImpl)
    assert dcp_impl._parallel_query_gather_dim() == 1
    assert combined_impl._parallel_query_gather_dim() == 0


@pytest.mark.parametrize("sfa_c8", [False, True])
@pytest.mark.parametrize("li_c8", [False, True])
@pytest.mark.parametrize("preprocess_type", [PreprocessType.NATIVE, PreprocessType.MLAPO, PreprocessType.PROLOG_V3])
@pytest.mark.parametrize(
    "has_indexer,is_mtp,skip_topk,expect_indexer",
    [(True, False, True, False), (True, True, True, True), (True, False, False, True), (False, False, True, False)],
)
def test_dsa_cp_indexer_cache_follows_runtime_ownership(
    sfa_c8, li_c8, preprocess_type, has_indexer, is_mtp, skip_topk, expect_indexer
):
    # Exercise the actual SFA forward, including cache composition and metadata
    # lookup. Only projections/kernels are mocked; static layers have no cache
    # or metadata, while MTP must call the indexer even when top-k is skipped.
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.has_indexer = has_indexer
    impl.layerwise_kv_cache_hook = None
    impl.g_proj = None
    impl._is_mtp_layer = is_mtp
    impl.skip_topk = skip_topk
    impl.use_index_cache = True
    impl.enable_sparse_sfa_c8 = sfa_c8
    impl.enable_sparse_li_c8 = li_c8
    impl.preprocess_type = preprocess_type
    impl.layer_name = "model.layers.80.self_attn.attn" if is_mtp else "model.layers.2.self_attn.attn"
    impl.q_lora_rank = 2
    impl.qk_rope_head_dim = 2
    impl.kv_lora_rank = 4
    hidden_states = torch.zeros(2, 4)
    shared_topk = torch.ones(2, 1, dtype=torch.int32)
    computed_topk = torch.zeros_like(shared_topk)
    main_cache = tuple(torch.empty(1) for _ in range(1 if sfa_c8 else 2))
    indexer_cache = tuple(torch.empty(1) for _ in range(2 if li_c8 else 1))
    indexer = MagicMock(return_value=computed_topk)
    indexer.k_cache = SimpleNamespace(prefix="indexer.k_cache", kv_cache=indexer_cache if expect_indexer else None)
    indexer.num_cache_tensors = len(indexer_cache)
    impl.indexer = indexer if has_indexer else None
    metadata = SimpleNamespace(
        cos=hidden_states,
        sin=hidden_states,
        num_input_tokens=2,
        num_decode_tokens=2,
        attn_state=AscendAttentionState.DecodeOnly,
    )
    slots = torch.tensor([0, 1])
    lengths = torch.tensor([1, 2])
    context = SimpleNamespace(
        actual_seq_lengths_query=lengths,
        actual_seq_lengths_key=lengths,
        kv_slot_mapping=slots,
        gather_full_o_proj=False,
        topk_num_tokens=2,
    )
    own_query_lengths = torch.tensor([9, 10])
    own_key_lengths = torch.tensor([11, 12])
    own_metadata = SimpleNamespace(
        actual_seq_lengths_query=own_query_lengths,
        actual_seq_lengths_key=own_key_lengths,
    )
    forward_context = SimpleNamespace(attn_metadata={"indexer.k_cache": own_metadata} if expect_indexer else {})
    impl._get_sfa_kv_slot_mapping = MagicMock(return_value=slots)
    impl._get_parallel_forward_context = MagicMock(return_value=context)
    impl._prepare_native_hidden_states = MagicMock(return_value=hidden_states)
    impl.fused_qkv_a_proj = MagicMock(return_value=(torch.zeros(2, 8),))
    impl.q_a_layernorm = MagicMock(side_effect=lambda x: x)
    impl.exec_kv = MagicMock(return_value=(hidden_states, hidden_states))
    impl._prepare_kv_for_parallel = MagicMock(return_value=(None, []))
    impl._store_parallel_kv = MagicMock(return_value=(hidden_states, hidden_states))
    impl._q_proj_and_k_up_proj = MagicMock(return_value=(hidden_states, hidden_states))
    impl.rope_single = MagicMock(return_value=hidden_states)
    impl._record_query_gather_context = MagicMock()
    fused_output = (hidden_states, hidden_states, hidden_states, hidden_states)
    impl._sfa_preprocess_mlapo = MagicMock(return_value=fused_output)
    impl._sfa_preprocess_prolog_v3 = MagicMock(return_value=fused_output)
    impl._get_indexcache_topk_indices = MagicMock(return_value=shared_topk)
    impl._update_indexcache_topk_indices = MagicMock()
    impl._execute_sparse_flash_attention_process = MagicMock(return_value=hidden_states)
    impl._v_up_proj = MagicMock(return_value=hidden_states)
    impl._finalize_o_proj = MagicMock(return_value=hidden_states)
    with (
        patch("vllm_ascend.attention.sfa_v1.get_forward_context", return_value=forward_context),
        patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written") as notify,
        patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start"),
        patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"),
    ):
        impl.forward(impl.layer_name, hidden_states, main_cache, metadata, output=torch.empty_like(hidden_states))
    if expect_indexer:
        indexer.assert_called_once()
        assert indexer.call_args.kwargs["compute_topk"] is (not skip_topk)
        assert indexer.call_args.args[2] is hidden_states
        assert indexer.call_args.args[3] is own_metadata
        assert own_metadata.actual_seq_lengths_query is own_query_lengths
        assert own_metadata.actual_seq_lengths_key is own_key_lengths
    else:
        indexer.assert_not_called()
    attention_args = impl._execute_sparse_flash_attention_process.call_args.args
    assert len(attention_args[2]) == len(main_cache) + (len(indexer_cache) if expect_indexer else 0)
    assert attention_args[3] is (shared_topk if skip_topk else computed_topk)
    notify.assert_called_once_with(impl.layer_name)


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
            patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_dcp", return_value=True) as dcp,
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

        dcp.assert_called_once_with()
        assert builder.dcp_enabled
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


@pytest.mark.parametrize(
    "is_consumer,is_producer,recompute", [(True, False, True), (True, False, False), (False, True, True)]
)
@pytest.mark.parametrize("query_lens", [[1, 1], [3, 3], [3, 5]])
def test_sfa_dcp_split_uses_builder_config_without_current_context(is_consumer, is_producer, recompute, query_lens):
    builder = _make_builder()
    builder.dcp_enabled = True
    builder.decode_threshold = 3
    builder.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer, is_kv_producer=is_producer),
    )
    builder.dcp_local_seq_lens_buf = torch.empty(2, dtype=torch.int32)
    slots = torch.arange(sum(query_lens), dtype=torch.int64)
    blocks = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    common = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=max(query_lens),
        num_reqs=2,
        num_actual_tokens=sum(query_lens),
        num_input_tokens=sum(query_lens),
        query_start_loc_cpu=torch.tensor([0, query_lens[0], sum(query_lens)], dtype=torch.int32),
        is_prefilling=torch.ones(2, dtype=torch.bool),
        slot_mapping=slots,
        block_table_tensor=blocks,
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        dcp_local_seq_lens=torch.tensor([6, 12], dtype=torch.int32),
    )
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=None),
        patch(
            "vllm_ascend.utils.get_ascend_config",
            return_value=SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute)),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.enable_dcp",
            side_effect=AssertionError("use cached DCP state"),
        ),
        patch.object(builder, "_get_dcp_local_block_table", return_value=blocks),
        patch.object(builder, "_build_block_table_replicated_view", return_value=blocks),
        patch.object(builder, "_build_slot_mapping_replicated_view", return_value=slots),
        patch.object(builder, "_build_compact_kv_gather_metadata", return_value=(torch.arange(4), blocks)) as gather,
        patch.object(builder, "_update_parallel_slot_mapping"),
    ):
        result = builder._build_with_metadata_view(common, lambda: metadata)
    num_decodes = sum(q <= 3 for q in query_lens) if is_consumer and not is_producer and recompute else 0
    assert result.num_decodes == num_decodes
    assert result.num_prefills == 2 - num_decodes
    assert result.num_decode_tokens == sum(query_lens[:num_decodes])
    assert gather.call_count == int(result.num_prefills > 0)
    assert common.slot_mapping is slots
    assert common.block_table_tensor is blocks


def test_sfa_dcp_prefill_passes_contiguous_gathered_cache() -> None:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_group = Mock()
    packed = torch.randn(2, 128, 1, 576)
    gathered = packed.split((512, 64), dim=-1)
    assert all(not tensor.is_contiguous() for tensor in gathered)
    context = SimpleNamespace(gather_context=object(), kv_gather_block_table=object())
    metadata = SimpleNamespace(dcp_context=context)
    output = object()
    with (
        patch.object(impl, "_has_prefill", return_value=True),
        patch.object(impl, "_finish_dcp_gather", return_value=gathered),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.execute_sparse_flash_attention_process",
            return_value=output,
        ) as execute,
    ):
        result = impl._execute_sparse_flash_attention_process(None, None, (), None, metadata, None, None)
    assert result is output
    for actual, expected in zip(execute.call_args.args[3], gathered):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("impl_cls", [AscendSFADCPImpl, AscendSFAPCPDCPImpl, AscendSFADSADCPImpl])
@pytest.mark.parametrize("num_prefills,num_decode_tokens", [(0, 2), (1, 0), (1, 1)])
@pytest.mark.parametrize("enable_c8", [False, True])
def test_sfa_dcp_slot_mapping_matches_parallel_layout(impl_cls, num_prefills, num_decode_tokens, enable_c8):
    impl = impl_cls.__new__(impl_cls)
    impl.enable_sparse_sfa_c8 = enable_c8
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_input_tokens = 2
    metadata.num_prefills = num_prefills
    metadata.num_decode_tokens = num_decode_tokens
    full_slots = torch.tensor([3200, -1, 3201, -1], dtype=torch.int32)
    metadata.dcp_context = SimpleNamespace(slot_mapping=full_slots)

    result = impl._get_sfa_kv_slot_mapping(metadata)

    if impl_cls is AscendSFAPCPDCPImpl and num_prefills:
        assert result is full_slots
        assert result.tolist() == [3200, -1, 3201, -1]
    else:
        assert result.tolist() == [3200, -1]
        assert result.data_ptr() == full_slots.data_ptr()
