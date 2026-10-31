# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs

import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
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
    PreprocessType,
    SFAForwardContext,
)
from vllm_ascend.quantization.methods import AscendLinearScheme, AscendW8A8DynamicLinearMethod
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
    metadata.dcp_context = SimpleNamespace(slot_mapping=torch.tensor([0, 1, -1, 2], dtype=torch.int32))
    metadata.num_prefills = 1
    metadata.pcp_has_global_prefill = True
    metadata.num_input_tokens = 2
    metadata.num_decode_tokens = 1
    common_metadata = SimpleNamespace(pcp_context=context, pcp_cache_group_idx=1)
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    with (
        patch.object(builder, "_build_pcp_ordered_indexer_slot_mapping", return_value=None),
        patch.object(builder, "_build_with_metadata_view", side_effect=lambda _, build, **kw: build()) as build_view,
        patch.object(builder, "_build", return_value=metadata) as build,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=group),
    ):
        assert builder.build(0, common_metadata) is metadata

    build.assert_called_once_with(common_metadata, draft_index=None, pcp_context=context)
    torch.testing.assert_close(metadata.pcp_prolog_local_slots, torch.tensor([0, 2], dtype=torch.int32))
    torch.testing.assert_close(metadata.pcp_prolog_global_slots, torch.tensor([1, 2], dtype=torch.int32))
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


def _make_pcp_o_proj_impl(
    *,
    sharded_decode=False,
    activation_dtype=torch.float32,
    weight_dtype=torch.float32,
    num_speculative_tokens=0,
    max_capture_size=4,
    capture_sizes=(1, 2, 4),
):
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = sharded_decode
    impl.vllm_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        model_config=SimpleNamespace(dtype=activation_dtype),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_capture_size,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
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
        weight=torch.nn.Parameter(
            torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]], dtype=weight_dtype), requires_grad=False
        ),
        bias=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]), requires_grad=False),
        quant_method=_PCPOProjLinearMethod(),
        reduce_results=True,
        tp_size=2,
        tp_rank=0,
        skip_bias_add=False,
    )
    load_state = impl.o_proj_weight_load_state
    with (
        patch.object(AscendSFAImpl, "__init__", return_value=None),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_pcp_o_proj_weight_sharding", return_value=True),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=pcp_group),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
            return_value=SimpleNamespace(world_size=2, rank_in_group=0),
        ),
        patch.object(impl.o_proj.quant_method, "prepare_layer_for_parallel_weight_load", return_value=load_state),
    ):
        AscendSFAPCPImpl.__init__(impl)
    return impl


@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_sfa_pcp_decode_buffers_exist_before_forward_with_activation_dtype(activation_dtype):
    impl = _make_pcp_o_proj_impl(sharded_decode=True, activation_dtype=activation_dtype, weight_dtype=torch.int8)
    assert impl.o_proj.weight.dtype == torch.int8
    expected_shapes = {
        "_pcp_o_proj_ag_in_buf": (4, 4),
        "_pcp_o_proj_ag_out_buf": (8, 4),
        "_pcp_o_proj_rs_in_buf": (8, 3),
        "_pcp_o_proj_rs_out_buf": (4, 3),
    }
    for name, shape in expected_shapes.items():
        buffer = getattr(impl, name)
        assert buffer.shape == shape
        assert buffer.dtype == activation_dtype
        assert buffer.device.type == "cpu"


@pytest.mark.parametrize(
    "num_speculative_tokens,max_capture_size,capture_sizes,expected_capacity",
    [
        (0, 0, [], 4),
        (2, 0, [], 12),
        (2, 16, [4, 8, 16], 16),
        (2, 4, [4, 8, 32], 32),
        (2, None, None, 12),
    ],
)
def test_sfa_pcp_decode_buffers_cover_verification_tokens_and_graph_padding(
    num_speculative_tokens, max_capture_size, capture_sizes, expected_capacity
):
    impl = _make_pcp_o_proj_impl(
        sharded_decode=True,
        num_speculative_tokens=num_speculative_tokens,
        max_capture_size=max_capture_size,
        capture_sizes=capture_sizes,
    )
    assert impl._pcp_o_proj_token_capacity == expected_capacity
    assert impl._pcp_o_proj_ag_in_buf.shape == (expected_capacity, 4)
    assert impl._pcp_o_proj_ag_out_buf.shape == (2 * expected_capacity, 4)
    assert impl._pcp_o_proj_rs_in_buf.shape == (2 * expected_capacity, 3)
    assert impl._pcp_o_proj_rs_out_buf.shape == (expected_capacity, 3)


def test_sfa_pcp_decode_buffers_are_shared_across_compatible_layers():
    with (
        patch.object(AscendSFAPCPImpl, "o_proj_full_pools", {}),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.torch.empty", wraps=torch.empty) as allocate,
    ):
        first = _make_pcp_o_proj_impl(sharded_decode=True)
        second = _make_pcp_o_proj_impl(sharded_decode=True)

        assert allocate.call_count == 4
        buffers = [
            value for name, value in vars(first).items() if name.startswith("_pcp_o_proj_") and name.endswith("_buf")
        ]
        assert len({buffer.data_ptr() for buffer in buffers}) == 4
        for name, buffer in vars(first).items():
            if name.startswith("_pcp_o_proj_") and name.endswith("_buf"):
                assert getattr(second, name) is buffer

        first._enable_o_proj_full_weight_switch()
        second._enable_o_proj_full_weight_switch()
        assert first.o_proj_weight_state.gather_parts["weight"].full_tensor.data_ptr() == (
            second.o_proj_weight_state.gather_parts["weight"].full_tensor.data_ptr()
        )
        assert first.o_proj_weight_state.gather_parts["weight"].full_tensor.data_ptr() not in {
            buffer.data_ptr() for buffer in buffers
        }


@pytest.mark.parametrize("changed_config", [{"activation_dtype": torch.bfloat16}, {"max_capture_size": 8}])
def test_sfa_pcp_decode_buffer_pool_separates_dtype_and_shape(changed_config):
    with patch.object(AscendSFAPCPImpl, "o_proj_full_pools", {}):
        first = _make_pcp_o_proj_impl(sharded_decode=True)
        second = _make_pcp_o_proj_impl(sharded_decode=True, **changed_config)

        for name, buffer in vars(first).items():
            if name.startswith("_pcp_o_proj_") and name.endswith("_buf"):
                assert getattr(second, name).data_ptr() != buffer.data_ptr()


def test_sfa_pcp_decode_buffer_pool_uses_weight_device():
    allocate_cpu = torch.empty

    def allocate(shape, *, dtype, device):
        return allocate_cpu(shape, dtype=dtype, device="cpu")

    with (
        patch.object(AscendSFAPCPImpl, "o_proj_full_pools", {}),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.torch.empty", side_effect=allocate) as allocate_mock,
        patch("torch.nn.Parameter") as parameter,
    ):
        parameter.return_value = SimpleNamespace(device=torch.device("npu:0"))
        first = _make_pcp_o_proj_impl(sharded_decode=True)
        parameter.return_value = SimpleNamespace(device=torch.device("npu:1"))
        second = _make_pcp_o_proj_impl(sharded_decode=True)
        parameter.return_value = SimpleNamespace(device=torch.device("npu:0"))
        same_device = _make_pcp_o_proj_impl(sharded_decode=True)

        assert allocate_mock.call_count == 8
        assert [call.kwargs["device"] for call in allocate_mock.call_args_list] == (
            [torch.device("npu:0")] * 4 + [torch.device("npu:1")] * 4
        )
        for name, buffer in vars(first).items():
            if name.startswith("_pcp_o_proj_") and name.endswith("_buf"):
                assert getattr(second, name) is not buffer
                assert getattr(same_device, name) is buffer


def test_sfa_pcp_weight_switch_does_not_install_loader_when_disabled() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    with (
        patch.object(AscendSFAImpl, "__init__", return_value=None),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.enable_pcp_o_proj_weight_sharding",
            return_value=False,
        ),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=pcp_group),
        patch.object(AscendSFAPCPImpl, "_get_o_proj_weight_switch_method") as get_method,
    ):
        impl = AscendSFAPCPImpl()

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


def test_sfa_dcp_resolution() -> None:
    for enabled, classes in (
        (False, (AscendSFAMetadataBuilder, AscendSFAImpl)),
        (True, (AscendSFADCPMetadataBuilder, AscendSFADCPImpl)),
    ):
        with patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_replicated_indexer", return_value=enabled
        ):
            assert resolve_sfa_metadata_builder() is classes[0]
            assert resolve_sfa_impl() is classes[1]


def test_sfa_pcp_resolution_for_mrv2_config() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
    )
    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_replicated_indexer",
            return_value=False,
        ),
    ):
        assert resolve_sfa_impl(vllm_config) is AscendSFAPCPImpl
        assert resolve_sfa_metadata_builder(vllm_config) is AscendSFAMetadataBuilder
        assert AscendSFAPCPImpl.supports_mtp_with_cp_non_trivial_interleave_size


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
    common_attn_metadata = SimpleNamespace(pcp_context=None, pcp_cache_group_idx=None)
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


@pytest.mark.parametrize("global_has_prefill", [False, True])
def test_sfa_pcp_dcp_empty_local_prefill_uses_global_kv_view(global_has_prefill: bool) -> None:
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.decode_threshold = 1
    builder.use_dcp = False
    builder.dcp_local_seq_lens_buf = torch.empty(1, dtype=torch.int32)
    local_block_table = torch.tensor([[7]], dtype=torch.int32)
    global_block_table = torch.tensor([[7], [9]], dtype=torch.int32)
    block_counts = torch.tensor([1, 1], dtype=torch.int32)
    compact_ids = torch.tensor([7, 9], dtype=torch.int32)
    compact_table = torch.tensor([[0, 2]], dtype=torch.int32)
    builder._get_dcp_local_block_table = Mock(return_value=local_block_table)
    builder._build_block_table_replicated_view = Mock(return_value=local_block_table)
    builder._build_slot_mapping_replicated_view = Mock(return_value=torch.tensor([0, -1]))
    builder._build_compact_kv_gather_metadata = Mock(return_value=(compact_ids, compact_table))
    original_slots = torch.tensor([0, -1], dtype=torch.int32)
    original_blocks = torch.tensor([[7]], dtype=torch.int32)
    common = SimpleNamespace(
        slot_mapping=original_slots,
        block_table_tensor=original_blocks,
        num_reqs=1,
        num_input_tokens=2,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        dcp_local_seq_lens=torch.tensor([10], dtype=torch.int32),
    )
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.seq_lens = torch.tensor([10], dtype=torch.int32)
    metadata.pcp_has_global_prefill = global_has_prefill

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills",
        return_value=(1, 0, 1, 0),
    ):
        result = builder._build_with_metadata_view(
            common,
            lambda: metadata,
            global_dcp_block_table=global_block_table,
            global_dcp_num_blocks=block_counts,
        )

    assert result is metadata
    assert metadata.num_prefills == 0
    assert common.slot_mapping is original_slots
    assert common.block_table_tensor is original_blocks
    assert AscendSFAPCPDCPImpl._has_prefill(metadata) is global_has_prefill
    if global_has_prefill:
        builder._build_compact_kv_gather_metadata.assert_called_once_with(
            local_block_table,
            global_dcp_block_table=global_block_table,
            global_dcp_num_blocks=block_counts,
        )
        assert metadata.dcp_context.kv_gather_block_ids is compact_ids
        assert metadata.dcp_context.kv_gather_block_table is compact_table
    else:
        builder._build_compact_kv_gather_metadata.assert_not_called()
        assert metadata.dcp_context.kv_gather_block_ids is None
        assert metadata.dcp_context.kv_gather_block_table is None


def test_sfa_pcp_dcp_empty_local_prefill_joins_dcp_kv_gather() -> None:
    impl = AscendSFAPCPDCPImpl.__new__(AscendSFAPCPDCPImpl)
    impl.dcp_group = object()
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl._start_dcp_gather = Mock(return_value="gathered")
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = 0
    metadata.pcp_has_global_prefill = True
    metadata.dcp_context = SimpleNamespace(
        kv_gather_block_ids=torch.tensor([0, 2]),
        kv_gather_block_table=torch.tensor([[0, 1]]),
        gather_context=None,
    )
    kv_cache = (torch.arange(12, dtype=torch.float32).view(3, 4),)

    impl._record_dcp_kv_gather_context(kv_cache, metadata)

    gather_input = impl._start_dcp_gather.call_args.args[0]
    torch.testing.assert_close(gather_input, kv_cache[0][[0, 2]])
    assert impl._start_dcp_gather.call_args.kwargs == {"dim": 0, "split_sizes": (4,)}
    assert metadata.dcp_context.gather_context == "gathered"
    with patch.object(impl, "_start_dcp_query_gather") as query_gather:
        impl._record_query_gather_context(torch.zeros(1, 2), torch.zeros(1, 2), metadata)
    query_gather.assert_not_called()


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
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 64
    attn_metadata = SimpleNamespace(
        num_decode_tokens=1,
        num_prefills=1,
        num_input_tokens=2,
        uses_pcp_sharded_layout=True,
        pcp_slot_mapping=torch.arange(4),
    )
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


def test_sfa_pcp_gathers_nope_kv_without_rope() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 0
    attn_metadata = SimpleNamespace(num_decode_tokens=1, num_prefills=1, num_input_tokens=2)
    kv_no_split = torch.arange(6, dtype=torch.float32).view(2, 3)
    slots = torch.tensor([0, 1, 0, 5], dtype=torch.int64)
    attn_metadata.uses_pcp_sharded_layout = True
    attn_metadata.pcp_slot_mapping = slots
    gathered_prefill_kv = torch.tensor([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]])
    gathered_kv = torch.cat((kv_no_split[:1], gathered_prefill_kv))
    gathered_slots = torch.tensor([0, 1, 5])
    kv_cache = (torch.empty(1),)

    with (
        patch(
            "vllm.v1.attention.ops.pcp.get_pcp_group",
            return_value=SimpleNamespace(world_size=2, all_gather=Mock(return_value=gathered_prefill_kv)),
        ) as group,
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs",
            wraps=_gather_prefill_cache_inputs,
        ) as gather,
        patch.object(AscendSFAImpl, "exec_kv", autospec=True, return_value="written") as base_exec_kv,
    ):
        result = impl.exec_kv(kv_no_split, None, None, kv_cache, slots, attn_metadata)

    assert result == "written"
    gather.assert_called_once_with((kv_no_split,), slots, 1)
    group.return_value.all_gather.assert_called_once()
    torch.testing.assert_close(group.return_value.all_gather.call_args.args[0], kv_no_split[1:])
    base_exec_kv.assert_called_once()
    args = base_exec_kv.call_args.args
    assert args[0] is impl and args[3] is None and args[2] is None
    assert args[4] is kv_cache and args[6] is attn_metadata
    torch.testing.assert_close(args[1], gathered_kv)
    torch.testing.assert_close(args[5], gathered_slots)


@pytest.mark.parametrize("packed_cache", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 1])
def test_sfa_pcp_gathers_nope_kv_with_padded_rank_slots(packed_cache: bool, num_decode_tokens: int) -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = packed_cache
    impl.enable_sparse_sfa_turboquant = False
    attn_metadata = SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_prefills=1,
        num_input_tokens=8,
    )
    kv_no_split = torch.arange(5, dtype=torch.float32).view(5, 1)
    # Slot mapping keeps eight positions per PCP rank, while NoPE KV has five.
    slots = torch.tensor([0, 1, 2, 3, 4, -1, -1, -1, 8, 9, 10, 11, 12, -1, -1, -1])
    slots[8 : 8 + num_decode_tokens] = slots[:num_decode_tokens]
    attn_metadata.uses_pcp_sharded_layout = True
    attn_metadata.pcp_slot_mapping = slots
    kv_cache = (torch.empty(1),)

    def all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        padded_kv = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0]).view(8, 1).float()
        torch.testing.assert_close(tensor, padded_kv[num_decode_tokens:])
        remote_kv = torch.arange(10 + num_decode_tokens, 18, dtype=torch.float32).view(-1, 1)
        return torch.cat((tensor, remote_kv), dim=dim)

    group = SimpleNamespace(
        world_size=2,
        all_gather=all_gather,
    )

    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch.object(AscendSFAImpl, "exec_kv", autospec=True, return_value="written") as base_exec_kv,
    ):
        result = impl.exec_kv(kv_no_split, None, None, kv_cache, slots, attn_metadata)

    assert result == "written"
    gathered_kv = base_exec_kv.call_args.args[1]
    gathered_slots = base_exec_kv.call_args.args[5]
    expected_kv = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0, *range(10 + num_decode_tokens, 18)]).view(-1, 1).float()
    torch.testing.assert_close(gathered_kv, expected_kv)
    torch.testing.assert_close(gathered_slots, torch.cat((slots[:8], slots[8 + num_decode_tokens :])))


def test_sfa_pcp_packed_kv_writes_gather_order_slots() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.sfa_qsfa_packed_kv_head_dim = 4
    metadata = SimpleNamespace(num_actual_tokens=1)
    k_pe = torch.ones(2, 1, 1)
    k_nope = torch.ones(2, 1, 2)
    knope_scale = torch.ones(2, 1, 1)
    kv_cache = (torch.empty(1, 1, 1, 4),)

    with patch("vllm_ascend.attention.sfa_v1.DeviceOperator.scatter_cache") as scatter:
        impl._store_parallel_kv(
            k_pe,
            k_nope,
            knope_scale,
            kv_cache,
            torch.tensor([4, 8]),
            metadata,
        )

    scatter.assert_called_once()
    torch.testing.assert_close(scatter.call_args.args[1], torch.tensor([[4], [8]]))
    assert scatter.call_args.args[2].shape[0] == 2


def _build_pcp_dcp_slot_metadata(metadata, group):
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    context = SimpleNamespace(
        global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([metadata.pcp_has_global_prefill]), num_reqs=1),
        global_block_tables=(torch.zeros(1, 1, dtype=torch.int32),),
        global_block_table_num_blocks=None,
    )
    common = SimpleNamespace(pcp_context=context, pcp_cache_group_idx=0)
    with (
        patch.object(builder, "_get_dcp_local_block_table", return_value=context.global_block_tables[0]),
        patch.object(builder, "_build_pcp_ordered_indexer_slot_mapping", return_value=metadata.pcp_slot_mapping),
        patch.object(builder, "_build_with_metadata_view", return_value=metadata),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=group),
    ):
        return builder.build(0, common)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "num_decode_tokens,num_tokens,local_prefill,global_prefill",
    [
        (0, 5, True, True),
        (2, 5, True, True),
        (2, 8, True, True),
        (2, 8, False, True),
        (5, 5, False, True),
        (5, 5, False, False),
        (0, 0, False, True),
    ],
)
def test_sfa_pcp_dcp_builder_slots_match_kv_gather(rank, num_decode_tokens, num_tokens, local_prefill, global_prefill):
    expanded_slots = torch.arange(2 * num_tokens, dtype=torch.int32).view(2, num_tokens)
    expanded_slots[1, :num_decode_tokens] = -1
    if num_tokens > num_decode_tokens:
        expanded_slots[:, -1] = -1
    if not local_prefill:
        expanded_slots[rank, num_decode_tokens:] = -1
    expanded_slots = expanded_slots.flatten()
    original_slots = expanded_slots.clone()
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_input_tokens = num_tokens
    metadata.num_decode_tokens = num_decode_tokens
    metadata.num_prefills = int(local_prefill)
    metadata.pcp_has_global_prefill = global_prefill
    metadata.slot_mapping = expanded_slots[:num_tokens]
    metadata.pcp_slot_mapping = expanded_slots.flip(0)
    metadata.dcp_context = SimpleNamespace(slot_mapping=expanded_slots)
    indexer_slots = metadata.pcp_slot_mapping
    kv = torch.arange(2 * num_tokens).view(2, num_tokens, 1)
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        all_gather=Mock(return_value=kv[:, num_decode_tokens:].reshape(-1, 1)),
    )

    assert _build_pcp_dcp_slot_metadata(metadata, group) is metadata
    group.all_gather.assert_not_called()
    with patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group):
        _, gathered_slots = _gather_prefill_cache_inputs((kv[rank],), expanded_slots, num_decode_tokens)
    torch.testing.assert_close(metadata.slot_mapping, gathered_slots)
    assert metadata.dcp_context.slot_mapping is expanded_slots
    assert metadata.pcp_slot_mapping is indexer_slots
    torch.testing.assert_close(expanded_slots, original_slots)


@pytest.mark.parametrize("num_decode_tokens", [0, 2, 5])
@pytest.mark.parametrize(
    "cache_mode,rope_dim,num_input_tokens", [("c8", 0, 5), ("c8", 0, 8), ("c8", 2, 5), ("tq4", 2, 5)]
)
def test_native_pcp_dcp_packed_kv_preserves_gathered_payload_addresses(
    num_decode_tokens, cache_mode, rope_dim, num_input_tokens
):
    impl = AscendSFAPCPDCPImpl.__new__(AscendSFAPCPDCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = rope_dim
    impl.kv_lora_rank = 3
    impl.num_kv_heads = 1
    impl.enable_sparse_sfa_c8 = cache_mode == "c8"
    impl.enable_sparse_sfa_turboquant = cache_mode == "tq4"
    impl.turboquant = Mock()
    impl.c8_cache_dtype = torch.int8
    impl.sfa_qsfa_tile_size = 32
    impl.sfa_qsfa_packed_kv_head_dim = 4 + rope_dim
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(3), variance_epsilon=1e-6)
    impl._record_dcp_kv_gather_context = Mock()
    local = torch.arange(1, 5 * (3 + rope_dim) + 1, dtype=torch.float32).reshape(5, 3 + rope_dim)
    peer_prefill = local[num_decode_tokens:] + 10
    expected_payload = torch.cat((local, peer_prefill)).to(torch.int8)
    expected_payload = torch.cat((expected_payload, torch.ones(len(expected_payload), 1, dtype=torch.int8)), dim=-1)
    padding_slots = [-1] * (num_input_tokens - len(local))
    expanded_slots = torch.tensor(
        list(range(5))
        + padding_slots
        + [-1] * num_decode_tokens
        + list(range(5, len(expected_payload)))
        + padding_slots,
        dtype=torch.int32,
    )
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_decode_tokens = num_decode_tokens
    metadata.num_prefills = int(num_decode_tokens < len(local))
    metadata.pcp_has_global_prefill = metadata.num_prefills > 0
    metadata.uses_pcp_sharded_layout = True
    metadata.num_input_tokens = num_input_tokens
    metadata.num_actual_tokens = len(local)
    metadata.slot_mapping = expanded_slots[:num_input_tokens]
    metadata.pcp_slot_mapping = expanded_slots
    metadata.cos = metadata.sin = torch.empty(len(local), rope_dim) if rope_dim else None
    metadata.attn_state = AscendAttentionState.ChunkedPrefill
    metadata.dcp_context = SimpleNamespace(slot_mapping=expanded_slots)
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        all_gather=Mock(side_effect=lambda values, dim: torch.cat((values, values + 10), dim)),
    )
    cache = torch.full((1, 12, 1, 4 + rope_dim), -99, dtype=torch.int8)
    hidden = torch.zeros(len(local), 2)
    impl.q_lora_rank = 2
    impl.preprocess_type = PreprocessType.NATIVE
    impl.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=2))
    impl.g_proj = impl.layerwise_kv_cache_hook = None
    impl.has_indexer = True
    impl._is_mtp_layer = impl.skip_topk = impl.use_index_cache = False
    impl.layer_name = "layer"
    impl._compose_sfa_kv_cache = Mock(return_value=(cache,))
    impl._get_indexer_attn_metadata = Mock(return_value=object())
    impl._get_parallel_forward_context = Mock(
        side_effect=lambda *_args: SimpleNamespace(
            actual_seq_lengths_query=[len(local)],
            actual_seq_lengths_key=[len(local)],
            kv_slot_mapping=impl._get_sfa_kv_slot_mapping(metadata),
            topk_num_tokens=len(local),
            gather_full_o_proj=False,
        )
    )
    impl.fused_qkv_a_proj = Mock(return_value=(torch.cat((hidden, local), dim=-1),))
    impl.q_a_layernorm = lambda values: values
    impl._q_proj_and_k_up_proj = Mock(return_value=(hidden.unsqueeze(1), hidden.unsqueeze(1)))
    impl.rope_single = lambda values, *_args: values
    impl._record_query_gather_context = Mock()
    impl._prepare_indexer_metadata = Mock()
    impl.indexer = Mock(return_value=torch.zeros(len(local), 1, 1, dtype=torch.int32))
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = lambda values: values
    impl._finalize_o_proj = lambda values, output, gather_full_o_proj: output.copy_(values)

    def quantize(values, *_args, **_kwargs):
        rows = len(values)
        return (
            values[..., 3:].to(torch.int8),
            values[..., :3].to(torch.int8),
            torch.ones(rows, 1, 1, 1, dtype=torch.int8),
        )

    def scatter(target, indices, values):
        indices = indices.flatten().long()
        valid = indices >= 0
        target.index_copy_(0, indices[valid], values[valid])

    expected = torch.full_like(cache.view(-1, 4 + rope_dim), -99)
    expected[: len(expected_payload)] = expected_payload
    original_slots = expanded_slots.clone()

    _build_pcp_dcp_slot_metadata(metadata, group)
    write_slots = metadata.slot_mapping
    torch.testing.assert_close(write_slots[write_slots >= 0], torch.arange(len(expected_payload), dtype=torch.int32))

    # Reuse metadata across layers: gather reads the original DCP mapping,
    # while cache stores use the builder's mapping without replacing it.
    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch("vllm_ascend.attention.sfa_v1.custom_kv_rmsnorm_rope", side_effect=quantize),
        patch("vllm_ascend.attention.sfa_v1.turboquant_kv_rmsnorm_rope", side_effect=quantize),
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.scatter_cache", side_effect=scatter),
        patch.object(
            sparse_mla, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
        ),
        patch.object(sparse_mla, "wait_for_kv_layer_from_connector"),
        patch.object(sparse_mla, "notify_kv_cache_written"),
        patch.object(sparse_mla, "attention_transfer_window", side_effect=nullcontext),
        patch.object(sparse_mla, "maybe_save_kv_layer_to_connector"),
    ):
        for _ in range(2):
            cache.fill_(-99)
            AscendSFAImpl.forward(impl, "layer", hidden, (cache,), metadata, torch.empty_like(hidden))
            torch.testing.assert_close(cache.view(-1, 4 + rope_dim), expected, rtol=0, atol=0)
            assert metadata.slot_mapping is write_slots
            assert metadata.dcp_context.slot_mapping is expanded_slots
            torch.testing.assert_close(expanded_slots, original_slots, rtol=0, atol=0)


def test_sfa_pcp_gathers_nope_kv_when_local_rank_has_no_prefill() -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_prefills=0,
        num_input_tokens=8,
        pcp_has_global_prefill=True,
    )
    slots = torch.tensor([-1] * 8 + [8, 9, 10, -1, -1, -1, -1, -1])
    metadata.uses_pcp_sharded_layout = True
    metadata.pcp_slot_mapping = slots
    group = SimpleNamespace(
        world_size=2,
        all_gather=lambda tensor, dim=0: torch.cat((tensor, torch.arange(10, 18).view(8, 1).float()), dim=dim),
    )

    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch.object(AscendSFAImpl, "exec_kv", autospec=True, return_value="written") as base_exec_kv,
    ):
        result = impl.exec_kv(torch.empty(0, 1), None, None, (torch.empty(1),), slots, metadata)

    assert result == "written"
    gathered_kv = base_exec_kv.call_args.args[1]
    gathered_slots = base_exec_kv.call_args.args[5]
    assert gathered_kv.shape == (16, 1)
    torch.testing.assert_close(gathered_slots, slots)
    torch.testing.assert_close(gathered_kv[gathered_slots >= 0], torch.tensor([[10.0], [11.0], [12.0]]))


def test_sfa_pcp_o_proj_switch_slices_the_tp_local_weight_by_pcp_rank() -> None:
    AscendSFAPCPImpl.o_proj_full_pools.clear()
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
    impl._enable_o_proj_full_weight_switch()

    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    input_ = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(input_, full_weight, impl.o_proj.bias)
    pcp_group.all_reduce = lambda _: torch.nn.functional.linear(input_, full_weight, bias=None)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group):
        result = impl._finalize_o_proj(input_, torch.empty_like(expected), gather_full_o_proj=False)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
def test_sfa_pcp_sharded_decode_preserves_token_owners(rank, tp_size, padding, quantized, num_speculative_tokens):
    group = SimpleNamespace(world_size=2, rank_in_group=rank, device_group=object(), all_reduce=Mock())
    impl = _make_pcp_o_proj_impl(sharded_decode=True, num_speculative_tokens=num_speculative_tokens)
    impl.o_proj_weight_switch_config = WeightSwitchConfig.from_group(group, shard_axis="input")
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    impl.o_proj.weight = torch.nn.Parameter(weight[:, rank * 2 : (rank + 1) * 2].clone(), requires_grad=False)
    impl._enable_o_proj_full_weight_switch()
    quant_bias = torch.tensor([0.25, 0.5, 0.75]) if quantized else torch.zeros(3)
    if quantized:
        method = MagicMock(spec=AscendLinearScheme)

        def apply(layer, x, bias=None, tp_rank=0):
            return torch.nn.functional.linear(x, layer.weight) + (quant_bias if tp_rank == 0 else 0)

        method.apply.side_effect = apply
        impl._get_o_proj_weight_switch_method = Mock(return_value=method)
    buffer_names = (
        "_pcp_o_proj_ag_in_buf",
        "_pcp_o_proj_ag_out_buf",
        "_pcp_o_proj_rs_in_buf",
        "_pcp_o_proj_rs_out_buf",
    )
    buffer_ptrs = tuple(getattr(impl, name).data_ptr() for name in buffer_names)
    for num_rows in [1, 2, impl._pcp_o_proj_token_capacity]:
        inputs = [torch.arange(num_rows * 4, dtype=torch.float32).view(num_rows, 4) + 1 + i * 20 for i in range(2)]
        if padding:
            inputs[1][-1].zero_()
        global_input = torch.cat(inputs, dim=0)
        full_projection = torch.nn.functional.linear(global_input, weight) + quant_bias

        def gather(destination, source, group, inputs=inputs, global_input=global_input):
            torch.testing.assert_close(source, inputs[rank])
            destination.copy_(global_input)

        def scatter(
            destination, source, group, global_input=global_input, full_projection=full_projection, num_rows=num_rows
        ):
            expected_partial = torch.nn.functional.linear(
                global_input[:, rank * 2 : (rank + 1) * 2], impl.o_proj.weight
            )
            if quantized and rank == 0:
                expected_partial += quant_bias
            torch.testing.assert_close(source, expected_partial)
            destination.copy_(full_projection[rank * num_rows : (rank + 1) * num_rows])

        tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=0, all_reduce=Mock(side_effect=lambda x: x + 3))
        expected = full_projection[rank * num_rows : (rank + 1) * num_rows] + impl.o_proj.bias
        if tp_size > 1:
            expected += 3
        with (
            patch("vllm_ascend.attention.context_parallel.sfa_cp.dist.all_gather_into_tensor", side_effect=gather),
            patch("vllm_ascend.attention.context_parallel.sfa_cp.dist.reduce_scatter_tensor", side_effect=scatter),
            patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group),
        ):
            actual = impl._finalize_o_proj(inputs[rank], torch.empty_like(expected), gather_full_o_proj=False)
        torch.testing.assert_close(actual, expected)
        assert tuple(getattr(impl, name).data_ptr() for name in buffer_names) == buffer_ptrs
        group.all_reduce.assert_not_called()
        assert tp_group.all_reduce.call_count == int(tp_size > 1)
        if quantized:
            assert method.apply.call_args.kwargs["tp_rank"] == (0 if rank == 0 else 1)


@pytest.mark.parametrize(
    "attn_state,global_has_prefill,weight_switch_enabled,expected_gather",
    [
        (AscendAttentionState.ChunkedPrefill, True, True, True),
        (AscendAttentionState.ChunkedPrefill, False, True, True),
        (AscendAttentionState.DecodeOnly, False, True, False),
        (AscendAttentionState.SpecDecoding, False, True, False),
        (AscendAttentionState.DecodeOnly, True, True, True),
        (AscendAttentionState.SpecDecoding, True, True, True),
        (AscendAttentionState.ChunkedPrefill, True, False, False),
        (AscendAttentionState.DecodeOnly, True, False, False),
    ],
)
def test_sfa_pcp_context_gathers_weight_for_global_prefill(
    attn_state, global_has_prefill, weight_switch_enabled, expected_gather
) -> None:
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl._o_proj_weight_switch_enabled = weight_switch_enabled
    impl._all_gather_o_proj_full_weight = MagicMock()
    base_context = SFAForwardContext(
        actual_seq_lengths_query=torch.empty(0),
        actual_seq_lengths_key=torch.empty(0),
        kv_slot_mapping=torch.empty(0),
        topk_num_tokens=0,
    )

    with patch.object(AscendSFAImpl, "_get_parallel_forward_context", return_value=base_context):
        context = impl._get_parallel_forward_context(
            SimpleNamespace(attn_state=attn_state, pcp_has_global_prefill=global_has_prefill),
            1,
            torch.empty(1),
        )
    assert context.gather_full_o_proj is expected_gather
    if expected_gather:
        impl._all_gather_o_proj_full_weight.assert_called_once_with()
    else:
        impl._all_gather_o_proj_full_weight.assert_not_called()


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
            patch("vllm_ascend.attention.context_parallel.sfa_cp.is_dcp_enabled", return_value=True) as dcp,
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
        assert builder.use_dcp
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


@pytest.mark.parametrize(
    "is_consumer,is_producer,recompute", [(True, False, True), (True, False, False), (False, True, True)]
)
@pytest.mark.parametrize("query_lens", [[1, 1], [3, 3], [3, 5]])
def test_sfa_dcp_split_uses_builder_config_without_current_context(is_consumer, is_producer, recompute, query_lens):
    builder = _make_builder()
    builder.use_dcp = True
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
            "vllm_ascend.attention.context_parallel.sfa_cp.is_dcp_enabled",
            side_effect=AssertionError("use cached DCP state"),
        ),
        patch.object(builder, "_get_dcp_local_block_table", return_value=blocks),
        patch.object(builder, "_build_block_table_replicated_view", return_value=blocks),
        patch.object(builder, "_build_slot_mapping_replicated_view", return_value=slots),
        patch.object(builder, "_build_compact_kv_gather_metadata", return_value=(torch.arange(4), blocks)) as gather,
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
    impl.enable_sparse_sfa_turboquant = False
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


@pytest.mark.parametrize(
    "impl_cls,local_prefill,global_prefill,expect_gathered",
    [
        (AscendSFADCPImpl, False, False, False),
        (AscendSFADCPImpl, True, False, False),
        (AscendSFAPCPDCPImpl, False, False, False),
        (AscendSFAPCPDCPImpl, True, False, True),
        (AscendSFAPCPDCPImpl, False, True, True),
    ],
)
def test_sfa_dcp_slot_mapping_matches_parallel_layout(impl_cls, local_prefill, global_prefill, expect_gathered):
    impl = impl_cls.__new__(impl_cls)
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_input_tokens = 2
    metadata.num_prefills = int(local_prefill)
    metadata.pcp_has_global_prefill = global_prefill
    full_slots = torch.tensor([3200, -1, 3201, -1], dtype=torch.int32)
    metadata.dcp_context = SimpleNamespace(slot_mapping=full_slots)
    metadata.slot_mapping = torch.tensor([3200, 3201, -1], dtype=torch.int32)

    result = impl._get_sfa_kv_slot_mapping(metadata)

    if expect_gathered:
        assert result is metadata.slot_mapping
    else:
        assert result.data_ptr() == full_slots.data_ptr()
        torch.testing.assert_close(result, full_slots[:2])


@pytest.mark.parametrize("is_kv_consumer,sfa_c8", [(False, False), (True, True)])
@pytest.mark.parametrize("quantized_indexer", [False, True])
def test_sfa_pcp_keeps_prolog_v3_enabled(is_kv_consumer, sfa_c8, quantized_indexer):
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    quant_cls = AscendW8A8DynamicLinearMethod
    impl.fused_qkv_a_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_cls.__new__(quant_cls)))
    impl.q_proj = SimpleNamespace(_chunk_size=0)
    impl.q_a_layernorm = object()
    impl.kv_a_layernorm = object()
    impl.qk_rope_head_dim = 64
    impl.is_kv_consumer = is_kv_consumer
    impl.enable_sparse_sfa_c8 = sfa_c8
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_mlapo = False
    impl.has_indexer = True
    impl.indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight_scale=torch.ones(1)) if quantized_indexer else SimpleNamespace()
    )
    with patch.object(impl, "_try_enable_type", return_value=True) as prepare_weights:
        expected = PreprocessType.PROLOG_V3 if quantized_indexer else PreprocessType.NATIVE
        assert impl._resolve_preprocess_type(torch.bfloat16) == expected
    if quantized_indexer:
        prepare_weights.assert_called_once_with(PreprocessType.PROLOG_V3, torch.bfloat16)
    else:
        prepare_weights.assert_not_called()


@pytest.mark.parametrize(
    "cache_dtype,pcp_size,rank,num_decode_tokens,num_tokens",
    [
        (torch.bfloat16, 2, 0, 0, 4),  # Prefill only.
        (torch.bfloat16, 4, 3, 2, 4),  # Mixed on a nonzero PCP rank.
        (torch.bfloat16, 2, 1, 4, 4),  # Decode only.
        (torch.int8, 2, 0, 0, 4),  # C8 prefill.
        (torch.int8, 2, 1, 2, 4),  # C8 mixed.
        (torch.float8_e4m3fn, 8, 7, 4, 7),  # FP8 with PCP8 padding.
        (torch.int8, 8, 0, 4, 7),  # Rank-zero decode with PCP8 padding.
    ],
)
def test_sfa_pcp_prolog_gathers_only_prefill_kv(cache_dtype, pcp_size, rank, num_decode_tokens, num_tokens):
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    c8 = impl.enable_sparse_sfa_c8 = cache_dtype != torch.bfloat16
    width = 656 if c8 else 5
    cache_blocks = (pcp_size * num_tokens + 7) // 8
    cache = (
        (torch.empty((cache_blocks, 8, 1, width), dtype=cache_dtype), torch.empty((cache_blocks, 8, 1, 128)))
        if c8
        else (torch.empty((cache_blocks, 8, 1, 3)), torch.empty((cache_blocks, 8, 1, 2)))
    )
    hidden = torch.zeros((num_tokens, 1))
    slots = torch.stack([torch.arange(num_tokens) + i * num_tokens for i in range(pcp_size)])
    slots[:, :num_decode_tokens] = torch.arange(num_decode_tokens)
    slots[1:, :num_decode_tokens] = -1
    rows = num_tokens - num_decode_tokens
    if rows:
        slots[:, -1] = -1
    original_slots = slots.clone()
    dtype = torch.int8 if c8 else cache[0].dtype
    gathered = (torch.arange(pcp_size * rows * width) % 256 - 128).to(dtype).view(-1, width)
    packed = gathered[rank * rows : (rank + 1) * rows]
    group = SimpleNamespace(world_size=pcp_size, rank_in_group=rank, all_gather=Mock(return_value=gathered))
    # Returning this exact tuple also preserves local Q and quantized Q scales.
    output = (hidden, hidden + 1, hidden + 2, (hidden + 3, hidden + 4))
    expected_local_slots = slots[rank].clone()
    expected_local_slots[:num_decode_tokens] = slots[0, :num_decode_tokens]
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_prefills=int(rows > 0),
        num_input_tokens=num_tokens,
        uses_pcp_sharded_layout=True,
        pcp_slot_mapping=slots.flatten(),
        pcp_has_global_prefill=False,
        pcp_prolog_local_slots=expected_local_slots if rows else None,
        pcp_prolog_global_slots=slots[:, num_decode_tokens:].reshape(-1) if rows else None,
    )
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=group),
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output) as prolog,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.copy_pcp_kv_cache", return_value=packed) as read,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.reshape_and_cache") as write,
    ):
        result = impl._sfa_preprocess_prolog_v3(
            hidden,
            cache,
            hidden,
            hidden,
            slots.flatten(),
            attn_metadata=metadata,
        )
    assert result is output
    assert prolog.call_args.args[0] is hidden
    # Upstream masks replicated decode copies outside rank 0, but each
    # rank must still write its locally computed decode KV to its own cache.
    torch.testing.assert_close(prolog.call_args.args[4], expected_local_slots)
    torch.testing.assert_close(slots, original_slots)
    if not rows:
        read.assert_not_called()
        group.all_gather.assert_not_called()
        write.assert_not_called()
        return
    read.assert_called_once()
    main_cache, local_slots = read.call_args.args  # No custom-kernel write call.
    assert len(main_cache) == (1 if c8 else 2)
    assert main_cache[0] is cache[0]
    torch.testing.assert_close(local_slots, slots[rank, num_decode_tokens:])
    group.all_gather.assert_called_once_with(packed, dim=0)
    write.assert_called_once()
    args = write.call_args.kwargs
    torch.testing.assert_close(args["slot_mapping"], slots[:, num_decode_tokens:].flatten())
    if c8:
        assert args["key"] is args["value"] and args["key_cache"] is args["value_cache"]
        assert args["key_cache"].dtype == torch.int8
        assert args["key_cache"].data_ptr() == cache[0].data_ptr()
        torch.testing.assert_close(args["key"], gathered.unsqueeze(1), rtol=0, atol=0)
    else:
        assert args["key_cache"] is cache[0] and args["value_cache"] is cache[1]
        torch.testing.assert_close(args["key"], gathered[:, :3].unsqueeze(1))
        torch.testing.assert_close(args["value"], gathered[:, 3:].unsqueeze(1))


def _make_sfa_split_builder(use_pcp: bool, threshold: int = 1) -> AscendSFAMetadataBuilder:
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    builder.speculative_config = None
    builder.use_pcp = use_pcp
    builder.is_pcp_decode_sharded = False
    builder.decode_threshold = threshold
    builder.nope = False
    builder.kernel_block_size = 128
    builder.metadata_cls = AscendSFAMetadata
    builder.model_config = SimpleNamespace(get_head_size=lambda: 64)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *_: None)
    return builder


def _make_sfa_split_common(query_lens, flags, actual, threshold, capture=False, padded_tokens=None):
    offsets = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0)
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens) if padded_tokens is None else padded_tokens
    seq_lens = torch.full((num_reqs,), 100, dtype=torch.int32)
    slots = torch.arange(num_tokens)
    slots[actual:] = -1
    return SimpleNamespace(
        context_parallel_metadata=None,
        pcp_context=None,
        pcp_cache_group_idx=None,
        num_reqs=num_reqs,
        num_actual_tokens=actual,
        num_input_tokens=num_tokens,
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=slots,
        positions=torch.arange(num_tokens),
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        seq_lens_cpu=seq_lens,
        causal=True,
        max_query_len=max(query_lens, default=0),
        max_seq_len=100,
        decode_token_per_req=threshold,
        is_prefilling=torch.tensor(flags, dtype=torch.bool),
        attn_state=AscendAttentionState.DecodeOnly if capture else AscendAttentionState.ChunkedPrefill,
    )


@pytest.mark.parametrize("use_pcp", [False, True])
@pytest.mark.parametrize(
    "query_lens,flags,actual,threshold,capture,expected_without,expected_with",
    [
        ([1], [True], 1, 1, False, (1, 0, 1), (0, 1, 0)),
        ([1, 1, 1], [False, True, True], 3, 1, False, (3, 0, 3), (1, 2, 1)),
        ([1, 1, 1, 1], [False, False], 2, 1, True, (4, 0, 2), (4, 0, 2)),
        ([4, 4, 4, 4], [False, False], 8, 4, True, (4, 0, 8), (4, 0, 8)),
        ([1, 2, 0], [False, True], 3, 1, False, (1, 2, 1), (1, 2, 1)),
        ([], [], 0, 1, False, (0, 0, 0), (0, 0, 0)),
    ],
)
def test_sfa_split_counts_with_and_without_pcp(
    use_pcp, query_lens, flags, actual, threshold, capture, expected_without, expected_with
):
    builder = _make_sfa_split_builder(use_pcp, threshold)
    common = _make_sfa_split_common(query_lens, flags, actual, threshold, capture)
    with (
        patch("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", return_value=False),
        patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=(torch.ones(common.num_input_tokens, 64), torch.zeros(common.num_input_tokens, 64)),
        ),
    ):
        meta = builder.build_for_cudagraph_capture(common) if capture else builder.build(0, common)
    assert (meta.num_decodes, meta.num_prefills, meta.num_decode_tokens) == (
        expected_with if use_pcp else expected_without
    )


@pytest.mark.parametrize("global_has_prefill", [False, True])
def test_sfa_pcp_metadata_keeps_global_prefill_when_local_rank_is_empty(global_has_prefill):
    builder = _make_sfa_split_builder(use_pcp=True)
    common = _make_sfa_split_common([1], [False], 1, 1, padded_tokens=2)
    common.slot_mapping = torch.tensor([0, -1, -1, -1])
    context = SimpleNamespace(
        global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([global_has_prefill])),
    )
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=(torch.ones(2, 64), torch.zeros(2, 64)),
        ),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=group),
    ):
        common.pcp_context = context
        metadata = builder.build(0, common)
        draft_metadata = builder.build_for_drafting(common, draft_index=0)
    assert metadata.num_prefills == 0
    assert metadata.uses_pcp_sharded_layout
    assert not draft_metadata.uses_pcp_sharded_layout
    assert metadata.pcp_has_global_prefill is global_has_prefill
    assert not draft_metadata.pcp_has_global_prefill
    if global_has_prefill:
        torch.testing.assert_close(metadata.pcp_prolog_local_slots, torch.tensor([0, -1]))
        torch.testing.assert_close(metadata.pcp_prolog_global_slots, torch.tensor([-1, -1]))
    else:
        assert metadata.pcp_prolog_local_slots is None
        assert metadata.pcp_prolog_global_slots is None


def test_sfa_pcp_builder_prepares_local_prolog_slots():
    builder = _make_sfa_split_builder(use_pcp=True)
    common = _make_sfa_split_common([1, 2], [False, True], 3, 1, padded_tokens=4)
    common.slot_mapping = torch.tensor([0, 1, 2, -1, -1, -1, 3, -1])
    context = SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([False, True])))
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=(torch.ones(4, 64), torch.zeros(4, 64)),
        ),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=group),
    ):
        common.pcp_context = context
        metadata = builder.build(0, common)
    assert metadata.num_decode_tokens == 1
    torch.testing.assert_close(metadata.pcp_prolog_local_slots, torch.tensor([0, -1, 3, -1]))
    torch.testing.assert_close(metadata.pcp_prolog_global_slots, torch.tensor([1, 2, -1, -1, 3, -1]))


def test_sfa_pcp_empty_local_prefill_joins_kv_gathers():
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 64
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        num_prefills=0,
        num_input_tokens=2,
        uses_pcp_sharded_layout=True,
        pcp_slot_mapping=torch.tensor([0, 1, -1, -1], dtype=torch.int64),
        pcp_has_global_prefill=True,
        pcp_prolog_local_slots=torch.tensor([0, -1], dtype=torch.int64),
        pcp_prolog_global_slots=torch.tensor([1, -1], dtype=torch.int64),
    )
    hidden = torch.zeros((2, 1))
    cache = (torch.empty((1, 8, 1, 3)), torch.empty((1, 8, 1, 2)))
    slots = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    packed = torch.zeros((1, 5))
    gathered = torch.zeros((2, 5))
    group = SimpleNamespace(world_size=2, rank_in_group=1, all_gather=Mock(return_value=gathered))
    output = (hidden, hidden, hidden, None)
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=group),
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output) as preprocess,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.copy_pcp_kv_cache", return_value=packed) as copy_cache,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.reshape_and_cache") as write_cache,
    ):
        assert impl._sfa_preprocess_prolog_v3(hidden, cache, hidden, hidden, slots, attn_metadata=metadata) is output
    torch.testing.assert_close(preprocess.call_args.args[4], torch.tensor([0, -1]))
    torch.testing.assert_close(copy_cache.call_args.args[1], torch.tensor([-1]))
    group.all_gather.assert_called_once_with(packed, dim=0)
    torch.testing.assert_close(write_cache.call_args.kwargs["slot_mapping"], torch.tensor([1, -1]))

    gathered_hidden = torch.zeros((3, 1))
    gathered_slots = torch.tensor([0, 1, -1], dtype=torch.int64)
    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs",
            return_value=((gathered_hidden, gathered_hidden, gathered_hidden), gathered_slots),
        ) as gather,
        patch.object(AscendSFAImpl, "exec_kv", return_value="written") as base_write,
    ):
        assert impl.exec_kv(hidden, hidden, hidden, cache, slots, metadata) == "written"
    gather.assert_called_once_with((hidden, hidden, hidden), slots, 1)
    base_write.assert_called_once()


def test_sfa_pcp_empty_global_slots_skips_kv_gather():
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    hidden = torch.empty((0, 1))
    slots = torch.empty(0, dtype=torch.int64)
    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_prefills=0,
        num_input_tokens=1,
        uses_pcp_sharded_layout=True,
        pcp_slot_mapping=torch.full((2,), -1),
        pcp_has_global_prefill=True,
        pcp_prolog_local_slots=slots,
        pcp_prolog_global_slots=slots,
    )
    output = (hidden, hidden, hidden, None)
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group") as group,
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output) as prolog,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.copy_pcp_kv_cache") as copy_cache,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.reshape_and_cache") as write_cache,
    ):
        assert (
            impl._sfa_preprocess_prolog_v3(hidden, (hidden, hidden), hidden, hidden, slots, attn_metadata=metadata)
            is output
        )
    assert prolog.call_args.args[4] is slots
    group.assert_not_called()
    copy_cache.assert_not_called()
    write_cache.assert_not_called()


def test_sfa_pcp_padded_decode_skips_kv_gather():
    num_decode_tokens = 2
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.is_pcp_decode_sharded = False
    num_input_tokens = num_decode_tokens * 2
    hidden = torch.zeros((num_input_tokens, 3))
    slots = torch.cat((torch.arange(num_decode_tokens), torch.full((num_decode_tokens,), -1)))
    cache = (torch.empty(0), torch.empty(0))
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_prefills=0,
        num_input_tokens=num_input_tokens,
        uses_pcp_sharded_layout=True,
        pcp_slot_mapping=torch.cat((slots, torch.full((num_input_tokens,), -1))),
        pcp_has_global_prefill=False,
    )
    output = (hidden, hidden, hidden, None)
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group") as group,
        patch("vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs") as gather,
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output),
        patch.object(AscendSFAImpl, "exec_kv", return_value="written") as base_write,
    ):
        assert impl._sfa_preprocess_prolog_v3(hidden, cache, hidden, hidden, slots, attn_metadata=metadata) is output
        assert impl.exec_kv(hidden, hidden, hidden, cache, slots, metadata) == "written"
    group.assert_not_called()
    gather.assert_not_called()
    torch.testing.assert_close(base_write.call_args.args[4], slots)


def test_sfa_pcp_replicated_draft_decode_skips_kv_gather():
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    # Sharded-decode config, but a draft decode built without a PCP context
    # keeps an unexpanded slot mapping and runs replicated on every rank.
    impl.is_pcp_decode_sharded = True
    hidden = torch.zeros((2, 3))
    slots = torch.tensor([5, 7])
    cache = (torch.empty(0), torch.empty(0))
    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_prefills=0,
        num_input_tokens=2,
        uses_pcp_sharded_layout=False,
        pcp_slot_mapping=slots,
        pcp_has_global_prefill=False,
    )
    output = (hidden, hidden, hidden, None)
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group") as group,
        patch("vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs") as gather,
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output),
        patch.object(AscendSFAImpl, "exec_kv", return_value="written") as base_write,
    ):
        assert impl._sfa_preprocess_prolog_v3(hidden, cache, hidden, hidden, slots, attn_metadata=metadata) is output
        assert impl.exec_kv(hidden, hidden, hidden, cache, slots, metadata) == "written"
    group.assert_not_called()
    gather.assert_not_called()
    torch.testing.assert_close(base_write.call_args.args[4], slots)


@pytest.mark.parametrize("dcp_size", [8, 16])
def test_sfa_pcp_dcp_builder_preserves_causal_multi_token_support(dcp_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=8, decode_context_parallel_size=dcp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )

    def init_base(self, spec, layers, vllm_config, device, metadata_cls, supports_dcp_with_varlen):
        # DSpark's causal target uses this capability in MLA's upstream guard.
        assert supports_dcp_with_varlen
        assert vllm_config is config

    with patch.object(AscendSFADCPMetadataBuilder, "__init__", init_base):
        builder = AscendSFAPCPDCPMetadataBuilder(object(), [], config, torch.device("cpu"))
    assert builder.pcp_indexer_slot_mapping_buf.shape == (256,)
