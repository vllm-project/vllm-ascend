# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import vllm.distributed as vllm_distributed
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs

import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPMetadataBuilder,
    AscendSFAPCPDCPMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFADCPMetadata,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    PreprocessType,
)
from vllm_ascend.quantization.methods import AscendLinearScheme, AscendW8A8DynamicLinearMethod
from vllm_ascend.weight_switch import (
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchLoadState,
    WeightSwitchMixin,
)
from vllm_ascend.weight_switch.o_proj import OProjectionMode, ParallelOProjection


def _make_pcp_sfa_impl(dcp_enabled=False):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
    impl.dcp_enabled = dcp_enabled
    impl._parallel_o_proj = None
    return impl


def _make_nope_pcp_impl(width, packed_cache=False):
    impl = _make_pcp_sfa_impl()
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = width
    impl.num_kv_heads = 1
    impl.is_pcp_decode_sharded = False
    impl.enable_sparse_sfa_c8 = packed_cache
    impl.enable_sparse_sfa_turboquant = False
    impl.c8_cache_dtype = torch.int8
    impl.sfa_qsfa_tile_size = 32
    impl.kv_a_layernorm = Mock(side_effect=lambda values: values)
    impl.kv_a_layernorm.weight = torch.ones(width)
    impl.kv_a_layernorm.variance_epsilon = 1e-6
    return impl


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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.pcp_enabled = False
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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = False
    impl.pcp_enabled = True
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
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    load_state = WeightSwitchLoadState(
        input_size_per_partition_before=4,
        input_size_per_partition_after=2,
    )
    impl.o_proj = SimpleNamespace(
        input_size=8,
        input_size_per_partition=4,
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

    def prepare_load(*_args):
        impl.o_proj.input_size_per_partition = 2
        return load_state

    with (
        patch("vllm_ascend.attention.sfa_v1.enable_pcp_o_proj_weight_sharding", return_value=True),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=pcp_group),
        patch(
            "vllm_ascend.attention.sfa_v1.get_tp_group",
            return_value=SimpleNamespace(world_size=2, rank_in_group=0),
        ),
        patch.object(vllm_distributed, "get_pcp_group", return_value=pcp_group),
        patch.object(
            vllm_distributed,
            "get_tp_group",
            return_value=SimpleNamespace(world_size=2, rank_in_group=0),
        ),
        patch.object(impl.o_proj.quant_method, "prepare_layer_for_parallel_weight_load", side_effect=prepare_load),
    ):
        impl._initialize_pcp_o_proj()
    return impl


@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_sfa_pcp_decode_buffers_exist_before_forward_with_activation_dtype(activation_dtype):
    impl = _make_pcp_o_proj_impl(sharded_decode=True, activation_dtype=activation_dtype, weight_dtype=torch.int8)
    assert impl.o_proj.weight.dtype == torch.int8
    projection = impl._parallel_o_proj
    assert projection.o_proj is impl.o_proj
    assert projection.input_size == 4
    assert projection.output_size == 3
    expected_shapes = {
        "all_gather_input": (4, 4),
        "all_gather_output": (8, 4),
        "reduce_scatter_input": (8, 3),
        "reduce_scatter_output": (4, 3),
    }
    for name, shape in expected_shapes.items():
        buffer = getattr(impl._parallel_o_proj, name)
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
    assert impl._parallel_o_proj.token_capacity == expected_capacity
    assert impl._parallel_o_proj.all_gather_input.shape == (expected_capacity, 4)
    assert impl._parallel_o_proj.all_gather_output.shape == (2 * expected_capacity, 4)
    assert impl._parallel_o_proj.reduce_scatter_input.shape == (2 * expected_capacity, 3)
    assert impl._parallel_o_proj.reduce_scatter_output.shape == (expected_capacity, 3)


def test_sfa_pcp_decode_buffers_are_shared_across_compatible_layers():
    with (
        patch.object(AscendSFAImpl, "o_proj_full_pools", {}),
        patch("vllm_ascend.weight_switch.o_proj.torch.empty", wraps=torch.empty) as allocate,
    ):
        first = _make_pcp_o_proj_impl(sharded_decode=True)
        second = _make_pcp_o_proj_impl(sharded_decode=True)

        assert allocate.call_count == 4
        buffers = [
            value
            for name, value in vars(first._parallel_o_proj).items()
            if name in {"all_gather_input", "all_gather_output", "reduce_scatter_input", "reduce_scatter_output"}
            and isinstance(value, torch.Tensor)
        ]
        assert len({buffer.data_ptr() for buffer in buffers}) == 4
        for name, buffer in vars(first._parallel_o_proj).items():
            if name in {
                "all_gather_input",
                "all_gather_output",
                "reduce_scatter_input",
                "reduce_scatter_output",
            } and isinstance(buffer, torch.Tensor):
                assert getattr(second._parallel_o_proj, name) is buffer

        first._parallel_o_proj.enable()
        second._parallel_o_proj.enable()
        assert first._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor.data_ptr() == (
            second._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor.data_ptr()
        )
        assert first._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor.data_ptr() not in {
            buffer.data_ptr() for buffer in buffers
        }


@pytest.mark.parametrize("changed_config", [{"activation_dtype": torch.bfloat16}, {"max_capture_size": 8}])
def test_sfa_pcp_decode_buffer_pool_separates_dtype_and_shape(changed_config):
    with patch.object(AscendSFAImpl, "o_proj_full_pools", {}):
        first = _make_pcp_o_proj_impl(sharded_decode=True)
        second = _make_pcp_o_proj_impl(sharded_decode=True, **changed_config)

        for name, buffer in vars(first._parallel_o_proj).items():
            if name in {
                "all_gather_input",
                "all_gather_output",
                "reduce_scatter_input",
                "reduce_scatter_output",
            } and isinstance(buffer, torch.Tensor):
                assert getattr(second._parallel_o_proj, name).data_ptr() != buffer.data_ptr()


def test_sfa_pcp_decode_buffer_pool_uses_weight_device():
    allocate_cpu = torch.empty

    def allocate(shape, *, dtype, device):
        return allocate_cpu(shape, dtype=dtype, device="cpu")

    with (
        patch.object(AscendSFAImpl, "o_proj_full_pools", {}),
        patch("vllm_ascend.weight_switch.o_proj.torch.empty", side_effect=allocate) as allocate_mock,
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
        for name, buffer in vars(first._parallel_o_proj).items():
            if name in {
                "all_gather_input",
                "all_gather_output",
                "reduce_scatter_input",
                "reduce_scatter_output",
            } and isinstance(buffer, torch.Tensor):
                assert getattr(second._parallel_o_proj, name) is not buffer
                assert getattr(same_device._parallel_o_proj, name) is buffer


def test_sfa_pcp_weight_switch_does_not_install_loader_when_disabled() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
    impl.o_proj = SimpleNamespace(input_size_per_partition=4, output_size_per_partition=3)
    impl.is_pcp_decode_sharded = False
    impl.vllm_config = SimpleNamespace()
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.enable_pcp_o_proj_weight_sharding",
            return_value=False,
        ),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=pcp_group),
        patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=SimpleNamespace()),
        patch.object(ParallelOProjection, "prepare_weight_load") as prepare_load,
    ):
        impl._initialize_pcp_o_proj()

    assert not impl.enable_pcp_o_proj_weight_sharding
    assert impl._parallel_o_proj.config.group is pcp_group
    assert not impl._parallel_o_proj._load_prepared
    prepare_load.assert_not_called()


def test_sfa_dcp_extends_v1_backend() -> None:
    assert AscendSFAImpl.supports_mtp_with_cp_non_trivial_interleave_size
    assert AscendSFAImpl.can_return_lse_for_decode
    assert issubclass(
        AscendSFADCPMetadataBuilder,
        AscendSFAMetadataBuilder,
    )
    assert "dcp_context" not in {field.name for field in fields(AscendSFAMetadata)}
    assert "dcp_context" in {field.name for field in fields(AscendSFADCPMetadata)}


@pytest.mark.parametrize("pcp_size,dcp_enabled", [(1, False), (2, False), (1, True), (2, True)])
def test_sfa_backend_uses_one_implementation_and_layout_specific_builder(pcp_size, dcp_enabled):
    config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size))
    with (
        patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.sfa_v1.get_ascend_config") as ascend_config,
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=dcp_enabled),
    ):
        ascend_config.return_value.sparse_kv_offload_config.enabled = False
        assert AscendSFABackend.get_impl_cls() is AscendSFAImpl
        expected = AscendSFAMetadataBuilder
        if dcp_enabled:
            expected = AscendSFAPCPDCPMetadataBuilder if pcp_size > 1 else AscendSFADCPMetadataBuilder
        assert AscendSFABackend.get_builder_cls() is expected


def test_sfa_pcp_implementation_supports_mtp_with_non_trivial_interleave():
    assert AscendSFAImpl.supports_pcp
    assert AscendSFAImpl.supports_dcp
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
    metadata.uses_pcp_sharded_layout = True
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
    assert AscendSFAImpl._has_prefill(metadata) is global_has_prefill
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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = impl.pcp_enabled = True
    impl.qk_rope_head_dim = 64
    impl.dcp_group = object()
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl._start_dcp_gather = Mock(return_value="gathered")
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.uses_pcp_sharded_layout = True
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


def test_sfa_pcp_dcp_native_cache_write_uses_full_dcp_slots():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = impl.dcp_enabled = True
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim, impl.kv_lora_rank, impl.num_kv_heads = 1, 2, 1
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.uses_pcp_sharded_layout = True
    metadata.num_prefills, metadata.num_decode_tokens, metadata.num_input_tokens = 1, 0, 2
    main_slots = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    metadata.dcp_context = SimpleNamespace(slot_mapping=main_slots)
    kv, rope = torch.zeros(2, 3), torch.zeros(2, 1)
    cache = (torch.empty(1), torch.empty(1))
    with (
        patch(
            "vllm_ascend.attention.sfa_v1._gather_prefill_cache_inputs",
            return_value=((kv.repeat(2, 1), rope.repeat(2, 1), rope.repeat(2, 1)), main_slots),
        ) as gather,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_kv_rmsnorm_rope_cache", create=True) as write,
    ):
        assert impl.exec_kv(kv, rope, rope, cache, torch.tensor([-1, -1]), metadata) == (None, None)
    gather.assert_called_once_with((kv, rope, rope), main_slots, 0)
    assert write.call_args.args[4] is main_slots


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize(
    "sharded_decode,num_prefills,num_decode_tokens,global_prefill",
    [
        (False, 1, 1, False),
        (True, 0, 2, False),
        (False, 0, 1, True),
        (False, 0, 2, False),
    ],
)
def test_sfa_gathers_main_kv_only_when_pcp_enabled(
    pcp_enabled, sharded_decode, num_prefills, num_decode_tokens, global_prefill
):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = False
    impl.pcp_enabled = pcp_enabled
    impl.is_pcp_decode_sharded = sharded_decode
    impl.qk_rope_head_dim = 1
    impl.kv_lora_rank = 2
    impl.num_kv_heads = 1
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        pcp_has_global_prefill=global_prefill,
        uses_pcp_sharded_layout=True,
    )
    kv = torch.arange(6, dtype=torch.float32).view(2, 3)
    cos = torch.arange(2, dtype=torch.float32).view(2, 1)
    sin = cos + 10
    slots = torch.tensor([4, 5], dtype=torch.int32)
    needs_gather = pcp_enabled and (sharded_decode or num_prefills > 0 or global_prefill)
    if pcp_enabled and not needs_gather:
        slots = torch.cat((slots, torch.tensor([-1, -1], dtype=slots.dtype)))
    gathered_kv = torch.arange(12, dtype=torch.float32).view(4, 3)
    gathered_cos = torch.arange(4, dtype=torch.float32).view(4, 1)
    gathered_sin = gathered_cos + 10
    gathered_slots = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    cache = (torch.empty(1), torch.empty(1))
    with (
        patch(
            "vllm_ascend.attention.sfa_v1._gather_prefill_cache_inputs",
            return_value=((gathered_kv, gathered_cos, gathered_sin), gathered_slots),
        ) as gather,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_kv_rmsnorm_rope_cache", create=True) as write,
    ):
        assert impl.exec_kv(kv, cos, sin, cache, slots, metadata) == (None, None)
    expected_kv = gathered_kv if needs_gather else kv
    torch.testing.assert_close(write.call_args.args[0].flatten(1), expected_kv)
    torch.testing.assert_close(write.call_args.args[2], gathered_cos if needs_gather else cos)
    torch.testing.assert_close(write.call_args.args[3], gathered_sin if needs_gather else sin)
    torch.testing.assert_close(write.call_args.args[4], (gathered_slots if needs_gather else slots[:2]).long())
    if needs_gather:
        gather.assert_called_once_with((kv, cos, sin), slots, 0 if sharded_decode else num_decode_tokens)
    else:
        gather.assert_not_called()


def test_sfa_pcp_replicated_draft_decode_skips_kv_gather() -> None:
    impl = _make_nope_pcp_impl(1)
    impl.is_pcp_decode_sharded = True
    metadata = SimpleNamespace(
        uses_pcp_sharded_layout=False,
        num_prefills=0,
        pcp_has_global_prefill=False,
        num_input_tokens=2,
        num_decode_tokens=0,
    )
    local = torch.tensor([[1.0], [2.0]])
    slots = torch.tensor([5, 7], dtype=torch.int64)
    with (
        patch("vllm_ascend.attention.sfa_v1._gather_prefill_cache_inputs") as gather,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_", create=True) as write,
    ):
        assert impl.exec_kv(local, None, None, (torch.empty(8, 1),), slots, metadata) == (None, None)
    gather.assert_not_called()
    torch.testing.assert_close(write.call_args.args[2], local)
    torch.testing.assert_close(write.call_args.args[1].flatten(), slots)


def test_sfa_pcp_gathers_nope_kv_without_rope() -> None:
    impl = _make_nope_pcp_impl(3)
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        num_prefills=1,
        num_input_tokens=2,
        uses_pcp_sharded_layout=True,
        pcp_has_global_prefill=False,
    )
    local = torch.arange(6, dtype=torch.float32).view(2, 3)
    slots = torch.tensor([0, 1, 0, 5], dtype=torch.int64)
    remote = torch.tensor([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]])
    group = SimpleNamespace(world_size=2, all_gather=Mock(return_value=remote))
    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch(
            "vllm_ascend.attention.sfa_v1._gather_prefill_cache_inputs", wraps=_gather_prefill_cache_inputs
        ) as gather,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_", create=True) as write,
    ):
        assert impl.exec_kv(local, None, None, (torch.empty(8, 3),), slots, metadata) == (None, None)
    gather.assert_called_once_with((local,), slots, 1)
    group.all_gather.assert_called_once()
    torch.testing.assert_close(group.all_gather.call_args.args[0], local[1:])
    torch.testing.assert_close(write.call_args.args[2], torch.cat((local[:1], remote)))
    torch.testing.assert_close(write.call_args.args[1].flatten(), torch.tensor([0, 1, 5]))


@pytest.mark.parametrize("packed_cache", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 1])
def test_sfa_pcp_gathers_nope_kv_with_padded_rank_slots(packed_cache: bool, num_decode_tokens: int) -> None:
    impl = _make_nope_pcp_impl(1, packed_cache)
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_prefills=1,
        num_input_tokens=8,
        uses_pcp_sharded_layout=True,
        pcp_has_global_prefill=False,
    )
    local = torch.arange(5, dtype=torch.float32).view(5, 1)
    slots = torch.tensor([0, 1, 2, 3, 4, -1, -1, -1, 8, 9, 10, 11, 12, -1, -1, -1])
    slots[8 : 8 + num_decode_tokens] = slots[:num_decode_tokens]

    def all_gather(tensor, dim=0):
        padded = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0]).view(8, 1).float()
        torch.testing.assert_close(tensor, padded[num_decode_tokens:])
        remote = torch.arange(10 + num_decode_tokens, 18, dtype=torch.float32).view(-1, 1)
        return torch.cat((tensor, remote), dim=dim)

    group = SimpleNamespace(world_size=2, all_gather=all_gather)
    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch("vllm_ascend.attention.sfa_v1.custom_kv_rmsnorm_rope", return_value=(None, None, None)) as quantize,
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_", create=True) as write,
    ):
        impl.exec_kv(local, None, None, (torch.empty(32, 1),), slots, metadata)
    expected = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0, *range(10 + num_decode_tokens, 18)]).view(-1, 1).float()
    if packed_cache:
        quantize.assert_called_once()
        torch.testing.assert_close(quantize.call_args.args[0].flatten(1), expected)
        write.assert_not_called()
    else:
        quantize.assert_not_called()
        torch.testing.assert_close(write.call_args.args[2], expected)
        torch.testing.assert_close(
            write.call_args.args[1].flatten(), torch.cat((slots[:8], slots[8 + num_decode_tokens :]))
        )


def test_sfa_pcp_packed_kv_writes_gather_order_slots() -> None:
    impl = _make_pcp_sfa_impl()
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.sfa_qsfa_packed_kv_head_dim = 4
    k_pe = torch.ones(2, 1, 1)
    k_nope = torch.ones(2, 1, 2)
    knope_scale = torch.ones(2, 1, 1)
    kv_cache = (torch.empty(1, 1, 1, 4),)

    with patch("vllm_ascend.attention.sfa_v1.DeviceOperator.scatter_cache") as scatter:
        impl._store_packed_kv_cache(
            k_pe,
            k_nope,
            knope_scale,
            kv_cache,
            torch.tensor([4, 8]),
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
    metadata.uses_pcp_sharded_layout = True
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
    impl = _make_pcp_sfa_impl(dcp_enabled=True)
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
    impl._start_dcp_query_gather = Mock()
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
    metadata.uses_pcp_sharded_layout = True
    metadata.num_decode_tokens = num_decode_tokens
    metadata.num_prefills = int(num_decode_tokens < len(local))
    metadata.pcp_has_global_prefill = metadata.num_prefills > 0
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
    metadata.cum_query_lens = metadata.seq_lens = torch.tensor([len(local)])
    impl.fused_qkv_a_proj = Mock(return_value=(torch.cat((hidden, local), dim=-1),))
    impl.q_a_layernorm = lambda values: values
    impl._q_proj_and_k_up_proj = Mock(return_value=(hidden.unsqueeze(1), hidden.unsqueeze(1)))
    impl.rope_single = lambda values, *_args: values
    impl._prepare_indexer_metadata = Mock()
    impl.indexer = Mock(return_value=torch.zeros(len(local), 1, 1, dtype=torch.int32))
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = lambda values: values
    impl._finalize_o_proj = lambda values, output: output.copy_(values)

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
    impl = _make_nope_pcp_impl(1)
    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_prefills=0,
        num_input_tokens=8,
        pcp_has_global_prefill=True,
        uses_pcp_sharded_layout=True,
    )
    slots = torch.tensor([-1] * 8 + [8, 9, 10, -1, -1, -1, -1, -1])
    group = SimpleNamespace(
        world_size=2, all_gather=lambda tensor, dim=0: torch.cat((tensor, torch.arange(10, 18).view(8, 1).float()))
    )
    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_", create=True) as write,
    ):
        impl.exec_kv(torch.empty(0, 1), None, None, (torch.empty(16, 1),), slots, metadata)
    write_slots = write.call_args.args[1].flatten()
    values = write.call_args.args[2]
    assert values.shape == (16, 1)
    torch.testing.assert_close(write_slots, slots)
    torch.testing.assert_close(values[write_slots >= 0], torch.tensor([[10.0], [11.0], [12.0]]))


def test_sfa_pcp_o_proj_switch_slices_the_tp_local_weight_by_pcp_rank() -> None:
    AscendSFAImpl.o_proj_full_pools.clear()
    impl = _make_pcp_o_proj_impl()

    impl._parallel_o_proj.enable()

    assert impl._parallel_o_proj._weight_switch_enabled
    torch.testing.assert_close(
        impl.o_proj.weight,
        torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]),
    )
    assert impl._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor.shape == (3, 4)


def test_sfa_pcp_prefill_gathers_weight_and_restores_local_view() -> None:
    impl = _make_pcp_o_proj_impl()
    impl._parallel_o_proj.enable()

    local_weight_ptr = impl.o_proj.weight.data_ptr()
    full_weight = impl._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor
    full_weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))

    def fake_project(_attn_output, output):
        assert impl.o_proj.weight.data_ptr() == full_weight_tensor.data_ptr()
        output.fill_(7)
        return output

    full_weight_tensor = full_weight
    impl._parallel_o_proj.set_mode(use_full_weight=True, exchange_tokens=False)
    with patch.object(impl._parallel_o_proj, "project_full", side_effect=fake_project):
        result = impl._finalize_o_proj(torch.empty(1, 4), torch.empty(1, 3))

    assert result.tolist() == [[7.0, 7.0, 7.0]]
    assert impl.o_proj.weight.data_ptr() == local_weight_ptr


def test_sfa_pcp_decode_projects_local_weight_then_reduces_pcp_and_tp() -> None:
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl = _make_pcp_o_proj_impl()
    impl._parallel_o_proj.config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
    impl._parallel_o_proj.enable()

    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    input_ = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(input_, full_weight, impl.o_proj.bias)
    pcp_group.all_reduce = lambda _: torch.nn.functional.linear(input_, full_weight, bias=None)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    impl._parallel_o_proj.set_mode(use_full_weight=False, exchange_tokens=False)
    impl._parallel_o_proj.tp_group = tp_group
    result = impl._finalize_o_proj(input_, torch.empty_like(expected))

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
def test_sfa_pcp_sharded_decode_preserves_token_owners(rank, tp_size, padding, quantized, num_speculative_tokens):
    group = SimpleNamespace(world_size=2, rank_in_group=rank, device_group=object(), all_reduce=Mock())
    impl = _make_pcp_o_proj_impl(sharded_decode=True, num_speculative_tokens=num_speculative_tokens)
    impl._parallel_o_proj.config = WeightSwitchConfig.from_group(group, shard_axis="input")
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    impl.o_proj.weight = torch.nn.Parameter(weight[:, rank * 2 : (rank + 1) * 2].clone(), requires_grad=False)
    impl._parallel_o_proj.enable()
    quant_bias = torch.tensor([0.25, 0.5, 0.75]) if quantized else torch.zeros(3)
    if quantized:
        method = MagicMock(spec=AscendLinearScheme)

        def apply_weight_shard(layer, x, *, bias_rank=0):
            return torch.nn.functional.linear(x, layer.weight) + (quant_bias if bias_rank == 0 else 0)

        method.apply_weight_shard.side_effect = apply_weight_shard
        impl.o_proj.quant_method = method
    buffer_names = (
        "all_gather_input",
        "all_gather_output",
        "reduce_scatter_input",
        "reduce_scatter_output",
    )
    buffer_ptrs = tuple(getattr(impl._parallel_o_proj, name).data_ptr() for name in buffer_names)
    for num_rows in [1, 2, impl._parallel_o_proj.token_capacity]:
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
        impl._parallel_o_proj.set_mode(use_full_weight=False, exchange_tokens=True)
        impl._parallel_o_proj.tp_group = tp_group
        with (
            patch("vllm_ascend.weight_switch.o_proj.dist.all_gather_into_tensor", side_effect=gather),
            patch("vllm_ascend.weight_switch.o_proj.dist.reduce_scatter_tensor", side_effect=scatter),
        ):
            actual = impl._finalize_o_proj(inputs[rank], torch.empty_like(expected))
        torch.testing.assert_close(actual, expected)
        assert tuple(getattr(impl._parallel_o_proj, name).data_ptr() for name in buffer_names) == buffer_ptrs
        group.all_reduce.assert_not_called()
        assert tp_group.all_reduce.call_count == int(tp_size > 1)
        if quantized:
            assert method.apply_weight_shard.call_args.kwargs["bias_rank"] == (0 if rank == 0 else 1)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize("skip_bias_add", [False, True])
@pytest.mark.parametrize("sharded_decode", [False, True])
def test_sfa_pcp_ordinary_bias_is_added_once_when_tp_reduction_is_deferred(
    tp_size, reduce_results, skip_bias_add, sharded_decode
):
    """The decoder's deferred TP sum must receive one complete ordinary bias."""
    num_rows = 2
    inputs = torch.arange(2 * num_rows * 4 * tp_size, dtype=torch.float32).view(2 * num_rows, 4 * tp_size) / 8
    weight = torch.arange(3 * 4 * tp_size, dtype=torch.float32).view(3, 4 * tp_size) / 16
    bias = torch.tensor([0.5, 1.0, 1.5])
    owner_outputs: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for pcp_rank in range(2):
        for tp_rank in range(tp_size):
            impl = _make_pcp_o_proj_impl(sharded_decode=sharded_decode)
            layer = impl.o_proj
            layer.input_size = 4 * tp_size
            layer.reduce_results = reduce_results
            layer.skip_bias_add = skip_bias_add
            layer.tp_size = tp_size
            layer.tp_rank = tp_rank
            layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
            tp_weight = weight[:, tp_rank * 4 : (tp_rank + 1) * 4]
            layer.weight = torch.nn.Parameter(
                tp_weight[:, pcp_rank * 2 : (pcp_rank + 1) * 2].clone(), requires_grad=False
            )
            all_tp_inputs = inputs[:, tp_rank * 4 : (tp_rank + 1) * 4]
            local_inputs = (
                all_tp_inputs[pcp_rank * num_rows : (pcp_rank + 1) * num_rows]
                if sharded_decode
                else all_tp_inputs[:num_rows]
            )
            tp_projection = torch.nn.functional.linear(all_tp_inputs, tp_weight)
            full_projection = torch.nn.functional.linear(inputs, weight)
            local_slice = (
                slice(pcp_rank * num_rows, (pcp_rank + 1) * num_rows) if sharded_decode else slice(0, num_rows)
            )
            events: list[str] = []

            local_partial = torch.nn.functional.linear(local_inputs[:, pcp_rank * 2 : (pcp_rank + 1) * 2], layer.weight)
            global_partial = torch.nn.functional.linear(
                all_tp_inputs[:, pcp_rank * 2 : (pcp_rank + 1) * 2], layer.weight
            )

            def reduce_pcp(partial, expected=local_partial, reduced=tp_projection[:num_rows], events=events):
                events.append("pcp")
                torch.testing.assert_close(partial, expected)
                return reduced.clone()

            def gather(destination, source, *, group, expected=local_inputs, gathered=all_tp_inputs, events=events):
                events.append("gather")
                torch.testing.assert_close(source, expected)
                destination.copy_(gathered)

            def scatter(
                destination,
                source,
                *,
                group,
                expected=global_partial,
                reduced=tp_projection[local_slice],
                events=events,
            ):
                events.append("pcp")
                torch.testing.assert_close(source, expected)
                destination.copy_(reduced)

            def reduce_tp(
                partial, expected=tp_projection[local_slice], reduced=full_projection[local_slice], events=events
            ):
                events.append("tp")
                torch.testing.assert_close(partial, expected)
                return reduced.clone()

            pcp_group = SimpleNamespace(
                world_size=2, rank_in_group=pcp_rank, device_group=object(), all_reduce=Mock(side_effect=reduce_pcp)
            )
            tp_group = SimpleNamespace(
                world_size=tp_size, rank_in_group=tp_rank, all_reduce=Mock(side_effect=reduce_tp)
            )
            executor = impl._parallel_o_proj
            executor.config = WeightSwitchConfig.from_group(pcp_group, shard_axis="input")
            executor.tp_group = tp_group
            executor.enable()
            executor.set_mode(use_full_weight=False, exchange_tokens=sharded_decode)
            with (
                patch("vllm_ascend.weight_switch.o_proj.dist.all_gather_into_tensor", side_effect=gather),
                patch("vllm_ascend.weight_switch.o_proj.dist.reduce_scatter_tensor", side_effect=scatter),
            ):
                actual = impl._finalize_o_proj(local_inputs, torch.empty(num_rows, 3))
            assert events == (["gather", "pcp"] if sharded_decode else ["pcp"]) + (
                ["tp"] if reduce_results and tp_size > 1 else []
            )
            assert tp_group.all_reduce.call_count == int(reduce_results and tp_size > 1)
            expected_partial = full_projection[local_slice] if reduce_results else tp_projection[local_slice]
            if not skip_bias_add and (reduce_results or tp_rank == 0):
                expected_partial = expected_partial + bias
            torch.testing.assert_close(actual, expected_partial)
            owner_outputs[pcp_rank].append(actual)
        expected = torch.nn.functional.linear(inputs[local_slice], weight, None if skip_bias_add else bias)
        combined = owner_outputs[pcp_rank][0] if reduce_results else torch.stack(owner_outputs[pcp_rank]).sum(0)
        torch.testing.assert_close(combined, expected)


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
@pytest.mark.parametrize("decode_sharded", [False, True])
def test_sfa_pcp_forward_selects_mode_and_gathers_weight_for_global_prefill(
    attn_state, global_has_prefill, weight_switch_enabled, expected_gather, decode_sharded
) -> None:
    impl = _make_pcp_o_proj_impl(sharded_decode=decode_sharded)
    impl._parallel_o_proj.enable()
    impl._parallel_o_proj.enabled = weight_switch_enabled
    impl._parallel_o_proj._weight_switch_enabled = weight_switch_enabled
    impl.o_proj.quant_method.all_gather_weight = MagicMock()
    expected_mode = (
        OProjectionMode.LOCAL_WEIGHT
        if not weight_switch_enabled
        else OProjectionMode.FULL_WEIGHT
        if expected_gather
        else OProjectionMode.EXCHANGE_TOKENS
        if decode_sharded
        else OProjectionMode.LOCAL_WEIGHT
    )

    def forward(owner, layer_name, hidden, cache, metadata, output):
        assert owner is impl
        assert owner._parallel_o_proj.mode is expected_mode
        assert owner._parallel_o_proj.uses_full_weight is expected_gather
        owner.o_proj.quant_method.all_gather_weight.assert_not_called()
        owner._parallel_o_proj.all_gather_weights()
        return output

    hidden = torch.empty(1, 4)
    output = torch.empty(1, 3)
    metadata = SimpleNamespace(
        attn_state=attn_state,
        pcp_has_global_prefill=global_has_prefill,
    )
    with (
        patch.object(AscendSFAImpl, "_forward_impl", forward),
    ):
        assert impl.forward("layer", hidden, (), metadata, output) is output
    assert impl._parallel_o_proj.mode is OProjectionMode.LOCAL_WEIGHT
    if expected_gather:
        impl.o_proj.quant_method.all_gather_weight.assert_called_once_with(
            impl._parallel_o_proj._weight_switch_states[0][2], impl._parallel_o_proj.config
        )
    else:
        impl.o_proj.quant_method.all_gather_weight.assert_not_called()


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
    metadata.uses_pcp_sharded_layout = True
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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.pcp_enabled = False
    impl.qk_rope_head_dim = 64
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
            "vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process",
            return_value=output,
        ) as execute,
    ):
        result = impl._execute_sparse_flash_attention_process(None, None, (), None, metadata, None, None)
    assert result is output
    for actual, expected in zip(execute.call_args.args[3], gathered):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "pcp_enabled,local_prefill,global_prefill,expect_full",
    [
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, False),
        (True, True, False, True),
        (True, False, True, True),
    ],
)
def test_sfa_dcp_slot_mapping_matches_parallel_layout(pcp_enabled, local_prefill, global_prefill, expect_full):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.pcp_enabled = pcp_enabled
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.uses_pcp_sharded_layout = True
    metadata.num_input_tokens = 2
    metadata.num_prefills = int(local_prefill)
    metadata.pcp_has_global_prefill = global_prefill
    full_slots = torch.tensor([3200, -1, 3201, -1], dtype=torch.int32)
    metadata.dcp_context = SimpleNamespace(slot_mapping=full_slots)
    metadata.slot_mapping = torch.tensor([3200, 3201, -1], dtype=torch.int32)

    result = impl._get_sfa_kv_slot_mapping(metadata)

    if expect_full:
        assert result is metadata.slot_mapping
    else:
        assert result.data_ptr() == full_slots.data_ptr()
        torch.testing.assert_close(result, full_slots[:2])


@pytest.mark.parametrize("is_kv_consumer,sfa_c8", [(False, False), (True, True)])
@pytest.mark.parametrize("quantized_indexer", [False, True])
def test_sfa_pcp_keeps_prolog_v3_enabled(is_kv_consumer, sfa_c8, quantized_indexer):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
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
@pytest.mark.parametrize("preprocess_type", [PreprocessType.PROLOG_V3, PreprocessType.MLAPO])
def test_sfa_pcp_fused_preprocess_gathers_only_prefill_kv(
    cache_dtype, pcp_size, rank, num_decode_tokens, num_tokens, preprocess_type
):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
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
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=group),
        patch.object(
            AscendSFAImpl,
            "_sfa_preprocess_prolog_v3" if preprocess_type == PreprocessType.PROLOG_V3 else "_sfa_preprocess_mlapo",
            return_value=output,
        ) as preprocess,
        patch("vllm_ascend.attention.sfa_v1.copy_pcp_kv_cache", return_value=packed) as read,
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.reshape_and_cache") as write,
    ):
        _run_fused_forward(impl, hidden, cache, slots.flatten(), metadata, preprocess_type)
    assert preprocess.call_args.kwargs["hidden_states"].data_ptr() == hidden.data_ptr()
    assert impl._execute_sparse_flash_attention_process.call_args.args[0] is output[1]
    assert impl._execute_sparse_flash_attention_process.call_args.args[1] is output[2]
    # Upstream masks replicated decode copies outside rank 0, but each
    # rank must still write its locally computed decode KV to its own cache.
    expected_slots = slots.flatten() if not rows and preprocess_type == PreprocessType.MLAPO else expected_local_slots
    torch.testing.assert_close(preprocess.call_args.kwargs["slot_mapping"], expected_slots)
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
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
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
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=group),
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output) as preprocess,
        patch("vllm_ascend.attention.sfa_v1.copy_pcp_kv_cache", return_value=packed) as copy_cache,
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.reshape_and_cache") as write_cache,
    ):
        _run_fused_forward(impl, hidden, cache, slots, metadata)
    torch.testing.assert_close(preprocess.call_args.kwargs["slot_mapping"], torch.tensor([0, -1]))
    torch.testing.assert_close(copy_cache.call_args.args[1], torch.tensor([-1]))
    group.all_gather.assert_called_once_with(packed, dim=0)
    torch.testing.assert_close(write_cache.call_args.kwargs["slot_mapping"], torch.tensor([1, -1]))


def test_sfa_pcp_empty_global_slots_skips_kv_gather():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
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
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group") as group,
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", return_value=output) as prolog,
        patch("vllm_ascend.attention.sfa_v1.copy_pcp_kv_cache") as copy_cache,
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.reshape_and_cache") as write_cache,
    ):
        _run_fused_forward(impl, hidden, (hidden, hidden), slots, metadata)
    assert prolog.call_args.kwargs["slot_mapping"] is slots
    group.assert_not_called()
    copy_cache.assert_not_called()
    write_cache.assert_not_called()


@pytest.mark.parametrize("preprocess_type", [PreprocessType.PROLOG_V3, PreprocessType.MLAPO])
def test_sfa_pcp_padded_decode_skips_kv_gather(preprocess_type):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = False
    hidden = torch.zeros((4, 3))
    slots = torch.tensor([0, 1, -1, -1, -1, -1], dtype=torch.int32)
    cache = (torch.empty(0), torch.empty(0))
    metadata = SimpleNamespace(
        num_decode_tokens=2, num_prefills=0, pcp_has_global_prefill=False, uses_pcp_sharded_layout=True
    )
    output = (hidden, hidden, hidden, None)
    with (
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group") as group,
        patch.object(impl, "_sync_pcp_kv_cache") as sync,
        patch.object(impl, "_sfa_preprocess_prolog_v3", return_value=output) as prolog,
        patch.object(impl, "_sfa_preprocess_mlapo", return_value=output) as mlapo,
    ):
        _run_fused_forward(impl, hidden, cache, slots, metadata, preprocess_type)
    group.assert_not_called()
    sync.assert_not_called()
    preprocess = prolog if preprocess_type == PreprocessType.PROLOG_V3 else mlapo
    expected = slots[: hidden.shape[0]].long() if preprocess_type == PreprocessType.PROLOG_V3 else slots
    torch.testing.assert_close(preprocess.call_args.kwargs["slot_mapping"], expected)
    if preprocess_type == PreprocessType.MLAPO:
        assert preprocess.call_args.kwargs["num_input_tokens"] == hidden.shape[0]


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


@pytest.mark.parametrize("dcp_enabled", [False, True])
@pytest.mark.parametrize("mode", ["prefill", "mixed", "sharded_decode", "replicated_decode"])
def test_sfa_pcp_nope_gathers_latent_kv_without_rope(dcp_enabled, mode):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = dcp_enabled
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = mode == "sharded_decode"
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = 2
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.kv_a_layernorm = lambda value: value * 2
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    peer_inputs = inputs + 10
    has_prefill = mode in {"prefill", "mixed"}
    num_decodes = 0 if mode == "prefill" else (2 if mode == "replicated_decode" else 1)
    slots = torch.tensor([0, 1, -1, 2] if mode == "mixed" else [0, 1, 2, -1])
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.uses_pcp_sharded_layout = True
    metadata.num_input_tokens = inputs.shape[0]
    metadata.pcp_slot_mapping = slots
    metadata.num_decode_tokens = num_decodes
    metadata.num_prefills = int(has_prefill)
    metadata.pcp_has_global_prefill = has_prefill
    metadata.dcp_context = SimpleNamespace(slot_mapping=slots)
    group = SimpleNamespace(all_gather=Mock(side_effect=lambda values, dim: torch.cat((values, values + 10), dim)))
    group.world_size = 2
    cache = torch.full((1, 5, 1, 2), -99.0)

    def scatter(target, indices, values):
        valid = indices[:, 0] >= 0
        target[indices[valid, 0]] = values[valid]
        return target

    with (
        patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group),
        patch("torch_npu.npu_scatter_nd_update_", side_effect=scatter),
    ):
        assert impl.exec_kv(inputs, None, None, (cache,), slots, metadata) == (None, None)

    expected = torch.full((5, 2), -99.0)
    expected[:2] = inputs * 2
    if mode == "mixed":
        expected[2] = peer_inputs[1] * 2
    elif mode != "replicated_decode":
        expected[2] = peer_inputs[0] * 2
    torch.testing.assert_close(cache.view(5, 2), expected)
    if mode == "replicated_decode":
        group.all_gather.assert_not_called()
    else:
        group.all_gather.assert_called_once()
        gathered_input = inputs[1:] if mode == "mixed" else inputs
        torch.testing.assert_close(group.all_gather.call_args.args[0], gathered_input)


def _run_fused_forward(impl, hidden, cache, slots, metadata, preprocess_type=PreprocessType.PROLOG_V3):
    """Exercise the shared forward while mocking device computation boundaries."""
    impl.dcp_enabled = False
    impl.enable_sparse_sfa_c8 = getattr(impl, "enable_sparse_sfa_c8", False)
    impl.enable_sparse_sfa_turboquant = False
    impl._parallel_o_proj = None
    impl.g_proj = None
    impl.qk_rope_head_dim = 1
    impl.has_indexer = False
    impl.skip_topk = True
    impl.topk_indices_buffer = torch.zeros(hidden.shape[0], 1, dtype=torch.int64)
    impl.preprocess_type = preprocess_type
    impl.layer_name = "model.layers.0"
    impl.layerwise_kv_cache_hook = None
    impl.local_num_heads = 1
    impl.v_head_dim = hidden.shape[-1]
    impl.o_proj = lambda x: (x,)
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = lambda x: x
    metadata.cos = metadata.sin = hidden
    metadata.slot_mapping = slots[: hidden.shape[0]]
    metadata.pcp_slot_mapping = slots
    metadata.num_input_tokens = hidden.shape[0]
    metadata.num_actual_tokens = hidden.shape[0]
    metadata.attn_state = AscendAttentionState.ChunkedPrefill
    metadata.cum_query_lens = metadata.seq_lens = torch.tensor([hidden.shape[0]])
    with (
        patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.sfa_v1.attention_transfer_window"),
        patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"),
    ):
        return impl.forward(impl.layer_name, hidden, cache, metadata, output=torch.empty_like(hidden))


def _make_initialized_sfa_impl(pcp_size, dcp_size, physical_pcp_size=None):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=128,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32, hf_config=SimpleNamespace(index_topk=4), hf_text_config=SimpleNamespace(index_topk=4)
        ),
        kv_transfer_config=None,
        num_speculative_tokens=0,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            static_forward_context={}, max_cudagraph_capture_size=4, cudagraph_capture_sizes=[1, 4]
        ),
    )
    pcp_group = SimpleNamespace(world_size=physical_pcp_size or pcp_size, rank_in_group=0)
    tp_group = SimpleNamespace(world_size=1, rank_in_group=0)
    dcp_group = SimpleNamespace(world_size=dcp_size, rank_in_group=0, device_group=object())
    o_proj = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(3, 4)),
        input_size_per_partition=4,
        output_size_per_partition=3,
        quant_method=_PCPOProjLinearMethod(),
    )
    with (
        patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.sfa_v1.get_tensor_model_parallel_world_size", return_value=1),
        patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=SimpleNamespace(
                enable_sparse_sfa_c8=False, enable_sparse_sfa_turboquant=False, enable_mlapo=False
            ),
        ),
        patch("vllm_ascend.attention.sfa_v1.is_rl_weight_update_enabled", return_value=False),
        patch("vllm_ascend.attention.sfa_v1.is_mtp_layer", return_value=False),
        patch("vllm_ascend.attention.sfa_v1.enable_sp", return_value=False),
        patch("vllm_ascend.attention.sfa_v1.enable_pcp_o_proj_weight_sharding", return_value=True),
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=pcp_group) as get_pcp,
        patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=tp_group),
        patch("vllm_ascend.attention.context_parallel.common_cp.get_pcp_group", return_value=pcp_group),
        patch("vllm_ascend.attention.context_parallel.common_cp.get_tp_group", return_value=tp_group),
        patch("vllm_ascend.attention.sfa_v1.get_dcp_group", return_value=dcp_group),
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=dcp_size > 1),
        patch.object(
            _PCPOProjLinearMethod, "prepare_layer_for_parallel_weight_load", return_value=WeightSwitchLoadState(4, 2)
        ) as prepare,
    ):
        impl = AscendSFAImpl(
            num_heads=1,
            head_size=2,
            scale=0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=2,
            kv_lora_rank=2,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            qk_head_dim=2,
            v_head_dim=2,
            q_b_proj=SimpleNamespace(weight=torch.ones(1)),
            kv_b_proj=SimpleNamespace(weight=torch.ones(3, 2), quant_method=None),
            o_proj=o_proj,
            indexer=SimpleNamespace(head_dim=128),
        )
    return impl, prepare.call_count, get_pcp.call_count


@pytest.mark.parametrize(
    "pcp_size,dcp_size",
    [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_sfa_initializes_pcp_once_across_parallel_layouts(pcp_size, dcp_size):
    impl, loader_calls, group_calls = _make_initialized_sfa_impl(pcp_size, dcp_size)
    assert impl.pcp_enabled is (pcp_size > 1)
    assert impl.is_pcp_decode_sharded is (pcp_size > 1 and dcp_size == 1)
    assert loader_calls == int(pcp_size > 1)
    assert bool(group_calls) is (pcp_size > 1)
    assert impl._parallel_o_proj is None or not impl._parallel_o_proj._weight_switch_enabled
    if pcp_size == 1:
        assert impl._parallel_o_proj is None
    assert impl.dcp_enabled is (dcp_size > 1)
    assert impl.dcp_size == dcp_size
    if dcp_size == 1:
        assert impl.dcp_group is None
        assert not hasattr(impl, "_remap_order")


def test_sfa_replicated_draft_does_not_initialize_pcp_from_physical_group():
    impl, loader_calls, group_calls = _make_initialized_sfa_impl(1, 1, physical_pcp_size=2)
    assert not impl.pcp_enabled and not impl.is_pcp_decode_sharded
    assert loader_calls == group_calls == 0
    assert impl._parallel_o_proj is None


@pytest.mark.parametrize(
    "pcp_size,dcp_size",
    [
        (1, 1),
        (2, 1),
        (2, 2),
    ],
)
def test_sfa_post_load_orders_weight_switch_after_shared_processing(pcp_size, dcp_size):
    impl, _, _ = _make_initialized_sfa_impl(pcp_size, dcp_size)
    events = []
    impl.indexer.process_weights_after_loading = lambda: events.append("indexer")
    if impl._parallel_o_proj is not None:
        impl._parallel_o_proj.enable = lambda: events.append("switch")
    if dcp_size > 1:
        impl._register_remap_buffers = lambda: events.append("remap")
    with (
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_format_cast", side_effect=lambda x, _: x),
        patch("vllm_ascend.attention.sfa_v1.dispose_layer"),
        patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz", side_effect=lambda x: x),
        patch.object(
            impl, "_resolve_preprocess_type", side_effect=lambda _: events.append("preprocess") or PreprocessType.NATIVE
        ),
    ):
        impl.process_weights_after_loading(torch.float32)
    expected = ["preprocess", "indexer"]
    if pcp_size > 1:
        expected.append("switch")
    if dcp_size > 1:
        expected.append("remap")
    assert events == expected
    assert impl.W_UK_T.shape == (1, 1, 2)
    assert impl.W_UV.shape == (1, 2, 2)


def test_sfa_disabled_dcp_initialization_leaves_fused_preprocessing_available():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.vllm_config = object()
    impl.enable_mlapo = True
    with (
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=False),
        patch("vllm_ascend.attention.sfa_v1.get_dcp_group") as group,
    ):
        impl._initialize_dcp()
    group.assert_not_called()
    assert not impl.dcp_enabled and impl.dcp_group is None
    assert impl.dcp_size == 1 and impl.dcp_rank == 0
    assert impl.enable_mlapo
    assert not hasattr(impl, "_remap_order")


@pytest.mark.parametrize(
    "pcp_enabled,prefill,global_prefill",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
@pytest.mark.parametrize("dcp_enabled", [False, True])
@pytest.mark.parametrize("packed_cache", [False, True])
def test_sfa_forward_orders_cache_writes_and_collectives(
    pcp_enabled, prefill, global_prefill, dcp_enabled, packed_cache
):
    """Exercise the public forward, with mocked device boundaries."""
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled, impl.pcp_enabled = dcp_enabled, pcp_enabled
    impl.is_pcp_decode_sharded = False
    impl._parallel_o_proj = None
    impl.qk_rope_head_dim = impl.q_lora_rank = impl.kv_lora_rank = 1
    impl.local_num_heads = impl.v_head_dim = 1
    impl.enable_sparse_sfa_c8 = packed_cache
    impl.enable_sparse_sfa_turboquant = False
    impl.sfa_qsfa_packed_kv_head_dim = 3
    impl.preprocess_type = PreprocessType.NATIVE
    impl.g_proj = impl.layerwise_kv_cache_hook = None
    impl.has_indexer = True
    impl.skip_topk = impl.use_index_cache = False
    impl.layer_name = "layer"
    impl._dcp_interleave_size, impl._dcp_index_topk = 1, 1
    impl._remap_order, impl._remap_invalid_index = torch.arange(1).float(), torch.tensor(-1.0)
    dcp = SimpleNamespace(world_size=4 if pcp_enabled else 2, unique_name="dcp")
    pcp = SimpleNamespace(world_size=2 if pcp_enabled else 1, unique_name="pcp")
    tp = SimpleNamespace(world_size=2 if pcp_enabled else 1, unique_name="tp")
    impl.dcp_group, impl.dcp_size, impl.dcp_rank = dcp, dcp.world_size, 0
    events = []
    if pcp_enabled:
        impl._parallel_o_proj = MagicMock()
        impl._parallel_o_proj.all_gather_weights.side_effect = lambda: events.append("o_proj_gather")
        impl._parallel_o_proj.forward.side_effect = lambda value, output: output.copy_(value)
    cache = (torch.ones(2, 2, 1, 3 if packed_cache else 1), torch.ones(2, 2, 1, 1))
    hidden, slots, lengths = torch.ones(2, 2), torch.arange(2), torch.tensor([2])
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.uses_pcp_sharded_layout = True
    vars(metadata).update(
        num_actual_tokens=2,
        num_input_tokens=2,
        num_prefills=int(prefill),
        num_decode_tokens=0 if prefill else 2,
        pcp_has_global_prefill=global_prefill,
        cos=hidden[:, :1],
        sin=hidden[:, :1],
        slot_mapping=slots,
        pcp_slot_mapping=slots,
        cum_query_lens=lengths,
        seq_lens=lengths,
        attn_state=AscendAttentionState.ChunkedPrefill,
        dcp_context=SimpleNamespace(
            slot_mapping=slots,
            block_table=torch.tensor([[0, 1]]),
            seq_lens=lengths,
            kv_gather_block_ids=torch.arange(2),
            kv_gather_block_table=torch.tensor([[0, 1]]),
            gather_context=None,
        ),
    )
    impl._compose_sfa_kv_cache = Mock(return_value=cache)
    impl._get_indexer_attn_metadata = Mock(return_value=object())
    impl.fused_qkv_a_proj = lambda value: (torch.ones(value.shape[0], 3),)
    impl.q_a_layernorm = lambda value: value

    def exec_kv(*args):
        torch.testing.assert_close(args[4], slots)
        events.append("kv_preprocess" if packed_cache else "cache_write")
        if packed_cache:
            return torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1)
        return None, None

    impl.exec_kv = Mock(side_effect=exec_kv)
    impl._q_proj_and_k_up_proj = lambda value: (torch.ones(2, 1, 1), torch.ones(2, 1, 1))
    impl.rope_single = lambda value, cos, sin: value
    impl._prepare_indexer_metadata = Mock()
    impl.indexer = Mock(
        side_effect=lambda *args, **kw: events.append("indexer") or torch.zeros(2, 1, dtype=torch.int32)
    )
    impl._v_up_proj = lambda value: value.flatten(1)
    impl.o_proj = lambda value: (value,)

    def gather(value, group):
        events.append("kv_gather" if value.ndim == 4 else "query_gather")
        return torch.cat([value] * group.world_size), SimpleNamespace(wait=lambda: events.append("wait"))

    def attention(owner, query, *args, **kwargs):
        events.append("attention")
        has_prefill = prefill or global_prefill
        if dcp_enabled:
            assert kwargs["sparse_mode"] == (3 if has_prefill else 0)
            assert kwargs["return_lse"] is not has_prefill
        assert args[4] is lengths
        output = torch.full_like(query, 3)
        if has_prefill or not dcp_enabled:
            return output
        return output, torch.zeros(1, *query.shape[:2]), torch.ones(1, *query.shape[:2])

    def merge(output, lse, *args):
        events.append("merge")
        assert torch.isfinite(lse).all()
        return output[:, :1]

    with (
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=pcp),
        patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=tp),
        patch(
            "vllm_ascend.attention.sfa_v1.prepare_query_head_major",
            side_effect=lambda nope, rope: torch.cat((nope, rope), dim=-1).transpose(0, 1).contiguous(),
        ),
        patch("vllm_ascend.attention.sfa_v1.all_gather_async", side_effect=gather) as collective,
        patch(
            "vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process", side_effect=attention
        ),
        patch(
            "vllm_ascend.attention.sfa_v1.DeviceOperator.scatter_cache",
            side_effect=lambda *args: events.append("cache_write"),
        ),
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_force_tmajor_restore", return_value=False),
        patch("torch.ops.vllm.dcp_a2a_fused", side_effect=merge),
        patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.sfa_v1.attention_transfer_window"),
        patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"),
    ):
        for _ in range(2):
            events.clear()
            collective.reset_mock()
            output = impl.forward("layer", hidden, cache, metadata, torch.empty(2, 1))
            expected = ["o_proj_gather"] if pcp_enabled else []
            if packed_cache:
                expected.append("kv_preprocess")
                if dcp_enabled and not (prefill or global_prefill):
                    expected.append("query_gather")
            expected.append("cache_write")
            if dcp_enabled and (not packed_cache or prefill or global_prefill):
                expected.append("kv_gather" if prefill or global_prefill else "query_gather")
            expected.append("indexer")
            if dcp_enabled:
                expected.append("wait")
            expected.append("attention")
            if dcp_enabled and not (prefill or global_prefill):
                expected.append("merge")
            assert events == expected
            assert collective.call_count == int(dcp_enabled)
            assert metadata.dcp_context.gather_context is None
            torch.testing.assert_close(output, torch.full((2, 1), 3.0))
