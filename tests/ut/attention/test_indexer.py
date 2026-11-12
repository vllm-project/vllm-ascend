# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch_npu
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.device_op import A5DeviceAdaptor

_KERNEL_BLOCK_SIZE = 128


def _make_builder(
    pcp_size: int = 1,
    dcp_size: int = 1,
    num_speculative_tokens: int | None = None,
    physical_block_size: int = 128,
) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=physical_block_size,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_topk=2048),
        hf_config=SimpleNamespace(),
        # DCP fixtures below provide two 128-token columns per rank.
        max_model_len=512,
    )
    vllm_config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=max(4, dcp_size),
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
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    metadata = SimpleNamespace(
        pcp_context=None,
        pcp_cache_group_idx=None,
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
    )

    def replace(**changes):
        values = vars(metadata).copy()
        values.pop("replace", None)
        values.update(changes)
        return SimpleNamespace(**values)

    metadata.replace = replace
    return metadata


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
    assert not metadata.uses_pcp_sharded_layout
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert torch.equal(metadata.actual_seq_lengths_query, common.query_start_loc[1:3])
    assert torch.equal(metadata.query_start_loc, common.query_start_loc[:3])
    assert metadata.query_start_loc.data_ptr() == common.query_start_loc.data_ptr()
    assert torch.equal(metadata.actual_seq_lengths_key, common.seq_lens[:2])
    assert metadata.num_decode_tokens == 0

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])
    # Ordinary target execution reuses the framework's graph-stable RoPE
    # tables instead of introducing another asynchronously copied buffer.
    assert metadata.cos.data_ptr() == cos.data_ptr()
    assert metadata.sin.data_ptr() == sin.data_ptr()


def _make_decode_pcp_context() -> SimpleNamespace:
    # Only the global-prefill flag is read when no request is prefilling.
    return SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([False])))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("rows", [8, 9, 10])
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sharded_indexer_pcp_request_capacity(mock_cos_sin, native, rows):
    # Four PCP prefills become eight segments, plus one possible graph dummy.
    builder = _make_builder(pcp_size=2)
    builder.use_sharded_dcp = True
    builder.use_native_dcp_qli = native
    builder.vllm_config.parallel_config.cp_kv_cache_interleave_size = 2
    mock_cos_sin.return_value = (torch.zeros(rows * 2, 1, 1, 8), torch.zeros(rows * 2, 1, 1, 8))
    common = _make_common_metadata()
    common.num_reqs = rows
    common.num_actual_tokens = common.num_input_tokens = rows * 2
    common.slot_mapping = torch.arange(rows * 2)
    common.positions = torch.arange(rows * 2)
    common.query_start_loc = torch.arange(0, rows * 2 + 1, 2, dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.seq_lens = torch.arange(2, rows * 2 + 1, 2, dtype=torch.int32)
    common.block_table_tensor = torch.zeros(rows, 2, dtype=torch.int32)
    common.is_prefilling = torch.ones(rows, dtype=torch.bool)
    expected_lengths = torch.tensor(
        [sum(p // 2 % 2 == 1 for p in range(int(n))) for n in common.seq_lens], dtype=torch.int32
    )

    def lengths(seq_lens, out, *args):
        assert out.numel() == seq_lens.numel(), "request output is smaller than the kernel write range"
        out.copy_(expected_lengths)

    ops: Any = ModuleType("vllm_ascend.ops.triton.sharded_indexer")
    ops.build_local_key_lengths = Mock(side_effect=lengths)
    ops.build_local_lengths = Mock(side_effect=lambda seq, query, out, visible, *args: lengths(seq, out))
    with (
        patch.dict("sys.modules", {ops.__name__: ops}),
        patch(
            "vllm_ascend.attention.indexer.get_dcp_group", return_value=SimpleNamespace(world_size=2, rank_in_group=1)
        ),
        patch(
            "vllm_ascend.attention.indexer.get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda capability: False),
        ),
    ):
        if rows > 9:
            with pytest.raises(ValueError, match="request.*capacity"):
                builder.build(0, common)
            ops.build_local_key_lengths.assert_not_called()
            ops.build_local_lengths.assert_not_called()
        else:
            metadata = builder.build(0, common)
            torch.testing.assert_close(metadata.actual_seq_lengths_key, expected_lengths)


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    common.pcp_context = _make_decode_pcp_context()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.uses_pcp_sharded_layout
    assert metadata.slot_mapping is common.slot_mapping


@pytest.mark.parametrize("capture", [False, True])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_native_dcp_decode_boundary(mock_cos_sin, mock_get_ascend_config, capture):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(2, 1, 1, 8), torch.zeros(2, 1, 1, 8))
    common = _make_common_metadata()
    common.num_actual_tokens = common.num_input_tokens = 2
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 1
    common.is_prefilling = torch.tensor([False, False])
    builder = _make_builder(pcp_size=1, dcp_size=8)
    metadata = builder.build_for_cudagraph_capture(common) if capture else builder.build(0, common)
    assert metadata.num_decode_tokens == metadata.num_actual_tokens == 2


@pytest.mark.parametrize("pcp_size,dcp_size,expected_decode_tokens", [(2, 1, 1), (2, 2, 1), (1, 8, 1)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_pcp_decode_boundary(
    mock_cos_sin, mock_get_ascend_config, pcp_size, dcp_size, expected_decode_tokens
):
    mock_cos_sin.return_value = (torch.zeros(4, 1, 1, 8), torch.zeros(4, 1, 1, 8))
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 3
    common.is_prefilling = torch.tensor([False, True])

    common.pcp_context = _make_decode_pcp_context()

    metadata = _make_builder(pcp_size=pcp_size, dcp_size=dcp_size).build(0, common)

    # Preserve the boundary for PCP gathering and native DCP fused stores.
    assert metadata.num_decode_tokens == expected_decode_tokens


@pytest.mark.parametrize("pcp_size,dcp_size", [(2, 2), (1, 8)])
@pytest.mark.parametrize("pd_decode_recompute,expected_decode_tokens", [(False, 0), (True, 2)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_rescheduled_prefill_boundary(
    mock_cos_sin,
    mock_get_ascend_config,
    pcp_size,
    dcp_size,
    pd_decode_recompute,
    expected_decode_tokens,
):
    mock_cos_sin.return_value = (torch.zeros(2, 1, 1, 8), torch.zeros(2, 1, 1, 8))
    common = _make_common_metadata()
    common.num_actual_tokens = 2
    common.num_input_tokens = 2
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 1
    common.is_prefilling = torch.tensor([True, True])

    with patch(
        "vllm_ascend.attention.indexer.is_pd_decode_recompute_scheduler_enabled",
        return_value=pd_decode_recompute,
    ):
        common.pcp_context = _make_decode_pcp_context()
        metadata = _make_builder(pcp_size=pcp_size, dcp_size=dcp_size).build(0, common)

    # Short rescheduled prefills remain prefills unless the PD decode
    # recompute scheduler intentionally treats them as decode requests.
    assert metadata.num_decode_tokens == expected_decode_tokens


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_owns_replicated_dcp_addresses(
    mock_cos_sin,
    mock_get_ascend_config,
):
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
    torch.testing.assert_close(
        metadata.cache_scatter_indices,
        torch.tensor([[0, 0], [0, 1], [4, 0], [4, 1]], dtype=torch.int32),
    )
    # The builder derives the replicated view without changing the local view
    # consumed by the SFA cache backend.
    torch.testing.assert_close(common.slot_mapping, torch.tensor([1, 2, 3, 4, 5]))
    torch.testing.assert_close(common.block_table_tensor, torch.arange(6).view(3, 2))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_draft_metadata_owns_per_step_dcp_buffers(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_cos_sin.return_value = (
        torch.zeros(5, 1, 1, 8),
        torch.zeros(5, 1, 1, 8),
    )
    builder = _make_builder(dcp_size=2, num_speculative_tokens=3)
    common = _make_common_metadata()

    step_one = builder.build_for_drafting(common, 1)
    step_one_slots = step_one.slot_mapping.clone()
    step_one_blocks = step_one.block_table.clone()
    step_one_indices = step_one.cache_scatter_indices.clone()
    # The proposer owns one persistent slot tensor per logical draft step;
    # its address is also the key for every derived metadata buffer.
    common.slot_mapping = common.slot_mapping.clone()
    common.positions = torch.tensor([2, 3, 2, 3, 9])
    step_two = builder.build_for_drafting(common, 2)

    assert step_one.slot_mapping.data_ptr() != step_two.slot_mapping.data_ptr()
    assert step_one.block_table.data_ptr() != step_two.block_table.data_ptr()
    assert step_one.cache_scatter_indices.data_ptr() != step_two.cache_scatter_indices.data_ptr()
    torch.testing.assert_close(step_one.slot_mapping, step_one_slots)
    torch.testing.assert_close(step_one.block_table, step_one_blocks)
    torch.testing.assert_close(step_one.cache_scatter_indices, step_one_indices)


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_graph_capture_owns_stable_per_step_buffers(mock_cos_sin, mock_get_ascend_config):
    mock_cos_sin.side_effect = [
        (torch.full((4, 1, 1, 8), value), torch.full((4, 1, 1, 8), -value)) for value in range(4)
    ]
    builder = _make_builder(num_speculative_tokens=3)
    common = _make_common_metadata()

    first = builder.build_for_graph_capture(common)
    first_runtime = builder.build(common_prefix_len=0, common_attn_metadata=common)
    common.slot_mapping = common.slot_mapping.clone()
    second = builder.build_for_graph_capture(common)
    second_runtime = builder.build_for_drafting(common, draft_index=1)

    for field in ("slot_mapping", "cos", "sin", "cache_scatter_indices"):
        assert getattr(first, field).data_ptr() != getattr(second, field).data_ptr()
        assert getattr(first, field).data_ptr() == getattr(first_runtime, field).data_ptr()
        assert getattr(second, field).data_ptr() == getattr(second_runtime, field).data_ptr()
    torch.testing.assert_close(first.cos, torch.ones_like(first.cos))
    torch.testing.assert_close(first.sin, -torch.ones_like(first.sin))
    torch.testing.assert_close(second.cos, torch.full_like(second.cos, 3))
    torch.testing.assert_close(second.sin, torch.full_like(second.sin, -3))


@pytest.mark.parametrize("for_cudagraph_capture", [False, True])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_pcp_dcp_slots(
    mock_cos_sin,
    mock_get_ascend_config,
    for_cudagraph_capture,
):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    common = _make_common_metadata()
    global_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_np=torch.tensor([0, 3], dtype=torch.int32).numpy(),
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

    builder = _make_builder(pcp_size=2, dcp_size=2)
    common.pcp_context = pcp_context
    common.pcp_cache_group_idx = 0
    if for_cudagraph_capture:
        metadata = builder.build_for_cudagraph_capture(common)
    else:
        metadata = builder.build(0, common)

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([2816, 2560, 2688, -1], dtype=torch.int32),
    )


def test_c4_cache_write_preserves_packed_bytes(monkeypatch):
    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    indexer.enable_sparse_li_c4 = True
    indexer.enable_sparse_li_c8 = False
    key_cache = torch.full((2, 128, 1, 64), 23, dtype=torch.uint8)
    scale_bytes = torch.full((2, 128, 1, 2, 2), 23, dtype=torch.uint8)
    indexer.k_cache = SimpleNamespace(kv_cache=(key_cache, scale_bytes.view(torch.float8_e8m0fnu)))
    packed_key = torch.arange(3 * 64, dtype=torch.uint8).view(3, 64)
    scale = torch.arange(12, dtype=torch.uint8).view(3, 2, 2)
    slots = torch.tensor([1, 129, 255])
    # A CPU tensor cannot use the NPU-only FP4 dtype. Its one-byte storage
    # surrogate exercises the packed-dtype branch without an NPU allocation.
    monkeypatch.setattr(torch_npu, "float4_e2m1fn_x2", torch.uint8, raising=False)

    def scatter(cache, indices, updates):
        assert cache.dtype == updates.dtype == torch.uint8
        cache[indices.flatten()] = updates

    scatter_op = Mock(side_effect=scatter)
    monkeypatch.setattr("vllm_ascend.attention.indexer.DeviceOperator.scatter_cache", scatter_op)

    indexer.write_cache(packed_key, scale.view(torch.float8_e8m0fnu), slots)

    expected_key = torch.full_like(key_cache, 23).view(-1, 64)
    expected_scale = torch.full_like(scale_bytes, 23).view(-1, 2, 2)
    expected_key[slots] = packed_key
    expected_scale[slots] = scale
    assert torch.equal(key_cache.view(-1, 64), expected_key)
    assert torch.equal(scale_bytes.view(-1, 2, 2), expected_scale)
    assert scatter_op.call_count == 2


@pytest.mark.parametrize("query_lengths,key_lengths", [([2, 3], [2, 3]), ([1, 1], [31, 65])])
def test_c4_a5_selector_operator_contract(monkeypatch, query_lengths, key_lengths):
    tokens, heads, head_dim = sum(query_lengths), 64, 128
    query = torch.zeros(tokens * heads, head_dim // 2, dtype=torch.uint8)
    query_scale = torch.zeros(tokens, heads, 2, 2, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    key = torch.zeros(4, 128, 1, head_dim // 2, dtype=torch.uint8)
    key_scale = torch.zeros(4, 128, 1, 2, 2, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weights = torch.arange(tokens * heads, dtype=torch.float32).view(tokens, heads).to(torch.bfloat16)
    query_ends = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    key_lengths = torch.tensor(key_lengths, dtype=torch.int32)
    metadata = SimpleNamespace(block_table=torch.tensor([[1], [3]], dtype=torch.int32))
    kernel_metadata = object()
    expected = torch.zeros(tokens, 1, 2048, dtype=torch.int32)
    metadata_op = Mock(return_value=kernel_metadata)
    select_op = Mock(return_value=(expected, torch.empty(0)))
    monkeypatch.setattr(A5DeviceAdaptor, "_load_cann_quant_lightning_indexer_ops", lambda: (metadata_op, select_op))

    result = A5DeviceAdaptor.indexer_select_post_process(
        query,
        query_scale,
        (tokens, heads, head_dim),
        weights,
        (key, key_scale),
        0,
        1,
        metadata,
        query_ends,
        key_lengths,
        False,
        True,
        False,
    )

    assert result is expected
    q_arg, k_arg, w_arg, qs_arg, ks_arg = select_op.call_args.args
    assert q_arg.shape == (tokens, heads, head_dim // 2)
    assert q_arg.data_ptr() == query.data_ptr()
    assert k_arg is key and qs_arg is query_scale and ks_arg is key_scale
    assert w_arg.dtype == torch.float32
    torch.testing.assert_close(w_arg, weights.float())
    for call in (metadata_op.call_args, select_op.call_args):
        args = call.kwargs
        assert args["quant_mode"] == 5 and args["topk"] == 2048
        assert args["layout_q"] == "TND" and args["layout_k"] == "PA_BBND"
        assert args["mask_mode"] == 3 and args["cmp_ratio"] == 1
        assert torch.equal(args["cu_seqlens_q"], torch.cat((torch.zeros(1, dtype=torch.int32), query_ends)))
        assert args["seqused_k"] is key_lengths
    assert metadata_op.call_args.kwargs["head_dim"] == head_dim
    assert metadata_op.call_args.kwargs["num_heads_q"] == heads
    assert select_op.call_args.kwargs["metadata"] is kernel_metadata
    assert select_op.call_args.kwargs["block_table"] is metadata.block_table


def test_c4_cache_spec_accounts_for_packed_key_and_mx_scales():
    spec = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
        scale_dim=4,
        scale_dtype=torch.float8_e8m0fnu,
        cache_sparse_li_c4=True,
    )
    assert spec.real_page_size_bytes == 128 * (64 + 4)
    assert spec.page_size_bytes == spec.real_page_size_bytes


@pytest.mark.parametrize("padded_tokens,actual_tokens", [(1, 1), (4, 1), (4, 0)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_pcp_builder_resolves_replicated_draft_layout_with_padding(
    mock_cos_sin, mock_config, padded_tokens, actual_tokens
):
    common = _make_common_metadata()
    common.num_input_tokens = padded_tokens
    common.num_actual_tokens = actual_tokens
    common.slot_mapping = torch.full((padded_tokens,), -1, dtype=torch.int64)
    common.positions = torch.arange(padded_tokens)
    common.pcp_context = None
    mock_cos_sin.return_value = (torch.zeros(padded_tokens, 1, 1, 8), torch.zeros(padded_tokens, 1, 1, 8))
    metadata = _make_builder(pcp_size=2).build(0, common)
    assert not metadata.uses_pcp_sharded_layout
    assert metadata.slot_mapping.numel() == padded_tokens
    assert metadata.cache_scatter_indices.shape == (actual_tokens, 2)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "pcp_size,dcp_size,sharded,local_tokens,actual_tokens,decode_tokens,slots,write_slots",
    [
        (1, 1, False, 4, 3, 0, [127, 128, -1, 511], [127, 128, -1, 511]),
        (1, 1, False, 0, 0, 0, [], []),
        (2, 1, True, 2, 2, 2, [127, -1, 256, 511], [127, -1, 256, 511]),
        (2, 2, True, 2, 2, 0, [127, -1, 256, 511], [127, -1, 256, 511]),
        (2, 2, True, 3, 3, 1, [127, 128, -1, 127, 256, 511], [127, 128, -1, 256, 511]),
        (2, 2, True, 2, 2, 2, [127, 128, 127, 128], [127, 128]),
        (2, 1, False, 4, 1, 0, [127, -1, -1, -1], [127]),
        (2, 1, False, 4, 0, 0, [-1, -1, -1, -1], []),
    ],
)
def test_cache_scatter_indices_follow_final_pcp_write_order(
    slot_dtype, pcp_size, dcp_size, sharded, local_tokens, actual_tokens, decode_tokens, slots, write_slots
):
    # Physical manager pages can span several kernel pages. Scatter addresses
    # still use the cache view's 128-token axis, including negative pad slots.
    builder = _make_builder(pcp_size=pcp_size, dcp_size=dcp_size, physical_block_size=256)
    indices = builder._build_cache_scatter_indices(
        torch.tensor(slots, dtype=slot_dtype), local_tokens, actual_tokens, decode_tokens, sharded, "step"
    )
    expected_slots = torch.tensor(write_slots, dtype=slot_dtype)
    expected = torch.stack((expected_slots // 128, expected_slots % 128), dim=-1)
    torch.testing.assert_close(indices, expected)


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_cache_scatter_indices_refresh_capture_storage_and_isolate_draft_steps(mock_cos_sin, mock_config):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    builder = _make_builder(num_speculative_tokens=3, physical_block_size=256)
    common = _make_common_metadata()
    captured = builder.build_for_graph_capture(common)
    common.slot_mapping[:4] = torch.tensor([127, 128, -1, 511])
    runtime = builder.build(0, common)
    assert captured.cache_scatter_indices.data_ptr() == runtime.cache_scatter_indices.data_ptr()
    expected_capture = torch.tensor([[0, 127], [1, 0], [-1, 127], [3, 127]])
    torch.testing.assert_close(captured.cache_scatter_indices, expected_capture)

    # A different persistent draft-step slot tensor must not overwrite the
    # addresses retained by the target or another captured draft graph.
    common.slot_mapping = torch.tensor([256, 257, 258, -1, -1])
    draft = builder.build_for_drafting(common, 1)
    assert draft.cache_scatter_indices.data_ptr() != captured.cache_scatter_indices.data_ptr()
    torch.testing.assert_close(captured.cache_scatter_indices, expected_capture)
    common.slot_mapping[:4] = torch.tensor([383, 384, -1, -1])
    rebuilt = builder.build_for_drafting(common, 1)
    assert rebuilt.cache_scatter_indices.data_ptr() == draft.cache_scatter_indices.data_ptr()
    torch.testing.assert_close(draft.cache_scatter_indices, torch.tensor([[2, 127], [3, 0], [-1, 127], [-1, 127]]))


def test_replicated_pcp_draft_invalidates_captured_padding_when_live_prefix_shrinks():
    builder = _make_builder(pcp_size=2)
    slots = torch.tensor([127, 128, 255, 256], dtype=torch.int64)
    captured = builder._build_cache_scatter_indices(slots, 4, 4, 0, False, "draft")
    slots.copy_(torch.tensor([511, -1, -1, -1]))
    runtime = builder._build_cache_scatter_indices(slots, 4, 1, 0, False, "draft")
    assert captured.data_ptr() == runtime.data_ptr()
    assert runtime.shape == (1, 2)
    torch.testing.assert_close(captured, torch.tensor([[3, 127], [-1, 127], [-1, 127], [-1, 127]]))
