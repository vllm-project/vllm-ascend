# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
    _balanced_query_shard_sizes,
    _query_parallel_supported,
)
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.device_op import A5DeviceAdaptor

_KERNEL_BLOCK_SIZE = 128


@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "tp_size", "expected"),
    [
        ([8], [8], 4, [3, 2, 1, 2]),
        ([4, 4], [4, 104], 4, [4, 1, 1, 2]),
        ([2, 2, 2, 2], [2, 2, 2, 2], 4, [1, 2, 2, 3]),
        ([3], [0], 2, None),
        ([1], [1], 2, None),
    ],
)
def test_query_shard_planner_balances_causal_work(query_lens, seq_lens, tp_size, expected):
    # Small PCP-local batches must work with both supported core versions;
    # a release core's CUDA row threshold must not disable this host plan.
    assert _balanced_query_shard_sizes(torch.tensor(seq_lens), torch.tensor(query_lens), tp_size) == expected


def _make_builder(
    pcp_size: int = 1,
    dcp_size: int = 1,
    num_speculative_tokens: int | None = None,
    tp_size: int | None = None,
    max_num_batched_tokens: int = 16,
    min_query_rows: int = 65536,
    max_model_len: int = 1024,
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
        max_model_len=max_model_len,
    )
    vllm_config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=max(4, dcp_size) if tp_size is None else tp_size,
    )
    vllm_config.scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    vllm_config.attention_config = SimpleNamespace(indexer_kv_dtype="fp8", indexer_tp_shard_min_rows=min_query_rows)
    vllm_config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
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
    # The proposer owns one persistent slot tensor per logical draft step;
    # its address is also the key for every derived metadata buffer.
    common.slot_mapping = common.slot_mapping.clone()
    common.positions = torch.tensor([2, 3, 2, 3, 9])
    step_two = builder.build_for_drafting(common, 2)

    assert step_one.slot_mapping.data_ptr() != step_two.slot_mapping.data_ptr()
    assert step_one.block_table.data_ptr() != step_two.block_table.data_ptr()
    torch.testing.assert_close(step_one.slot_mapping, step_one_slots)
    torch.testing.assert_close(step_one.block_table, step_one_blocks)


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

    for field in ("slot_mapping", "cos", "sin"):
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


def _query_parallel_config(**overrides):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        attention_config=SimpleNamespace(indexer_kv_dtype="fp8", indexer_tp_shard_min_rows=256),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048), hf_config=SimpleNamespace()),
    )
    for key, value in overrides.items():
        for section in vars(config).values():
            if hasattr(section, key):
                setattr(section, key, value)
    return config


@pytest.mark.parametrize(
    "overrides,enabled,offload,invariant,expected",
    [
        ({}, True, False, False, True),
        ({}, False, False, False, False),
        ({"tensor_parallel_size": 1}, True, False, False, False),
        ({"prefill_context_parallel_size": 2}, True, False, False, False),
        ({"decode_context_parallel_size": 2}, True, False, False, True),
        ({"decode_context_parallel_size": 2, "tensor_parallel_size": 4}, True, False, False, False),
        ({"decode_context_parallel_size": 4, "tensor_parallel_size": 4}, True, False, False, True),
        (
            {"prefill_context_parallel_size": 4, "tensor_parallel_size": 4, "decode_context_parallel_size": 16},
            True,
            False,
            False,
            True,
        ),
        (
            {"prefill_context_parallel_size": 4, "tensor_parallel_size": 4, "decode_context_parallel_size": 4},
            True,
            False,
            False,
            False,
        ),
        ({"cudagraph_mode": CUDAGraphMode.FULL}, True, False, False, False),
        ({"indexer_kv_dtype": "mxfp4"}, True, False, False, False),
        ({"indexer_kv_dtype": "int8"}, True, False, False, True),
        ({"indexer_kv_dtype": "bf16"}, True, False, False, True),
        ({}, True, True, False, False),
        ({}, True, False, True, False),
    ],
)
def test_query_parallel_rejects_incompatible_layouts(overrides, enabled, offload, invariant, expected):
    ascend_config = SimpleNamespace(
        enable_indexer_query_parallel=enabled, sparse_kv_offload_config=SimpleNamespace(enabled=offload)
    )
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", invariant),
    ):
        assert _query_parallel_supported(_query_parallel_config(**overrides)) is expected


@pytest.mark.parametrize(
    "enabled,pcp,min_rows,expected_warning",
    [
        (False, 8, 65536, None),
        (True, 8, 65536, "PCP=8"),
        (True, 1, 65536, "max_num_batched_tokens=16384"),
        (True, 1, 2048, None),
    ],
)
def test_query_parallel_reports_why_requested_optimization_cannot_run(caplog, enabled, pcp, min_rows, expected_warning):
    """A requested flag must not silently imply activation on an ineligible deployment."""
    ascend_config = SimpleNamespace(
        enable_indexer_query_parallel=enabled, sparse_kv_offload_config=SimpleNamespace(enabled=False)
    )
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", False),
    ):
        builder = _make_builder(pcp_size=pcp, tp_size=2, max_num_batched_tokens=16384, min_query_rows=min_rows)
    assert builder.query_parallel_enabled == (enabled and pcp == 1)
    messages = [record.getMessage() for record in caplog.records if "SFA indexer query parallel" in record.getMessage()]
    if expected_warning is None:
        assert not messages
    else:
        assert any(expected_warning in message for message in messages)


def _query_shard_builder(query_lens, seq_lens, tp_size=2):
    with patch("vllm_ascend.attention.indexer.get_ascend_config"):
        builder = _make_builder()
    builder.vllm_config = _query_parallel_config(tensor_parallel_size=tp_size)
    builder.query_parallel_enabled = True
    common = _make_common_metadata()
    common.num_reqs = len(query_lens)
    common.num_actual_tokens = common.num_input_tokens = sum(query_lens)
    common.query_start_loc_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = common.seq_lens.clone()
    return builder, common


@pytest.mark.parametrize(
    "query_lens,seq_lens,tp",
    [
        ([333, 184], [700, 400], 2),
        ([1024], [1024], 4),
        ([4096, 1024], [4096, 1000000], 8),
        ([16384], [32768], 16),
        ([129, 8192], [129, 32768], 16),
    ],
)
def test_query_shards_preserve_every_causal_row_and_bound_padding(query_lens, seq_lens, tp):
    builder, common = _query_shard_builder(query_lens, seq_lens, tp)
    expected = [list(range(k - q + 1, k + 1)) for q, k in zip(query_lens, seq_lens)]
    observed: list[list[int]] = [[] for _ in query_lens]
    shards = []
    for rank in range(tp):
        with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=rank)):
            shard = builder._build_query_shard(common)
        assert shard is not None
        shards.append(shard)
        local_queries = shard.query_lens.diff(prepend=torch.tensor([0])).tolist()
        for req, (q, k) in enumerate(zip(local_queries, shard.key_lens.tolist())):
            if q == 0:
                assert k == 0
            observed[req].extend(range(k - q + 1, k + 1))
    assert observed == expected
    assert shards[0].start == 0 and shards[-1].end == sum(query_lens)
    assert all(a.end == b.start for a, b in zip(shards, shards[1:]))
    assert min(shards[0].sizes) >= 128
    assert max(shards[0].sizes) <= 2 * ((sum(query_lens) + tp - 1) // tp)
    starts = common.query_start_loc_cpu[:-1].tolist()
    for shard in shards[:-1]:
        request_start = max(start for start in starts if start <= shard.end)
        assert (shard.end - request_start) % 128 == 0


@pytest.mark.parametrize(("rows", "eligible"), [(4096, False), (65536, True)])
def test_query_shard_legacy_core_without_configurable_threshold(rows, eligible):
    builder, common = _query_shard_builder([rows], [rows], 4)
    del builder.vllm_config.attention_config.indexer_tp_shard_min_rows
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config") as ascend,
        patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=0)),
    ):
        ascend.return_value.enable_indexer_query_parallel = True
        builder._warn_if_query_parallel_inactive(builder.vllm_config)
        assert (builder._build_query_shard(common) is not None) is eligible


def test_query_shards_fall_back_when_aligned_cuts_make_a_rank_too_small():
    builder, common = _query_shard_builder([191, 193], [4096, 4096], 3)
    with patch("vllm_ascend.attention.indexer.get_tp_group") as group:
        assert builder._build_query_shard(common) is None
        group.assert_not_called()


@pytest.mark.parametrize("case", ["small", "padded", "mixed", "decode", "missing_cpu_lengths", "inconsistent_rows"])
def test_query_shard_fallbacks_do_not_enter_collectives(case):
    builder, common = _query_shard_builder([333, 184], [700, 400])
    if case == "small":
        builder.vllm_config.attention_config.indexer_tp_shard_min_rows = 1024
    elif case == "padded":
        common.num_input_tokens += 1
    elif case == "mixed":
        common.query_start_loc_cpu = torch.tensor([0, 1, 517])
    elif case == "decode":
        builder.decode_threshold = 333
    elif case == "missing_cpu_lengths":
        common.seq_lens_cpu_upper_bound = None
    elif case == "inconsistent_rows":
        common.num_actual_tokens = common.num_input_tokens = 518
    with patch("vllm_ascend.attention.indexer.get_tp_group") as group:
        assert builder._build_query_shard(common) is None
        group.assert_not_called()


@pytest.mark.parametrize("quantized", [False, True])
def test_query_parallel_topk_matches_full_history_reference(quantized):
    """A CPU scoring oracle checks slicing, scale alignment and gather order."""
    builder, common = _query_shard_builder([333, 184], [700, 400])
    torch.manual_seed(7)
    queries = torch.randint(-8, 8, (517, 2, 4)).to(torch.int8 if quantized else torch.bfloat16)
    scales = torch.rand(517, 2) + 0.1 if quantized else None
    weights = torch.randn(517, 2)
    keys = [torch.randn(length, 4, dtype=torch.float64) for length in (700, 400)]
    metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=517,
        slot_mapping=torch.arange(517),
        seq_lens=common.seq_lens,
        cum_query_lens=common.query_start_loc[1:],
        block_table=torch.tensor([[0], [1]]),
        sin=torch.empty(0),
        cos=torch.empty(0),
        actual_seq_lengths_query=common.query_start_loc[1:],
        actual_seq_lengths_key=common.seq_lens,
    )
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.enable_sparse_li_c8 = quantized
    backend.enable_sparse_li_c4 = False
    backend.use_torch_npu_lightning_indexer = True
    backend.k_cache = SimpleNamespace(kv_cache=(object(), object()))

    def score(q, scale, shape, w, cache, k_slot, scale_slot, meta, q_ends, k_ends, *unused):
        assert cache is backend.k_cache.kv_cache
        assert meta.block_table is metadata.block_table
        q = q.reshape(shape or q.shape).double()
        if scale is not None:
            q = q * scale.reshape(q.shape[:-1]).double().unsqueeze(-1)
        result = []
        start = 0
        for req, (end, key_end) in enumerate(zip(q_ends.tolist(), k_ends.tolist())):
            for row in range(start, end):
                visible = key_end - (end - row - 1)
                logits = (torch.relu(q[row] @ keys[req][:visible].T) * w[row].double()[:, None]).sum(0)
                result.append(logits.topk(8).indices.to(torch.int32))
            start = end
        return torch.stack(result)

    shape = queries.shape if quantized else None
    with patch("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process", side_effect=score):
        expected = backend.select_topk(queries, scales, shape, weights, metadata)
        local_results, plans = [], []
        for rank in range(2):
            with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=rank)):
                plan = builder._build_query_shard(common)
            plans.append(plan)
            metadata.query_shard = plan
            with patch.object(plan, "gather", side_effect=lambda result: result):
                local_results.append(backend.select_topk(queries, scales, shape, weights, metadata))
        assert torch.equal(torch.cat(local_results), expected)
        max_rows = max(plans[0].sizes)
        payload = torch.full((2 * max_rows, 8), -1, dtype=torch.int32)
        for rank, result in enumerate(local_results):
            payload[rank * max_rows : rank * max_rows + len(result)] = result
        for plan, result in zip(plans, local_results):
            group = SimpleNamespace(all_gather=Mock(return_value=payload))
            with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=group):
                assert torch.equal(plan.gather(result), expected)
            sent = group.all_gather.call_args.args[0]
            assert torch.equal(sent[: len(result)], result)
            assert (sent[len(result) :] == -1).all()
    assert torch.equal(metadata.actual_seq_lengths_key, torch.tensor([700, 400]))
    assert torch.equal(metadata.actual_seq_lengths_query, torch.tensor([333, 517]))


def test_query_sharding_keeps_custom_selector_authoritative():
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    selector = Mock(return_value=torch.tensor([[3]], dtype=torch.int32))
    metadata = SimpleNamespace(topk_selector=selector, query_shard=Mock())
    with patch("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process") as native:
        assert torch.equal(
            backend.select_topk(torch.empty(0), None, None, torch.empty(0), metadata), selector.return_value
        )
        native.assert_not_called()
        metadata.query_shard.gather.assert_not_called()


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_query_parallel_profile_reserves_padded_exchange_peak(dcp_size):
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.topk_tokens = 2048
    config = _query_parallel_config(decode_context_parallel_size=dcp_size)
    with (
        patch("vllm_ascend.attention.indexer.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.indexer.get_ascend_config") as ascend,
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", False),
        patch("vllm_ascend.attention.indexer.torch.empty") as empty,
    ):
        ascend.return_value.enable_indexer_query_parallel = True
        ascend.return_value.sparse_kv_offload_config.enabled = False
        backend.reserve_query_parallel_workspace(torch.device("cpu"))
        rows, width = empty.call_args.args[0]
        assert rows >= (2 + 2) * (2 * (8192 // 2)) + 8192
        assert width == 2048 and empty.call_args.kwargs["dtype"] == torch.int32


@pytest.mark.parametrize(
    "entry", ["build", "build_for_drafting", "build_for_graph_capture", "build_for_cudagraph_capture"]
)
def test_only_normal_prefill_builder_publishes_query_shards(entry):
    builder, common = _query_shard_builder([333, 184], [700, 400])
    common.positions = common.slot_mapping = torch.arange(517)
    builder._rope_capacity = 517
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config"),
        patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=0)),
        patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla", return_value=(torch.zeros(517, 1, 1, 8),) * 2),
    ):
        if entry == "build":
            metadata = builder.build(0, common)
        elif entry == "build_for_drafting":
            metadata = builder.build_for_drafting(common, 0)
        else:
            metadata = getattr(builder, entry)(common)
    assert (metadata.query_shard is not None) == (entry == "build")


def test_host_cost_estimate_never_replaces_live_causal_bounds():
    builder, common = _query_shard_builder([333, 184], [700, 400])
    common.seq_lens_cpu_upper_bound += 10000
    with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=0)):
        shard = builder._build_query_shard(common)
    assert shard is not None
    assert shard.key_lens[0] == 700 - max(333 - shard.end, 0)
    assert shard.key_lens.max() <= 700


@pytest.mark.parametrize("unsupported", ["kpool", "compressed", "missing_model"])
def test_query_parallel_dcp_requires_ordinary_replicated_indexer(unsupported):
    config = _query_parallel_config(decode_context_parallel_size=2)
    if unsupported == "kpool":
        config.model_config.hf_text_config.index_kpool = 4
    elif unsupported == "compressed":
        config.model_config.hf_text_config.compress_ratios = [1, 2]
    else:
        config.model_config = None
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config") as ascend,
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", False),
    ):
        ascend.return_value.enable_indexer_query_parallel = True
        ascend.return_value.sparse_kv_offload_config.enabled = False
        assert not _query_parallel_supported(config)


def _dcp_query_parallel_case(dcp_size, physical_block_size):
    """Build nonidentity physical pages independently of production address helpers."""
    query_lens, key_lens = [768, 512], [1537, 1151]
    rows = sum(query_lens)
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config") as ascend,
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", False),
    ):
        ascend.return_value.enable_indexer_query_parallel = True
        ascend.return_value.sparse_kv_offload_config.enabled = False
        builder = _make_builder(
            dcp_size=dcp_size,
            tp_size=dcp_size,
            max_num_batched_tokens=rows,
            min_query_rows=256,
            max_model_len=2048,
            physical_block_size=physical_block_size,
        )
    # Exercise both one and two logical indexer blocks per physical page.
    assert builder.blocks_per_phys_block == physical_block_size // _KERNEL_BLOCK_SIZE
    pages_per_request = (2048 + physical_block_size * dcp_size - 1) // (physical_block_size * dcp_size)
    pages = 3 + 5 * torch.arange(pages_per_request).unsqueeze(0) + 31 * torch.arange(2).unsqueeze(1)
    subblocks = torch.arange(builder.blocks_per_phys_block)
    local_table = (pages.unsqueeze(-1) * builder.blocks_per_phys_block + subblocks).flatten(1)
    common = _make_common_metadata()
    common.num_actual_tokens = common.num_input_tokens = rows
    common.query_start_loc_cpu = torch.tensor([0, query_lens[0], rows], dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor(key_lens, dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = common.seq_lens + 10000
    common.max_query_len = max(query_lens)
    common.is_prefilling = torch.ones(2, dtype=torch.bool)
    common.positions = torch.cat([torch.arange(k - q, k) for q, k in zip(query_lens, key_lens)])
    common.block_table_tensor = local_table.to(torch.int32)
    common.slot_mapping = torch.full((rows,), -1, dtype=torch.int64)
    common.dcp_local_seq_lens = torch.zeros(2, dtype=torch.int32)
    addresses, keys = [], []
    torch.manual_seed(31)
    cache = torch.zeros(((int(pages.max()) + 1) * dcp_size * physical_block_size, 4), dtype=torch.float64)
    for req, length in enumerate(key_lens):
        positions = torch.arange(length)
        page = pages[req, positions // (physical_block_size * dcp_size)]
        owner = positions // physical_block_size % dcp_size
        address = (page * dcp_size + owner) * physical_block_size + positions % physical_block_size
        key = torch.randn(length, 4, dtype=torch.float64)
        cache[address] = key
        addresses.append(address)
        keys.append(key)
    return builder, common, query_lens, key_lens, addresses, keys, cache


def _causal_indexer_reference(query, weights, keys, key_end):
    logits = (torch.relu(torch.einsum("qhd,kd->qhk", query.double(), keys)) * weights.double().unsqueeze(-1)).sum(1)
    horizon = key_end - query.shape[0] + torch.arange(query.shape[0]) + 1
    logits.masked_fill_(torch.arange(keys.shape[0]).unsqueeze(0) >= horizon.unsqueeze(1), float("-inf"))
    return logits.topk(8, dim=-1).indices.to(torch.int32)


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("physical_block_size", [128, 256])
@pytest.mark.parametrize("quantized", [False, True])
def test_dcp_query_parallel_matches_global_history_and_physical_cache(dcp_size, physical_block_size, quantized):
    builder, common, q_lens, k_lens, addresses, keys, cache = _dcp_query_parallel_case(dcp_size, physical_block_size)
    assert builder.query_parallel_enabled and builder.use_dcp
    common_table_before = common.block_table_tensor.clone()
    cache_before = cache.clone()
    rows = sum(q_lens)
    queries = torch.randint(-8, 8, (rows, 2, 4)).to(torch.int8 if quantized else torch.bfloat16)
    scales = torch.rand(rows, 2) + 0.1 if quantized else None
    weights = torch.randn(rows, 2)
    dequantized = queries.double() * scales.double().unsqueeze(-1) if quantized else queries.double()
    expected = torch.cat(
        [
            _causal_indexer_reference(q, w, key, k)
            for q, w, key, k in zip(dequantized.split(q_lens), weights.split(q_lens), keys, k_lens)
        ]
    )
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.enable_sparse_li_c8 = quantized
    backend.enable_sparse_li_c4 = False
    backend.use_torch_npu_lightning_indexer = True
    backend.k_cache = SimpleNamespace(kv_cache=(cache, object()))

    def score(q, scale, shape, w, kv, k_slot, scale_slot, meta, q_ends, k_ends, *unused):
        assert kv is backend.k_cache.kv_cache
        q = q.reshape(shape or q.shape).double()
        if scale is not None:
            q = q * scale.reshape(q.shape[:-1]).double().unsqueeze(-1)
        result, start = [], 0
        for req, (end, key_end) in enumerate(zip(q_ends.tolist(), k_ends.tolist())):
            if end > start:
                positions = torch.arange(key_end)
                slots = meta.block_table[req, positions // _KERNEL_BLOCK_SIZE] * _KERNEL_BLOCK_SIZE
                slots = slots + positions % _KERNEL_BLOCK_SIZE
                assert torch.equal(slots, addresses[req][:key_end])
                result.append(_causal_indexer_reference(q[start:end], w[start:end], kv[0][slots], key_end))
            start = end
        return torch.cat(result)

    plans, results = [], []
    with (
        patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla", return_value=(torch.zeros(rows, 1, 1, 8),) * 2),
        patch("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process", side_effect=score),
    ):
        for rank in range(dcp_size):
            with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=rank)):
                metadata = builder.build(0, common)
            plan = metadata.query_shard
            assert plan is not None
            assert torch.equal(metadata.actual_seq_lengths_key, common.seq_lens)
            assert metadata.num_decode_tokens == 0
            expected_slots = torch.cat([address[-q:] for address, q in zip(addresses, q_lens)])
            assert torch.equal(metadata.slot_mapping, expected_slots)
            with patch.object(plan, "gather", side_effect=lambda value: value):
                result = backend.select_topk(queries, scales, queries.shape if quantized else None, weights, metadata)
            assert result.shape[0] == plan.end - plan.start
            plans.append(plan)
            results.append(result)
    assert torch.equal(torch.cat(results), expected)
    max_rows = max(plans[0].sizes)
    payload = torch.full((dcp_size * max_rows, 8), -1, dtype=torch.int32)
    for rank, result in enumerate(results):
        payload[rank * max_rows : rank * max_rows + result.shape[0]] = result
    for plan, result in zip(plans, results):
        group = SimpleNamespace(all_gather=Mock(return_value=payload))
        with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=group):
            assert torch.equal(plan.gather(result), expected)
    assert torch.equal(common.block_table_tensor, common_table_before)
    assert torch.equal(cache, cache_before)


@pytest.mark.parametrize(
    "entry", ["build", "build_for_drafting", "build_for_graph_capture", "build_for_cudagraph_capture"]
)
def test_dcp_only_ordinary_prefill_publishes_query_shards(entry):
    builder, common, *_ = _dcp_query_parallel_case(2, 128)
    with (
        patch("vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=0)),
        patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla", return_value=(torch.zeros(1280, 1, 1, 8),) * 2),
    ):
        if entry == "build":
            metadata = builder.build(0, common)
        elif entry == "build_for_drafting":
            metadata = builder.build_for_drafting(common, 0)
        else:
            metadata = getattr(builder, entry)(common)
    assert (metadata.query_shard is not None) == (entry == "build")


def _pcp_query_parallel_case(pcp, tp, rank, total=4096, history=257):
    """A real PCP metadata layout: two virtual requests alias one paged history."""
    chunk = (total + 2 * pcp - 1) // (2 * pcp)
    offsets = [rank * chunk, (2 * pcp - 1 - rank) * chunk]
    lengths = [min(chunk, total - offset) for offset in offsets]
    ends = [history + offset + length for offset, length in zip(offsets, lengths)]
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config") as ascend,
        patch("vllm_ascend.attention.indexer.envs.VLLM_BATCH_INVARIANT", False),
    ):
        ascend.return_value.enable_indexer_query_parallel = True
        ascend.return_value.sparse_kv_offload_config.enabled = False
        builder = _make_builder(
            pcp_size=pcp,
            tp_size=tp,
            dcp_size=pcp * tp,
            max_model_len=history + total,
            max_num_batched_tokens=total,
            min_query_rows=256,
        )
    common = _make_common_metadata()
    common.num_actual_tokens = common.num_input_tokens = sum(lengths)
    common.query_start_loc_cpu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor(ends, dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = common.seq_lens + 10000
    common.max_query_len = max(lengths)
    common.positions = torch.cat([torch.arange(end - length, end) for length, end in zip(lengths, ends)])
    common.slot_mapping = torch.full((sum(lengths),), -1, dtype=torch.int64)
    common.dcp_local_seq_lens = torch.ones(2, dtype=torch.int32)  # deliberately not the indexer history
    pages = 3 + 5 * torch.arange((history + total + 128 * pcp * tp - 1) // (128 * pcp * tp))
    common.block_table_tensor = pages.to(torch.int32).unsqueeze(0).repeat(2, 1)
    starts = torch.tensor([0, total], dtype=torch.int32)
    gathered = torch.cat(
        [
            torch.arange(part * chunk, min((part + 1) * chunk, total))
            for owner in range(pcp)
            for part in [owner, 2 * pcp - 1 - owner]
        ]
    )
    common.pcp_context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=total,
            query_start_loc=starts,
            query_start_loc_np=starts.numpy(),
            seq_lens=torch.tensor([history + total]),
            positions=torch.arange(history, history + total),
            is_prefilling_np=torch.tensor([True]).numpy(),
        ),
        global_block_tables=(pages.to(torch.int32).unsqueeze(0),),
        padded_gather_idx=gathered,
        gathered_kv_write_mask=torch.ones(total, dtype=torch.bool),
    )
    common.pcp_cache_group_idx = 0
    return builder, common, lengths


@pytest.mark.parametrize("pcp,tp", [(2, 4), (4, 4), (8, 2)])
@pytest.mark.parametrize("total", [4096, 4099])
@pytest.mark.parametrize("quantized", [False, True])
def test_pcp_query_shards_preserve_dual_chunk_causality_and_topk(pcp, tp, total, quantized):
    """Compare every PCP/TP rank against a full-history scoring oracle, including odd tails."""
    torch.manual_seed(53)
    keys = torch.randn(total + 257, 4, dtype=torch.float64)
    for pcp_rank in range(pcp):
        builder, common, lengths = _pcp_query_parallel_case(pcp, tp, pcp_rank, total)
        assert builder.query_parallel_enabled
        rows = sum(lengths)
        query = torch.randint(-8, 8, (rows, 2, 4)).to(torch.int8 if quantized else torch.bfloat16)
        scale = torch.rand(rows, 2) + 0.1 if quantized else None
        weights = torch.rand(rows, 2) + 0.1
        dequantized = query.double() * scale.double().unsqueeze(-1) if quantized else query.double()
        expected = torch.cat(
            [
                _causal_indexer_reference(q, w, keys, end)
                for q, w, end in zip(dequantized.split(lengths), weights.split(lengths), common.seq_lens.tolist())
            ]
        )
        backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        torch.nn.Module.__init__(backend)
        backend.enable_sparse_li_c8 = quantized
        backend.enable_sparse_li_c4 = False
        backend.use_torch_npu_lightning_indexer = True
        backend.k_cache = SimpleNamespace(kv_cache=(object(), object()))

        def score(q, qs, shape, w, kv, k_slot, scale_slot, meta, q_ends, k_ends, *unused):
            q = q.reshape(shape or q.shape).double()
            if qs is not None:
                q = q * qs.reshape(q.shape[:-1]).double().unsqueeze(-1)
            result, start = [], 0
            for end, key_end in zip(q_ends.tolist(), k_ends.tolist()):
                if end > start:
                    result.append(_causal_indexer_reference(q[start:end], w[start:end], keys, key_end))
                start = end
            return torch.cat(result)

        plans, outputs = [], []
        with (
            patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla", return_value=(torch.zeros(rows, 1, 1, 8),) * 2),
            patch("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process", side_effect=score),
        ):
            for rank in range(tp):
                with patch(
                    "vllm_ascend.attention.indexer.get_tp_group", return_value=SimpleNamespace(rank_in_group=rank)
                ):
                    metadata = builder.build(0, common)
                plan = metadata.query_shard
                assert plan is not None and metadata.uses_pcp_sharded_layout
                assert torch.equal(metadata.actual_seq_lengths_key, common.seq_lens)
                assert torch.equal(metadata.block_table[0], metadata.block_table[1])
                assert metadata.slot_mapping.numel() == total  # PCP K writes remain global
                with patch.object(plan, "gather", side_effect=lambda value: value):
                    outputs.append(
                        backend.select_topk(query, scale, query.shape if quantized else None, weights, metadata)
                    )
                plans.append(plan)
        assert torch.equal(torch.cat(outputs), expected)
        for plan in plans[:-1]:
            start = max(x for x in common.query_start_loc_cpu[:-1].tolist() if x <= plan.end)
            assert (plan.end - start) % 128 == 0
        padded_rows = max(plans[0].sizes)
        gathered = torch.full((tp * padded_rows, 8), -1, dtype=torch.int32)
        for rank, output in enumerate(outputs):
            gathered[rank * padded_rows : rank * padded_rows + len(output)] = output
        group = SimpleNamespace(all_gather=Mock(return_value=gathered))
        with patch("vllm_ascend.attention.indexer.get_tp_group", return_value=group):
            for plan, output in zip(plans, outputs):
                assert torch.equal(plan.gather(output), expected)


@pytest.mark.parametrize("case", ["padded", "mixed", "capture", "draft", "legacy", "threshold"])
def test_pcp_query_parallel_fallbacks(case):
    builder, common, _ = _pcp_query_parallel_case(4, 4, 0)
    if case == "padded":
        common.num_actual_tokens -= 1
    elif case == "mixed":
        common.query_start_loc_cpu[1] = common.query_start_loc[1] = 1
    elif case == "legacy":
        common.pcp_context = None
    elif case == "threshold":
        builder.vllm_config.attention_config.indexer_tp_shard_min_rows = 8192
    with (
        patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla", return_value=(torch.zeros(1024, 1, 1, 8),) * 2),
        patch("vllm_ascend.attention.indexer.get_tp_group") as group,
    ):
        if case == "capture":
            metadata = builder.build_for_cudagraph_capture(common)
        elif case == "draft":
            metadata = builder.build_for_drafting(common, 0)
        else:
            metadata = builder.build(0, common)
        assert metadata.query_shard is None
        group.assert_not_called()
