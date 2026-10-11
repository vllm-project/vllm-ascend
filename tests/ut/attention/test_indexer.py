# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch_npu
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPMetadataBuilder
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


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.slot_mapping is common.slot_mapping


@pytest.mark.parametrize("dcp_size,expected_decode_tokens", [(1, 1), (2, 1)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_pcp_decode_boundary(
    mock_cos_sin, mock_get_ascend_config, dcp_size, expected_decode_tokens
):
    mock_cos_sin.return_value = (torch.zeros(4, 1, 1, 8), torch.zeros(4, 1, 1, 8))
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 3
    common.is_prefilling = torch.tensor([False, True])

    metadata = _make_builder(pcp_size=2, dcp_size=dcp_size).build(0, common)

    # Preserve the leading-decode count for PCP with or without DCP.
    assert metadata.num_decode_tokens == expected_decode_tokens


@pytest.mark.parametrize("pd_decode_recompute,expected_decode_tokens", [(False, 0), (True, 2)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_rescheduled_prefill_boundary(
    mock_cos_sin,
    mock_get_ascend_config,
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
        metadata = _make_builder(pcp_size=2, dcp_size=2).build(0, common)

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


@pytest.mark.parametrize("dcp_size,expected_slots", [(1, [1, 2, 3, -1]), (2, [0, 1, 512, -1])])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.get_tp_group")
def test_sfa_indexer_metadata_builder_pads_dsa_slots(
    mock_get_tp_group, mock_cos_sin, mock_get_ascend_config, dcp_size, expected_slots
):
    mock_get_tp_group.return_value.world_size = 4
    mock_get_tp_group.return_value.rank_in_group = 0
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
    build_kwargs = dict(
        pcp_context=pcp_context,
        pcp_cache_group_idx=0,
    )
    if for_cudagraph_capture:
        metadata = builder.build_for_cudagraph_capture(common, **build_kwargs)
    else:
        metadata = builder.build(0, common, **build_kwargs)

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([2816, 2560, 2688, -1], dtype=torch.int32),
    )


@pytest.mark.parametrize("with_dependency", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_indexer_orders_cache_gathers_after_query_dependency(with_dependency, quantized):
    events = []
    k = torch.zeros(2, 4)
    scale = torch.ones(2, 1) if quantized else None
    slots = torch.arange(2)
    metadata = SimpleNamespace(cos=None, sin=None, slot_mapping=slots)
    indexer = SimpleNamespace(
        _pcp_active=False,
        _dsa_cp_active=True,
        enable_sparse_li_c8=quantized,
        enable_sparse_li_quant=quantized,
    )

    def forward_k(*args):
        events.append("forward_k")
        return k, scale, None

    def gather(tensor, group, async_op):
        label = "scale" if tensor is scale else "k"
        events.append("gather_" + label)
        return tensor, SimpleNamespace(wait=lambda: events.append("wait_" + label))

    indexer.forward_k = forward_k
    indexer._gather_cache_inputs = lambda *args: AscendSFAIndexerBackend._gather_cache_inputs(indexer, *args)
    indexer.write_cache = lambda *args, **kwargs: events.append("write_cache")
    dependency = SimpleNamespace(wait=lambda: events.append("wait_q")) if with_dependency else None
    with (
        patch("vllm_ascend.attention.indexer.get_tp_group", return_value=object()),
        patch("vllm_ascend.attention.indexer.all_gather_async", side_effect=gather),
    ):
        assert (
            AscendSFAIndexerBackend.forward(
                indexer, k, k, k, metadata, compute_topk=False, attn_q_gather_handle=dependency
            )
            is None
        )
    expected = ["forward_k"] + (["wait_q"] if with_dependency else []) + ["gather_k"]
    if quantized:
        expected += ["gather_scale", "wait_k", "wait_scale"]
    else:
        expected += ["wait_k"]
    assert events == expected + ["write_cache"]


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
