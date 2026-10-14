# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.dsa_v1 import (
    DSA_METADATA_BUFFER_SIZE,
    AscendDSAImpl,
    AscendDSALayerMetadata,
    AscendDSAMetadata,
    AscendDSAReqMetadata,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


def _make_req_metadata(seq_lens, block_table, query_start_loc=None):
    if query_start_loc is None:
        query_start_loc = list(range(len(seq_lens) + 1))
    return AscendDSAReqMetadata(
        block_table=torch.tensor(block_table, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        slot_mapping=None,
        storage_block_size=2,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        sas_metadata=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32),
    )


def _make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_attention_heads=16,
                sliding_window=8,
                index_topk=8,
                index_n_heads=2,
                index_head_dim=4,
            ),
            get_head_size=lambda: 3,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )


def _make_plan(**method_returns):
    plan = MagicMock()
    plan.layout_kv = "PA_ND"
    plan.uses_sparse_flash_mla = False
    plan.uses_kv_compress_epilog = False
    plan.get_dsa_sparse_attn_base_kwargs.side_effect = dict
    plan.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
    for name, value in method_returns.items():
        getattr(plan, name).return_value = value
    return plan


@pytest.mark.parametrize(
    ("dcp_rank", "expected"),
    [
        (0, [[1, 3, 0, -1, -1, -1, -1, -1], [0, 4, -1, -1, -1, -1, -1, -1]]),
        (1, [[1, 2, 3, -1, -1, -1, -1, -1], [0, 4, -1, -1, -1, -1, -1, -1]]),
        (2, [[2, -1, -1, -1, -1, -1, -1, -1], [0, 4, -1, -1, -1, -1, -1, -1]]),
    ],
)
def test_c4_topk_remap_owns_whole_pages_and_preserves_indexer_order(dcp_rank, expected, monkeypatch):
    monkeypatch.setattr(dsa_cp, "HAS_TRITON", False)
    # Two compressed rows per physical page; ownership rotates over three ranks.
    topk = torch.tensor([[[1, 3, 8, 7, 0, -1, 10, 9]], [[-1, 0, 2, 4, 12, 14, 16, -1]]], dtype=torch.int32)
    req = _make_req_metadata([44, 68], [[3, 7], [9, 11]])
    req.dcp_local_cmp_sparse_indices = torch.full_like(topk, -9)
    req.dcp_local_remap_chunk_out = torch.empty((2, 8), dtype=torch.int32)
    req.dcp_local_remap_chunk_count = torch.empty((2, 1), dtype=torch.int32)
    original_topk = topk.clone()

    actual = dsa_cp.remap_dsa_c4_sparse_indices(topk, req, dcp_size=3, dcp_rank=dcp_rank)

    assert actual.data_ptr() == req.dcp_local_cmp_sparse_indices.data_ptr()
    assert actual[:, 0].tolist() == expected
    torch.testing.assert_close(topk, original_topk)


@pytest.mark.parametrize(
    ("dcp_rank", "expected_lengths"),
    [(0, [0, 1, 2, 2, 4, 4, 6, 6, 8]), (1, [0, 0, 1, 2, 3, 4, 5, 6, 7])],
)
def test_c4_decode_metadata_keeps_global_lengths_and_builds_local_scfa_view(dcp_rank, expected_lengths, monkeypatch):
    builder = dsa_cp.AscendDSADCPMetadataBuilder.__new__(dsa_cp.AscendDSADCPMetadataBuilder)
    config = _make_config()
    builder.model_config = config.model_config
    builder.vllm_config = config
    builder.compressor_ratio = 4
    builder.storage_block_size = 2
    builder.dcp_size = 2
    builder.dcp_rank = dcp_rank
    seq_lens = [3, 4, 15, 16, 31, 32, 47, 48, 61]
    req = _make_req_metadata(seq_lens, [[3, 7, 99]] * len(seq_lens))
    builder.common_ratio_to_sas_metadata = {"seq_lens_cpu": req.seq_lens.clone()}
    builder.num_actual_tokens = len(seq_lens)
    builder.dcp_global_compressed_seq_lens_buf = torch.full((10,), -9, dtype=torch.int32)
    builder.dcp_local_compressed_seq_lens_buf = torch.full((10,), -9, dtype=torch.int32)
    builder.dcp_local_seqused_kv_buf = torch.full((10,), -9, dtype=torch.int32)
    builder.dcp_local_sas_metadata_buffer = torch.empty(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    builder.sas_metadata_buffer = torch.full((DSA_METADATA_BUFFER_SIZE,), -9, dtype=torch.int32)
    builder.seqused_q = torch.ones(len(seq_lens), dtype=torch.int32)
    builder.dcp_local_block_table_cols = 2
    builder.dcp_remap_num_chunks = 1
    builder.dcp_local_cmp_sparse_indices_buf = torch.empty((10, 1, 8), dtype=torch.int32)
    builder.dcp_local_remap_chunk_out_buf = torch.empty((10, 8), dtype=torch.int32)
    builder.dcp_local_remap_chunk_count_buf = torch.empty((10, 1), dtype=torch.int32)
    builder.dcp_cmp_only_sinks = torch.full((4,), dsa_cp.DSA_CMP_ONLY_SINK)
    builder._device_metadata_enabled = False
    metadata_op = MagicMock(return_value=torch.arange(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))
    plan = _make_plan(get_dsa_sparse_attn_metadata_op=metadata_op)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", lambda _: plan)

    builder._populate_c4_local_attention_metadata(SimpleNamespace(query_start_loc_cpu=req.query_start_loc), req)

    assert req.seq_lens.tolist() == seq_lens
    assert req.block_table.tolist() == [[3, 7, 99]] * len(seq_lens)
    assert req.dcp_local_block_table.tolist() == [[3, 7]] * len(seq_lens)
    assert req.dcp_local_compressed_seq_lens.tolist() == expected_lengths
    assert req.dcp_local_seqused_kv.tolist() == [length * 4 for length in expected_lengths]
    assert builder.dcp_local_compressed_seq_lens_buf[-1].item() == -9
    assert req.dcp_local_cmp_sparse_indices.shape == (len(seq_lens), 1, 8)
    assert req.dcp_local_cmp_sparse_indices.data_ptr() == builder.dcp_local_cmp_sparse_indices_buf.data_ptr()
    assert req.dcp_local_sas_metadata is builder.dcp_local_sas_metadata_buffer
    assert builder.sas_metadata_buffer.tolist() == [-9] * DSA_METADATA_BUFFER_SIZE
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["num_heads_q"] == 4  # TP=8, DCP=2: only the DCP subgroup's heads are gathered.
    assert kwargs["has_ori_kv"] is False
    assert kwargs["has_cmp_kv"] is True
    assert kwargs["cmp_ratio"] == 4
    assert kwargs["cmp_topk"] == 8
    assert kwargs["seqused_kv"].tolist() == [length * 4 for length in expected_lengths]
    assert kwargs["max_seqlen_kv"] == max(expected_lengths) * 4
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["cu_seqlens_ori_kv"] is None
    assert kwargs["cu_seqlens_cmp_kv"] is None


def test_c4_main_and_indexer_share_global_qli_length_buffers(monkeypatch):
    builders = [dsa_cp.AscendDSADCPMetadataBuilder.__new__(dsa_cp.AscendDSADCPMetadataBuilder) for _ in range(2)]
    for builder in builders:
        builder.model_config = _make_config().model_config
        builder.qli_seqused_k = torch.full((4,), -9, dtype=torch.int32)
        builder.qli_cmp_residual_k = torch.full((4,), -9, dtype=torch.int32)
        builder.qli_metadata_buffer = torch.empty(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
        builder.seqused_q = torch.ones(3, dtype=torch.int32)
    seq_lens = torch.tensor([3, 16, 19], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    metadata_cache = {}
    metadata_op = MagicMock(return_value=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", metadata_op, raising=False)
    monkeypatch.setattr(dsa_cp.DeviceOperator, "get_dsa_indexer_quant_mode", lambda: 0)

    for builder in builders:
        builder._build_qli_metadata(metadata_cache, query_start_loc, seq_lens, 1, 19)
    main_lengths, main_residual = builders[0]._get_qli_length_buffers(metadata_cache, 3)
    indexer_lengths, indexer_residual = builders[1]._get_qli_length_buffers(metadata_cache, 3)

    metadata_op.assert_called_once()
    assert main_lengths.tolist() == [0, 4, 4]
    assert main_residual.tolist() == [3, 0, 3]
    assert indexer_lengths.data_ptr() == main_lengths.data_ptr()
    assert indexer_residual.data_ptr() == main_residual.data_ptr()
    assert metadata_op.call_args.kwargs["seqused_k"].data_ptr() == indexer_lengths.data_ptr()
    assert metadata_op.call_args.kwargs["cmp_residual_k"].data_ptr() == indexer_residual.data_ptr()
    assert builders[1].qli_seqused_k.tolist() == [-9] * 4


def test_prefill_compact_table_restores_global_pages_in_dcp_rank_order():
    builder = dsa_cp.AscendDSADCPMetadataBuilder.__new__(dsa_cp.AscendDSADCPMetadataBuilder)
    builder.dcp_size = 3
    builder.dcp_local_block_table_cols = 2
    builder.dcp_collective_rank_order = torch.tensor([2, 0, 1], dtype=torch.int32)
    req = _make_req_metadata([32, 32], [[9, 3, 99], [3, 7, 99]])

    builder._populate_c4_prefill_gather_metadata(req)

    assert req.dcp_prefill_kv_gather_block_ids.tolist() == [3, 7, 9]
    assert req.dcp_prefill_kv_gather_block_table.tolist() == [[8, 2, 5, 6, 0, 3], [6, 0, 3, 7, 1, 4]]
    # Values encode collective rank and physical block ID, independently of the compact table.
    gathered_kv = torch.tensor([3, 7, 9, 103, 107, 109, 203, 207, 209])
    assert gathered_kv[req.dcp_prefill_kv_gather_block_table].tolist() == [
        [209, 9, 109, 203, 3, 103],
        [203, 3, 103, 207, 7, 107],
    ]
    assert req.block_table.tolist() == [[9, 3, 99], [3, 7, 99]]


def test_decode_query_gather_restores_token_major_layout_after_collective_completion(monkeypatch):
    # The collective concatenates the two rank-local head-major Q tensors.
    gathered_q = torch.tensor([[[1], [2]], [[11], [12]], [[101], [102]], [[111], [112]]])
    completion_event = object()
    events = []
    work = MagicMock()
    work.wait.side_effect = lambda: events.append("work_wait")

    def wait_event(event):
        assert event is completion_event
        assert events == ["work_wait"]
        events.append("event_wait")

    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(wait_event=wait_event))
    context = dsa_cp.DSAQueryGatherContext(gathered_q, completion_event, work)

    actual = dsa_cp.AscendDSADCPImpl._finish_attention_query(context)

    assert actual.tolist() == [[[1], [11], [101], [111]], [[2], [12], [102], [112]]]
    assert actual.is_contiguous()
    assert events == ["work_wait", "event_wait"]


def test_c4_local_attention_passes_local_addresses_and_masks_empty_rows(monkeypatch):
    monkeypatch.setattr(dsa_cp, "HAS_TRITON", False)
    req = _make_req_metadata([16, 16], [[5, 7], [9, 11]])
    req.dcp_local_block_table = torch.tensor([[5], [9]], dtype=torch.int32)
    req.dcp_local_seqused_kv = torch.tensor([8, 8], dtype=torch.int32)
    req.dcp_local_sas_metadata = torch.ones(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    req.dcp_cmp_only_sinks = torch.full((4,), dsa_cp.DSA_CMP_ONLY_SINK)
    topk = torch.tensor([[[0, 1, -1, 8]], [[2, 3, 6, 7]]], dtype=torch.int32)
    req.dcp_local_cmp_sparse_indices = torch.empty_like(topk)
    req.dcp_local_remap_chunk_out = torch.empty((2, 4), dtype=torch.int32)
    req.dcp_local_remap_chunk_count = torch.empty((2, 1), dtype=torch.int32)
    swa_req = _make_req_metadata([16, 16], [[3, 4], [6, 8]])
    q = torch.ones((2, 4, 3))
    c4_kv = torch.ones((12, 2, 3))
    swa_kv = torch.ones((9, 2, 3))
    events = []

    def attention_op(query, **kwargs):
        assert events == ["metadata_ready"]
        assert query is q
        assert kwargs["cmp_kv"] is c4_kv
        assert kwargs["ori_kv"] is swa_kv
        assert kwargs["cmp_block_table"] is req.dcp_local_block_table
        assert kwargs["ori_block_table"] is swa_req.block_table
        assert kwargs["seqused_kv"] is req.dcp_local_seqused_kv
        assert kwargs["metadata"] is req.dcp_local_sas_metadata
        assert kwargs["sinks"] is req.dcp_cmp_only_sinks
        assert kwargs["cmp_sparse_indices"][:, 0].tolist() == [[-1, -1, -1, -1], [0, 1, 2, 3]]
        assert kwargs["return_softmax_lse"] is True
        assert kwargs["cmp_ratio"] == 4
        return torch.full_like(query, 7), torch.full((2, 4, 1), 2.0)

    def wait(stage, group_id):
        assert (stage, group_id) == (DeviceMetadataStage.ATTENTION, id(req.dcp_local_sas_metadata))
        events.append("metadata_ready")

    plan = _make_plan(get_dsa_sparse_attn_op=attention_op)
    monkeypatch.setattr(dsa_cp, "get_dsa_attn_kv_plan", lambda _: plan)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", lambda _: plan)
    monkeypatch.setattr(dsa_cp, "wait_for_device_metadata", wait)

    output, lse = dsa_cp.forward_dsa_c4_local_attention(q, c4_kv, swa_kv, topk, req, swa_req, _make_config(), 2, 1, 0.5)

    assert torch.count_nonzero(output[0]).item() == 0
    assert torch.isneginf(lse[0]).all()
    assert output[1].eq(7).all()
    assert lse[1].eq(2).all()


def test_c4_decode_exchanges_partial_attention_and_merges_swa_once(monkeypatch):
    impl = dsa_cp.AscendDSADCPImpl.__new__(dsa_cp.AscendDSADCPImpl)
    impl.compress_ratio, impl.dcp_size, impl.dcp_rank = 4, 2, 1
    impl.n_local_heads, impl.head_dim, impl.softmax_scale = 2, 3, 0.5
    impl.vllm_config = _make_config()
    impl.dcp_group = SimpleNamespace(unique_name="test_dcp_group")
    req = _make_req_metadata([16, 20], [[3, 7], [5, 9]])
    swa_req = _make_req_metadata([16, 20], [[11, 13], [15, 17]])
    layer_metadata = AscendDSALayerMetadata(
        AscendDSAMetadata(2, 2, 2, 0, req_metadata=req),
        AscendDSAMetadata(2, 2, 2, 0, req_metadata=swa_req),
    )
    q = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    global_q = torch.cat((q, q + 100), dim=1)
    context = dsa_cp.DSADCPAttentionContext(
        query_gather=dsa_cp.DSAQueryGatherContext(global_q.transpose(0, 1).contiguous(), object(), None)
    )
    c4_kv, swa_kv = torch.zeros((10, 2, 3)), torch.zeros((20, 2, 3))
    topk = torch.zeros((2, 1, 8), dtype=torch.int32)
    c4_output, c4_lse = torch.ones_like(global_q), torch.ones((2, 4, 1))
    swa_output, swa_lse = torch.zeros_like(q), torch.zeros((2, 2, 1))
    exchanged = torch.ones((2, 4, 4))
    events = []
    main_stream, comm_stream = MagicMock(), MagicMock()
    c4_ready, c4_done = object(), object()
    main_stream.record_event.return_value = c4_ready
    comm_stream.record_event.return_value = c4_done

    def wait_c4_ready(event):
        assert event is c4_ready
        assert events == ["c4"]

    def wait_on_main_stream(event):
        if event is c4_done:
            assert events == ["c4", "exchange", "swa"]

    comm_stream.wait_event.side_effect = wait_c4_ready
    main_stream.wait_event.side_effect = wait_on_main_stream

    def c4_attention(query, cache, placeholder, indices, c4_metadata, swa_metadata, config, size, rank, scale):
        events.append("c4")
        torch.testing.assert_close(query, global_q)
        assert (cache is c4_kv) and (placeholder is swa_kv) and (indices is topk)
        assert (c4_metadata is req) and (swa_metadata is swa_req) and (config is impl.vllm_config)
        assert (size, rank, scale) == (2, 1, 0.5)
        return c4_output, c4_lse

    def exchange(output, lse, size, scatter_dim, group, *, defer_combine):
        assert events == ["c4"]
        events.append("exchange")
        assert (output is c4_output) and (lse is c4_lse)
        assert (size, scatter_dim, group, defer_combine) == (2, 1, "test_dcp_group", True)
        return exchanged

    def swa_attention(query, cache, metadata):
        assert events == ["c4", "exchange"]
        events.append("swa")
        assert (query is q) and (cache is swa_kv) and (metadata is swa_req)
        return swa_output, swa_lse

    def combine(received, head_dim, *, scatter_dim, local_output, local_lse):
        assert events == ["c4", "exchange", "swa"]
        main_stream.wait_event.assert_called_with(c4_done)
        events.append("combine")
        assert received is exchanged
        assert (head_dim, scatter_dim) == (3, 1)
        assert (local_output is swa_output) and (local_lse is swa_lse)
        return local_output

    monkeypatch.setattr(dsa_cp, "forward_dsa_c4_local_attention", c4_attention)
    monkeypatch.setattr(torch.ops.vllm, "sfa_dcp_a2a_fused", exchange, raising=False)
    monkeypatch.setattr(dsa_cp, "fused_sfa_dcp_lse_combine", combine)
    monkeypatch.setattr(impl, "_forward_swa_local_attention", swa_attention)
    monkeypatch.setattr(impl, "_log_dcp_memory_once", lambda *_: None)
    monkeypatch.setattr(impl, "_log_dcp_path_once", lambda *_: None)
    monkeypatch.setattr(dsa_cp, "record_attention_compute_start", lambda: None)
    monkeypatch.setattr(dsa_cp, "_dsv4_dcp_comm_stream", lambda: comm_stream)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: main_stream)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    # CPU tensors cannot record NPU streams; only this device boundary is stubbed.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)

    result = impl._execute_attention(q, c4_kv, swa_kv, topk, layer_metadata, context)

    assert result is swa_output
    assert events == ["c4", "exchange", "swa", "combine"]
    comm_stream.wait_event.assert_called_once_with(c4_ready)
    assert main_stream.wait_event.call_args_list[-1].args == (c4_done,)


@pytest.mark.parametrize("cache_write_hook_runs", [False, True])
def test_prefill_attention_gathers_once_and_does_not_replace_persistent_metadata(cache_write_hook_runs, monkeypatch):
    impl = dsa_cp.AscendDSADCPImpl.__new__(dsa_cp.AscendDSADCPImpl)
    impl.compress_ratio = 4
    impl.dcp_size = 2
    impl.vllm_config = _make_config()
    impl.window_size = 8
    impl.attn_sink = torch.zeros(2)
    impl.softmax_scale = 0.5
    req = _make_req_metadata([16, 12], [[9], [3]])
    req.dcp_prefill_kv_gather_block_table = torch.tensor([[2, 3], [0, 1]], dtype=torch.int32)
    req.cu_cmp_seqlen_list = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = AscendDSAMetadata(2, 1, 1, 1, req_metadata=req)
    swa_req = _make_req_metadata([16, 12], [[5, 6], [7, 8]])
    layer_metadata = AscendDSALayerMetadata(
        metadata,
        AscendDSAMetadata(2, 1, 1, 1, req_metadata=swa_req),
        compressor=SimpleNamespace(),
        indexer=SimpleNamespace(),
    )
    q = torch.ones((2, 2, 3))
    persistent_kv = torch.zeros((10, 2, 3))
    gathered_kv = torch.ones((4, 2, 3))
    swa_kv = torch.ones((9, 2, 3))
    topk = torch.zeros((2, 1, 8), dtype=torch.int32)
    context = dsa_cp.DSADCPAttentionContext()
    events = []
    work = MagicMock()
    work.wait.side_effect = lambda: events.append("work_wait")
    completion_event = object()
    gather_context = dsa_cp.DSAPrefillKVGatherContext(gathered_kv, completion_event, work)
    expected_events = ["compress", "scatter"] if cache_write_hook_runs else []
    expected_events += ["gather", "work_wait", "event_wait"]

    def start_gather(cache, request):
        assert cache is persistent_kv
        assert request is req
        assert cache[0, 0].eq(7 if cache_write_hook_runs else 0).all()
        events.append("gather")
        return gather_context

    def wait_event(event):
        assert event is completion_event
        events.append("event_wait")

    def attention_op(query, **kwargs):
        assert events == expected_events
        assert query is q
        assert kwargs["cmp_kv"] is gathered_kv
        assert kwargs["cmp_block_table"] is req.dcp_prefill_kv_gather_block_table
        assert kwargs["ori_block_table"] is swa_req.block_table
        assert kwargs["cmp_sparse_indices"] is topk
        assert kwargs["seqused_kv"] is req.seq_lens
        assert kwargs["metadata"] is req.sas_metadata
        events.append("attention")
        return query, None

    plan = _make_plan(get_dsa_sparse_attn_op=attention_op)

    def compressor(**kwargs):
        events.append("compress")
        return torch.full((1, 1, 3), 7.0), torch.tensor([[0, 0]], dtype=torch.int32)

    def scatter(cache, compressed_kv, slot_mapping):
        events.append("scatter")
        cache[slot_mapping[:, 0], slot_mapping[:, 1]] = compressed_kv[:, 0]

    def indexer(**kwargs):
        overlap_plan = kwargs["overlap_plan"]
        compressed_kv, slot_mapping = overlap_plan.compute_attention_compressed_kv()
        overlap_plan.scatter_attention_compressed_kv(compressed_kv, slot_mapping)
        return topk

    impl.compressor = compressor
    impl.indexer = indexer
    impl.multistream_dsv4_dsa_overlap = False
    plan.dsa_kv_compress_scatter.side_effect = scatter
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", lambda _: plan)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.wait_for_device_metadata", lambda *_: None)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.record_attention_compute_start", lambda: None)
    monkeypatch.setattr(impl, "_start_prefill_kv_gather", start_gather)
    monkeypatch.setattr(impl, "_log_dcp_path_once", lambda *_: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(wait_event=wait_event))

    if cache_write_hook_runs:
        impl._maybe_update_compressed_caches_and_select_topk(
            "layer",
            torch.ones((2, 3)),
            torch.ones((2, 3)),
            (persistent_kv,),
            layer_metadata,
            None,
            persistent_kv,
            torch.empty(0),
            query_context=context,
        )
        impl._compressed_cache_write_complete(persistent_kv, layer_metadata, context)
    impl._execute_attention(q, persistent_kv, swa_kv, topk, layer_metadata, context)

    assert events == [*expected_events, "attention"]
    assert layer_metadata.attention is metadata
    assert metadata.req_metadata is req
    assert req.block_table.tolist() == [[9], [3]]


@pytest.mark.parametrize(("compress_ratio", "dcp_size"), [(4, 1), (1, 2), (128, 2)])
def test_unsharded_layers_keep_original_attention_contract(compress_ratio, dcp_size, monkeypatch):
    impl = dsa_cp.AscendDSADCPImpl.__new__(dsa_cp.AscendDSADCPImpl)
    impl.compress_ratio = compress_ratio
    impl.dcp_size = dcp_size
    impl.vllm_config = _make_config()
    impl.window_size = 8
    impl.attn_sink = torch.zeros(2)
    impl.softmax_scale = 0.5
    req = _make_req_metadata([16], [[3, 7]])
    swa_req = _make_req_metadata([16], [[9, 11]])
    swa_req.vision_swa_indices = torch.tensor([[[1, 2]]], dtype=torch.int32)
    metadata = AscendDSAMetadata(1, 1, 1, 0, req_metadata=req)
    swa_metadata = AscendDSAMetadata(1, 1, 1, 0, req_metadata=swa_req)
    layer_metadata = AscendDSALayerMetadata(metadata if compress_ratio > 1 else None, swa_metadata)
    q = torch.ones((1, 2, 3))
    c4_kv = torch.ones((12, 2, 3))
    swa_kv = torch.ones((12, 2, 3))
    topk = torch.zeros((1, 1, 8), dtype=torch.int32)
    calls = []

    def attention_op(query, **kwargs):
        calls.append(kwargs)
        assert query is q
        assert kwargs["ori_kv"] is swa_kv
        assert kwargs["ori_sparse_indices"] is swa_req.vision_swa_indices
        assert kwargs["sinks"] is impl.attn_sink
        assert kwargs["cmp_ratio"] == compress_ratio
        if compress_ratio > 1:
            assert kwargs["cmp_kv"] is c4_kv
            assert kwargs["cmp_block_table"] is req.block_table
            assert kwargs["metadata"] is req.sas_metadata
        else:
            assert "cmp_kv" not in kwargs
            assert kwargs["metadata"] is swa_req.sas_metadata
        if compress_ratio == 4:
            assert kwargs["cmp_sparse_indices"] is topk
        else:
            assert "cmp_sparse_indices" not in kwargs
        return query, None

    plan = _make_plan(get_dsa_sparse_attn_op=attention_op)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", lambda _: plan)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.wait_for_device_metadata", lambda *_: None)
    monkeypatch.setattr("vllm_ascend.attention.dsa_v1.record_attention_compute_start", lambda: None)

    assert impl._start_attention_query(q, layer_metadata) is None
    impl._execute_attention(q, c4_kv, swa_kv, topk, layer_metadata)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("unsupported_feature", "message"),
    [
        ("sparse_flash_mla", "supports the Ascend A2 SCFA path only"),
        ("kv_compress_epilog", "supports the Ascend A2 SCFA path only"),
        ("graph", "requires eager execution"),
        ("prefix", "does not support prefix caching"),
        ("speculative", "does not support speculative decoding"),
    ],
)
def test_phase_one_rejects_unsupported_execution_modes(unsupported_feature, message, monkeypatch):
    config = _make_config()
    config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    config.cache_config = SimpleNamespace(enable_prefix_caching=False)
    config.speculative_config = None
    plan = _make_plan()
    if unsupported_feature in ("sparse_flash_mla", "kv_compress_epilog"):
        setattr(plan, f"uses_{unsupported_feature}", True)
    elif unsupported_feature == "graph":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    elif unsupported_feature == "prefix":
        config.cache_config.enable_prefix_caching = True
    else:
        config.speculative_config = SimpleNamespace(num_speculative_tokens=1)

    def init_base(self):
        self.compress_ratio = 4
        self.vllm_config = config

    monkeypatch.setattr(AscendDSAImpl, "__init__", init_base)
    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.common_cp.get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=object()),
    )
    monkeypatch.setattr(dsa_cp, "get_dsa_attn_kv_plan", lambda _: plan)

    with pytest.raises(NotImplementedError, match=message):
        dsa_cp.AscendDSADCPImpl()
