# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadata,
    AscendAttentionDCPMetadataBuilder,
    AscendMetadataForDecode,
    AscendMetadataForPrefill,
)
from vllm_ascend.attention.context_parallel.common_cp import _update_out_and_lse


def test_gqa_dcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionDCPImpl, AscendAttentionBackendImpl)
    assert issubclass(
        AscendAttentionDCPMetadataBuilder,
        AscendAttentionMetadataBuilder,
    )
    assert AscendAttentionDCPMetadataBuilder.metadata_cls is (AscendAttentionDCPMetadata)
    assert not hasattr(AscendMetadata(), "decode")
    assert not hasattr(AscendMetadata(), "prefill")


def test_gqa_dcp_builder_consumes_pcp_context() -> None:
    assert AscendAttentionDCPMetadataBuilder.consumes_pcp_context


def test_gqa_tp_only_dcp_still_gathers_heads() -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.pcp_group = SimpleNamespace(world_size=1)
    query = torch.arange(8).view(1, 4, 2)
    gathered = torch.cat((query, query + 100), dim=1)
    impl._dcp_all_gather = Mock(return_value=gathered)
    (actual,) = impl._dcp_all_gather_fragments(query, dim=1)
    torch.testing.assert_close(actual, gathered)
    impl._dcp_all_gather.assert_called_once()


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_gqa_pcp_dcp_preserves_tp_local_heads_and_merges_pcp_shards(kv_heads) -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.num_kv_heads = kv_heads
    impl.dcp_size = 2
    impl.pcp_group = SimpleNamespace(world_size=2, unique_name="pcp-kv-shards")
    impl.dcp_group = SimpleNamespace(unique_name="dcp-kv-shards")
    impl.tp_group = SimpleNamespace(world_size=8, unique_name="tp-heads", all_gather=Mock())
    impl._dcp_all_gather = Mock(side_effect=AssertionError("Identical TP heads must not be gathered"))
    query = torch.randn(3, kv_heads * 4, 2)
    (gathered,) = impl._dcp_all_gather_fragments(query, dim=1)
    assert gathered is query
    impl._dcp_all_gather.assert_not_called()
    impl.tp_group.all_gather.assert_not_called()
    lse = torch.zeros(*query.shape[:2], 1)
    with patch("torch.ops.vllm.dcp_a2a_fused", return_value=query) as combine:
        actual = impl._merge_dcp_attention_output(query, lse)
    assert actual is query
    combine.assert_called_once_with(query, lse, 1, 1, "tp-heads", "pcp-kv-shards", defer_combine=False)


def test_gqa_dcp_capture_forwards_pcp_context() -> None:
    builder = AscendAttentionDCPMetadataBuilder.__new__(AscendAttentionDCPMetadataBuilder)
    builder.build = Mock(return_value="metadata")
    common = object()
    context = object()

    assert builder.build_for_cudagraph_capture(common, pcp_context=context, pcp_cache_group_idx=2) == "metadata"
    builder.build.assert_called_once_with(
        common_prefix_len=0, common_attn_metadata=common, pcp_context=context, pcp_cache_group_idx=2
    )


@pytest.mark.parametrize("rank", [0, 1], ids=["head-tail", "middle"])
@pytest.mark.parametrize("history", [0, 101], ids=["first-chunk", "uneven-history"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunked_prefill_enabled", [False, True])
def test_gqa_pcp_dcp_metadata_separates_current_prefix_and_dcp_history(
    history, chunked_prefill_enabled, causal, rank
) -> None:
    builder = AscendAttentionDCPMetadataBuilder.__new__(AscendAttentionDCPMetadataBuilder)
    builder.pcp_group = SimpleNamespace(world_size=2, rank_in_group=rank)
    builder.pcp_enabled = True
    builder.chunked_prefill_enabled = chunked_prefill_enabled
    builder.dcp_size = 2
    builder.dcp_rank = rank
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))
    builder._pcp_cache_group_idx = 0
    global_table = torch.tensor([[3]], dtype=torch.int32)
    builder._pcp_context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=1,
            is_prefilling_np=np.array([True]),
            num_computed_tokens_np=np.array([history]),
            query_start_loc_np=np.array([0, 8]),
        ),
        local_to_global_req_indices=(0, 0) if rank == 0 else (0,),
        hidden_restore_idx=torch.tensor([0, 1, 4, 5, 6, 7, 2, 3]),
        padded_gather_idx=torch.tensor([0, 1, 6, 7, 2, 3, 4, 5]),
        global_block_tables=(global_table,),
    )
    common = SimpleNamespace(slot_mapping=torch.zeros(8, dtype=torch.int64), num_actual_tokens=4, causal=causal)
    query_lens = [2, 2] if rank == 0 else [4]
    current_ends = [2, 8] if rank == 0 else [6]
    prefill = builder._build_pcp_prefill_metadata(
        common,
        global_table.expand(len(query_lens), -1),
        torch.tensor(query_lens),
        torch.tensor([history + end for end in current_ends]),
        0,
        len(query_lens),
    )
    ordered = [0, 1, 4, 5, 6, 7, 2, 3]
    expected_indices = ([0, 1] + ordered if rank == 0 else ordered[:6]) if causal else ordered * len(query_lens)
    expected_kv_ends = ([2, 10] if rank == 0 else [6]) if causal else ([8, 16] if rank == 0 else [8])
    assert prefill.pcp_current_kv_indices.tolist() == expected_indices
    assert prefill.pcp_actual_seq_lengths_kv == expected_kv_ends
    assert prefill.actual_seq_lengths_q == ([2, 4] if rank == 0 else [4])
    if history:
        assert prefill.block_tables.shape[0] == 1
        expected_history = [sum(position % 2 == rank for position in range(history))]
        assert prefill.chunked_context.local_context_lens.tolist() == expected_history
        assert prefill.chunked_context.actual_seq_lengths_kv == expected_history
        assert prefill.chunked_context.actual_chunk_seq_lengths == [8]
        assert prefill.chunked_context.pcp_query_restore_idx.tolist() == [0, 1, 4, 5, 6, 7, 2, 3]
        assert prefill.chunked_context.pcp_local_query_indices.tolist() == ([0, 1, 6, 7] if rank == 0 else [2, 3, 4, 5])
        impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
        impl.pcp_enabled = True
        gathered_query = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5]).view(8, 1, 1)
        impl.pcp_group = SimpleNamespace(all_gather=Mock(return_value=gathered_query))
        impl._dcp_all_gather_fragments = Mock(side_effect=lambda query, dim: (query,))
        restored_query = impl._prefill_query_all_gather(
            SimpleNamespace(prefill=prefill), gathered_query[rank * 4 : (rank + 1) * 4]
        )
        torch.testing.assert_close(restored_query, torch.arange(8).view(8, 1, 1))
        local_query = gathered_query[rank * 4 : (rank + 1) * 4]
        impl.pcp_group.all_gather.assert_called_once()
        torch.testing.assert_close(impl.pcp_group.all_gather.call_args.args[0], local_query)
        assert impl.pcp_group.all_gather.call_args.kwargs["dim"] == 0
    else:
        assert prefill.chunked_context is None


def test_gqa_first_prefill_uses_current_kv_prefixes_without_cache_tables() -> None:
    builder = AscendAttentionDCPMetadataBuilder.__new__(AscendAttentionDCPMetadataBuilder)
    builder.pcp_enabled = True
    builder.pcp_group = SimpleNamespace(world_size=2)
    builder.chunked_prefill_enabled = True
    builder._pcp_context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=2,
            is_prefilling_np=np.array([False, True]),
            num_computed_tokens_np=np.array([10, 0]),
            query_start_loc_np=np.array([0, 1, 9]),
        ),
        local_to_global_req_indices=(0, 1, 1),
        hidden_restore_idx=torch.tensor([0, 1, 2, 6, 7, 8, 9, 3, 4]),
    )
    query_lens = torch.tensor([1, 2, 2], dtype=torch.int32)
    seq_lens = torch.tensor([11, 2, 8], dtype=torch.int32)
    common = SimpleNamespace(slot_mapping=torch.zeros(10, dtype=torch.int64), dcp_local_seq_lens_cpu=None, causal=True)
    # Avoid decode construction here; directly verify the mixed-batch prefix map.
    prefill = builder._build_pcp_prefill_metadata(
        common, torch.zeros(3, 1, dtype=torch.int32), query_lens, seq_lens, 1, 2
    )
    indices = prefill.pcp_current_kv_indices
    assert indices.tolist() == [1, 2, 1, 2, 5, 6, 7, 8, 3, 4]
    cache_order_tokens = torch.tensor([-1, 0, 1, 6, 7, 2, 3, 4, 5])
    assert cache_order_tokens[indices].tolist() == [0, 1, 0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("pcp_enabled", [False, True], ids=["dcp-only", "pcp-dcp"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_gqa_current_attention_routes_local_q_and_ordered_kv(pcp_enabled, causal) -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.pcp_enabled = pcp_enabled
    impl.num_heads, impl.num_kv_heads = 4, 1
    impl.scale = 0.5
    ordered = [1, 2, 5, 6, 7, 8, 3, 4]
    kv_rows = [1, 2] + ordered if causal else ordered * 2
    kv_ends = [2, 10] if causal else [8, 16]
    prefill = AscendMetadataForPrefill(
        actual_seq_lengths_q=[2, 4],
        pcp_current_kv_indices=torch.tensor(kv_rows),
        pcp_actual_seq_lengths_kv=kv_ends,
    )
    metadata = AscendAttentionDCPMetadata(
        num_decode_tokens=1,
        num_actual_tokens=5,
        causal=causal,
        attn_mask=torch.ones(4, 4, dtype=torch.bool),
        prefill=prefill,
    )
    query = torch.arange(7 * 4 * 2).float().view(7, 4, 2)
    key = torch.arange(9 * 2).float().view(9, 1, 2)
    value = key + 1000
    expected_query = query[1:5]
    lse = torch.zeros(4, 4, 1)
    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with (
        patch(module + ".record_attention_compute_start"),
        patch(module + ".torch_npu.npu_fused_infer_attention_score", return_value=(expected_query, lse)) as attention,
    ):
        actual, actual_lse = impl._forward_prefill_current_kv(query, key, value, metadata)
    attention.assert_called_once()
    sent_query, sent_key, sent_value = attention.call_args.args
    torch.testing.assert_close(sent_query, expected_query)
    expected_rows = kv_rows if pcp_enabled else [1, 2, 3, 4]
    torch.testing.assert_close(sent_key, key[expected_rows])
    torch.testing.assert_close(sent_value, value[expected_rows])
    torch.testing.assert_close(actual, expected_query)
    assert actual_lse is lse
    assert attention.call_args.kwargs["actual_seq_lengths"] is prefill.actual_seq_lengths_q
    assert attention.call_args.kwargs["actual_seq_lengths_kv"] == (kv_ends if pcp_enabled else [2, 4])
    assert attention.call_args.kwargs["sparse_mode"] == (3 if causal else 0)
    assert attention.call_args.kwargs["atten_mask"] is (metadata.attn_mask if causal else None)


@pytest.mark.parametrize("has_history", [False, True])
@pytest.mark.parametrize("decode_tokens", [0, 1])
@pytest.mark.parametrize("padding", [0, 2])
def test_gqa_pcp_dcp_merges_history_only_for_local_query_fragments(has_history, decode_tokens, padding) -> None:
    from vllm_ascend.attention.context_parallel import attention_cp

    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.head_size = 2
    impl.pcp_enabled = True
    impl._forward_decode_dcp = Mock(return_value=torch.full((decode_tokens, 1, 2), 9.0))
    current_output = torch.ones(4, 1, 2)
    current_lse = torch.zeros(4, 1, 1)
    impl._forward_prefill_current_kv = Mock(return_value=(current_output, current_lse))
    impl._prefill_query_all_gather = Mock(return_value=torch.zeros(8, 1, 2))
    impl._compute_prefill_context = Mock(return_value=(torch.zeros(8, 1, 2), torch.zeros(8, 1, 1)))
    recv = torch.arange(2 * 8 * 3).view(2, 1, 8, 3)
    impl._merge_dcp_attention_output = Mock(return_value=recv)
    prefill = AscendMetadataForPrefill()
    if has_history:
        prefill.chunked_context = AscendMetadataForPrefill.ChunkedContextMetadata(
            actual_chunk_seq_lengths=[8],
            actual_seq_lengths_kv=[2],
            starts=torch.zeros(1, dtype=torch.int32),
            pcp_local_query_indices=torch.tensor([0, 1, 6, 7]),
        )
    metadata = AscendAttentionDCPMetadata(
        num_decodes=decode_tokens,
        num_prefills=2,
        num_decode_tokens=decode_tokens,
        num_actual_tokens=decode_tokens + 4,
        pcp_local_num_input_tokens=decode_tokens + 4 + padding,
        prefill=prefill,
        decode=AscendMetadataForDecode(actual_seq_lengths_q=list(range(1, decode_tokens + 1)))
        if decode_tokens
        else None,
    )
    query = torch.arange((decode_tokens + 4 + padding) * 2).float().reshape(-1, 1, 2)
    output = torch.full_like(query, -99)
    stream = Mock()
    with (
        patch.object(attention_cp, "cp_chunkedprefill_comm_stream", return_value=stream),
        patch.object(torch.npu, "current_stream", return_value=stream),
        patch.object(attention_cp.torch_npu.npu, "stream", side_effect=lambda *_: nullcontext()),
        patch.object(attention_cp, "fused_dcp_lse_combine", return_value=current_output) as combine,
    ):
        actual = impl.forward_impl(query, query, query, (), metadata, output)
    assert actual is output
    torch.testing.assert_close(actual[decode_tokens : decode_tokens + 4], current_output)
    if decode_tokens:
        torch.testing.assert_close(actual[:decode_tokens], torch.full((decode_tokens, 1, 2), 9.0))
        impl._forward_decode_dcp.assert_called_once()
        torch.testing.assert_close(impl._forward_decode_dcp.call_args.args[0], query[:decode_tokens])
    else:
        impl._forward_decode_dcp.assert_not_called()
    if padding:
        torch.testing.assert_close(actual[-padding:], torch.full((padding, 1, 2), -99.0))
    if has_history:
        torch.testing.assert_close(impl._prefill_query_all_gather.call_args.args[1], query[decode_tokens:])
        selected = recv[:, :, [0, 1, 6, 7]]
        torch.testing.assert_close(combine.call_args.args[0], selected)
        assert combine.call_args.kwargs["local_output"] is current_output
        assert combine.call_args.kwargs["local_lse"] is current_lse
        impl._merge_dcp_attention_output.assert_called_once()
    else:
        impl._prefill_query_all_gather.assert_not_called()
        impl._compute_prefill_context.assert_not_called()
        impl._merge_dcp_attention_output.assert_not_called()
        combine.assert_not_called()


def test_gqa_chunked_prefill_uses_shared_dcp_merge_for_pcp_overlap() -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.pcp_enabled = False
    impl.num_heads = 4
    impl.num_kv_heads = 1
    impl.head_size = 2
    impl.scale = 0.5

    chunked = AscendMetadataForPrefill.ChunkedContextMetadata(
        actual_chunk_seq_lengths=[3],
        actual_seq_lengths_kv=[5],
        starts=torch.zeros(1, dtype=torch.int32),
    )
    metadata = AscendAttentionDCPMetadata(
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        num_actual_tokens=3,
        causal=True,
        attn_mask=torch.ones(3, 3, dtype=torch.bool),
        prefill=AscendMetadataForPrefill(
            chunked_context=chunked,
            actual_seq_lengths_q=[3],
        ),
    )
    query = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
    key = value = torch.arange(6, dtype=torch.float32).view(3, 1, 2)
    output = torch.zeros_like(query)
    current_output = torch.full_like(query, 1)
    current_lse = torch.zeros(3, 4, 1)
    history_output = torch.full_like(query, 2)
    history_lse = torch.ones(3, 4, 1)
    packed_history = object()
    merged = torch.full_like(query, 3)
    impl._prefill_query_all_gather = Mock(return_value=query)
    impl._compute_prefill_context = Mock(return_value=(history_output, history_lse))
    impl._merge_dcp_attention_output = Mock(return_value=packed_history)
    stream = Mock()

    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with (
        patch(module + ".cp_chunkedprefill_comm_stream", return_value=stream),
        patch(module + ".torch.npu.current_stream", return_value=stream),
        patch(module + ".torch_npu.npu.stream", return_value=nullcontext()),
        patch(module + ".record_attention_compute_start"),
        patch(
            module + ".torch_npu.npu_fused_infer_attention_score",
            return_value=(current_output, current_lse),
        ),
        patch(module + ".fused_dcp_lse_combine", return_value=merged) as combine,
    ):
        actual = impl.forward_impl(query, key, value, (object(), object()), metadata, output)

    impl._merge_dcp_attention_output.assert_called_once_with(
        history_output,
        history_lse,
        defer_combine=True,
    )
    combine.assert_called_once()
    assert combine.call_args.args == (packed_history, 2)
    assert combine.call_args.kwargs["scatter_dim"] == 1
    torch.testing.assert_close(combine.call_args.kwargs["local_output"], current_output)
    torch.testing.assert_close(combine.call_args.kwargs["local_lse"], current_lse)
    torch.testing.assert_close(actual, merged)


@pytest.mark.parametrize("size,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("interleave", [1, 8])
@pytest.mark.parametrize("device_context_offset", [0, 3])
def test_dcp_chunked_prefill_keeps_host_parameters_and_device_history_separate(
    size, rank, interleave, device_context_offset
):
    builder = object.__new__(AscendAttentionDCPMetadataBuilder)
    builder.chunked_prefill_enabled = True
    builder.pcp_enabled = False
    builder.dcp_size, builder.dcp_rank = size, rank
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave))
    # Include a decode request and a prefill request with no cached history.
    seq_lens = torch.tensor([11, 20, 5, 14], dtype=torch.int32)
    query_lens = torch.tensor([1, 3, 5, 6], dtype=torch.int32)
    device_seq_lens = seq_lens.clone()
    device_seq_lens[[1, 3]] += device_context_offset
    common = SimpleNamespace(
        seq_lens=device_seq_lens,
        query_start_loc=torch.cat([torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0)]),
        dcp_local_seq_lens_cpu=torch.tensor(
            [
                sum((position // interleave) % size == rank for position in range(length))
                for length in seq_lens.tolist()
            ],
            dtype=torch.int32,
        ),
    )
    tensor_to = torch.Tensor.to

    def reject_all_rank_transfer(tensor, *args, **kwargs):
        assert tensor.ndim != 2, "All-rank history lengths must stay on CPU"
        return tensor_to(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", reject_all_rank_transfer):
        metadata = builder._build_backend_metadata(
            common,
            block_table=torch.zeros(4, 2, dtype=torch.int32),
            query_lens=query_lens,
            seq_lens=seq_lens,
            num_decodes=1,
            num_prefills=3,
        )
    assert metadata["prefill"].actual_seq_lengths_q == [3, 8, 14]
    chunked = metadata["prefill"].chunked_context
    expected = [sum((position // interleave) % size == rank for position in range(length)) for length in [17, 0, 8]]
    expected_device = [
        sum((position // interleave) % size == rank for position in range(length))
        for length in [17 + device_context_offset, 0, 8 + device_context_offset]
    ]
    torch.testing.assert_close(chunked.local_context_lens, torch.tensor(expected_device, dtype=torch.int32))
    assert chunked.actual_seq_lengths_kv == np.cumsum(expected).tolist()
    assert chunked.local_total_toks == sum(expected)
    assert chunked.starts.tolist() == [0, 0, 0]


@pytest.mark.parametrize("rank", [0, 1])
def test_dcp_decode_builder_consumes_producer_local_lengths(rank):
    builder = object.__new__(AscendAttentionDCPMetadataBuilder)
    builder.dcp_size, builder.dcp_rank = 2, rank
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=8))
    # The producer's local bound can differ from a recomputed global bound.
    local_lengths = torch.tensor([101, 202, 303], dtype=torch.int32)
    common = SimpleNamespace(dcp_local_seq_lens_cpu=local_lengths)
    result = builder._build_backend_metadata(
        common,
        block_table=torch.zeros(3, 2, dtype=torch.int32),
        query_lens=torch.tensor([3, 5, 1], dtype=torch.int32),
        seq_lens=torch.tensor([13, 23, 31], dtype=torch.int32),
        num_decodes=2,
        num_prefills=0,
    )
    np.testing.assert_array_equal(result["decode"].num_computed_tokens_of_dcp[:, rank], [101, 202])
    expected = [sum((position // 8) % 2 == rank for position in range(length)) for length in [10, 18]]
    assert result["decode"].cp_history_seq_len == expected


def test_dcp_decode_metadata_keeps_rank_local_context_lengths() -> None:
    local_context_lens = np.array([[11, 12], [21, 22]], dtype=np.int32)
    block_tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    metadata = AscendMetadataForDecode(
        num_computed_tokens_of_dcp=local_context_lens,
        block_tables=block_tables,
    )

    np.testing.assert_array_equal(metadata.num_computed_tokens_of_dcp[:, 1], [12, 22])
    assert metadata.block_tables is block_tables


def test_dcp_partial_attention_merge_matches_weighted_reference() -> None:
    outputs = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
        ]
    ).reshape(2, 1, 1, 2)
    lse = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32).reshape(2, 1, 1, 1)

    output, merged_lse = _update_out_and_lse(outputs, lse)

    torch.testing.assert_close(output, torch.tensor([[[4.0, 6.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[[np.log(4.0)]]], dtype=torch.float32))


@pytest.mark.parametrize(
    "is_consumer,is_producer,recompute", [(True, False, True), (True, False, False), (False, True, True)]
)
@pytest.mark.parametrize("query_lens", [[1, 1], [3, 3], [3, 5]])
@pytest.mark.parametrize("pcp_enabled", [False, True])
def test_dcp_split_uses_builder_config_without_current_context(
    is_consumer, is_producer, recompute, query_lens, pcp_enabled
):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer, is_kv_producer=is_producer),
    )
    with (
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DCPMetadataBuilderMixin.__init__", return_value=None
        ),
        patch("vllm_ascend.attention.context_parallel.attention_cp.enable_dcp", return_value=True) as dcp,
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.get_pcp_group",
            return_value=SimpleNamespace(world_size=1, rank_in_group=0),
        ) as pcp_group,
    ):
        builder = AscendAttentionDCPMetadataBuilder()
    dcp.assert_called_once_with()
    pcp_group.assert_called_once_with()
    assert builder.pcp_group is pcp_group.return_value
    builder.vllm_config = config
    builder.pcp_enabled = pcp_enabled
    builder.decode_threshold = 3
    builder.speculative_config = None
    query_start_loc = torch.tensor([0, query_lens[0], sum(query_lens)], dtype=torch.int32)
    common = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=max(query_lens),
        num_reqs=2,
        num_actual_tokens=sum(query_lens),
        query_start_loc_cpu=query_start_loc,
        is_prefilling=torch.ones(2, dtype=torch.bool),
    )
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=None),
        patch(
            "vllm_ascend.utils.get_ascend_config",
            return_value=SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute)),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.enable_dcp",
            side_effect=AssertionError("use cached DCP state"),
        ),
    ):
        actual = builder._split_decodes_and_prefills(common)
    num_decodes = (
        sum(q <= 3 for q in query_lens) if not pcp_enabled and is_consumer and not is_producer and recompute else 0
    )
    num_decode_tokens = sum(query_lens[:num_decodes])
    assert actual == (num_decodes, 2 - num_decodes, num_decode_tokens, sum(query_lens) - num_decode_tokens)
