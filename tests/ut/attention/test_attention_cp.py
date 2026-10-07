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
from vllm_ascend.attention.context_parallel.common_cp import (
    _update_out_and_lse,
)


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


def test_gqa_full_dcp_gathers_distinct_kv_heads_across_tp() -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.replicate_full_dcp_kv_heads = True
    impl.kv_head_replication_factor = 2
    gathered_key = torch.arange(16).view(1, 8, 2)
    gathered_value = gathered_key + 100
    impl.tp_group = SimpleNamespace(
        all_gather=Mock(side_effect=(gathered_key, gathered_value)),
    )

    key, value = impl._gather_full_dcp_kv_heads(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )

    torch.testing.assert_close(key, gathered_key[:, ::2])
    torch.testing.assert_close(value, gathered_value[:, ::2])
    assert impl.tp_group.all_gather.call_count == 2


def test_gqa_dcp_capture_forwards_pcp_context() -> None:
    builder = AscendAttentionDCPMetadataBuilder.__new__(AscendAttentionDCPMetadataBuilder)
    builder.build = Mock(return_value="metadata")
    common = object()
    context = object()

    assert (
        builder.build_for_cudagraph_capture(common, pcp_context=context, pcp_cache_group_idx=2) == "metadata"
    )
    builder.build.assert_called_once_with(
        common_prefix_len=0, common_attn_metadata=common, pcp_context=context, pcp_cache_group_idx=2
    )


def test_gqa_pcp_dcp_builder_uses_global_prefill_history_and_local_restore_indices() -> None:
    builder = object.__new__(AscendAttentionDCPMetadataBuilder)
    builder.chunked_prefill_enabled = True
    builder.dcp_size = 2
    builder.dcp_rank = 0
    builder.device = torch.device("cpu")
    builder.pcp_enabled = True
    builder.pcp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1)
    )
    global_table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    builder._pcp_cache_group_idx = 0
    builder._pcp_context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=3,
            is_prefilling_np=np.array([False, True, True]),
            query_start_loc_np=np.array([0, 1, 5, 11]),
            seq_lens_np=np.array([11, 10, 20]),
        ),
        global_block_tables=(global_table,),
        hidden_restore_idx=torch.tensor([0, 1, 2, 7, 8, 3, 4, 5, 9, 10, 11]),
        padded_gather_idx=torch.tensor([0, 1, 2, 5, 6, 7, 0, 3, 4, 8, 9, 10]),
    )
    query_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([11, 8, 17], dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=6,
        query_start_loc=torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        seq_lens=seq_lens,
        dcp_local_seq_lens_cpu=torch.tensor([6], dtype=torch.int32),
    )

    result = builder._build_backend_metadata(
        common,
        block_table=torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.int32),
        query_lens=query_lens,
        seq_lens=seq_lens,
        num_decodes=1,
        num_prefills=2,
    )

    prefill = result["prefill"]
    chunked = prefill.chunked_context
    torch.testing.assert_close(prefill.block_tables, global_table[1:])
    assert prefill.actual_seq_lengths_q.tolist() == [2, 5]
    assert prefill.pcp_actual_seq_lengths_q == [4, 10]
    assert prefill.pcp_local_num_input_tokens == 6
    assert prefill.pcp_local_num_decode_tokens == 1
    assert prefill.pcp_global_num_decode_tokens == 1
    assert prefill.pcp_prefill_restore_idx.tolist() == [0, 1, 5, 6, 2, 3, 4, 7, 8, 9]
    assert prefill.pcp_local_prefill_indices.tolist() == [0, 1, 4, 5, 6]
    assert chunked.actual_seq_lengths_kv == [3, 10]
    assert chunked.actual_chunk_seq_lengths.tolist() == [4, 10]
    assert chunked.local_context_lens.tolist() == [3, 7]
    assert chunked.starts.tolist() == [0, 0]


def test_gqa_pcp_dcp_prefill_skips_replicated_decode_and_restores_global_current() -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.pcp_group = SimpleNamespace(
        all_gather=Mock(side_effect=lambda tensor, dim: torch.cat((tensor, tensor + 100), dim=dim))
    )
    impl.num_heads = 1
    impl.num_kv_heads = 1
    impl.head_size = 2
    impl.scale = 0.5
    impl._pcp_gathered_kv = (
        torch.arange(14).view(7, 1, 2),
        torch.arange(14, 28).view(7, 1, 2),
    )
    prefill = AscendMetadataForPrefill(
        pcp_actual_seq_lengths_q=[2, 4],
        pcp_prefill_restore_idx=torch.tensor([0, 1, 3, 4]),
        pcp_local_prefill_indices=torch.tensor([0, 1]),
        pcp_local_num_input_tokens=4,
        pcp_local_num_decode_tokens=1,
        pcp_global_num_decode_tokens=1,
    )
    metadata = AscendAttentionDCPMetadata(
        prefill=prefill,
        causal=True,
        attn_mask=torch.ones(4, 4, dtype=torch.bool),
    )
    query = torch.arange(8).view(4, 1, 2)
    key = value = query
    current = torch.arange(8).view(4, 1, 2) + 1000

    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with patch(module + ".torch.ops.npu.npu_fused_infer_attention_score", return_value=(current, torch.zeros(4, 1, 1))) as fia:
        actual = impl._forward_prefill_pcp_dcp(query, key, value, (), metadata)

    torch.testing.assert_close(actual, current[:2])
    restored_query = fia.call_args.args[0]
    torch.testing.assert_close(restored_query, torch.tensor([[[2, 3]], [[4, 5]], [[102, 103]], [[104, 105]]]))
    torch.testing.assert_close(
        fia.call_args.args[1],
        torch.tensor([[[2, 3]], [[4, 5]], [[8, 9]], [[10, 11]]]),
    )
    assert fia.call_args.kwargs["actual_seq_lengths"] == [2, 4]
    assert impl.pcp_group.all_gather.call_count == 1


def test_gqa_chunked_prefill_uses_shared_dcp_merge_for_pcp_overlap() -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.pcp_enabled = False
    impl.num_heads = 4
    impl.num_kv_heads = 1
    impl.head_size = 2
    impl.scale = 0.5

    chunked = AscendMetadataForPrefill.ChunkedContextMetadata(
        actual_chunk_seq_lengths=torch.tensor([3], dtype=torch.int32),
        actual_seq_lengths_kv=[5],
        starts=torch.zeros(1, dtype=torch.int32),
        chunk_seq_mask_filtered_indices=torch.arange(3),
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
            actual_seq_lengths_q=torch.tensor([3], dtype=torch.int32),
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
            module + ".torch.ops.npu.npu_fused_infer_attention_score",
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
    combine.assert_called_once_with(
        packed_history,
        2,
        scatter_dim=1,
        local_output=current_output,
        local_lse=current_lse,
    )
    torch.testing.assert_close(actual, merged)


def test_dcp_chunked_request_mask_marks_nonempty_contexts() -> None:
    local_context_lens = torch.tensor([0, 4, 7], dtype=torch.int32)

    assert AscendAttentionDCPMetadataBuilder._get_chunked_req_mask(local_context_lens) == [
        False,
        True,
        True,
    ]

@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [1, 8])
def test_dcp_chunked_prefill_keeps_host_parameters_and_device_history_separate(size, rank, interleave):
    builder = object.__new__(AscendAttentionDCPMetadataBuilder)
    builder.chunked_prefill_enabled = True
    builder.dcp_size, builder.dcp_rank = size, rank
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave))
    # Include a decode request and a prefill request with no cached history.
    seq_lens = torch.tensor([11, 20, 5, 14], dtype=torch.int32)
    query_lens = torch.tensor([1, 3, 5, 6], dtype=torch.int32)
    common = SimpleNamespace(
        seq_lens=seq_lens.clone(),
        query_start_loc=torch.cat([torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0)]),
        dcp_local_seq_lens_cpu=torch.tensor([
            sum((position // interleave) % size == rank for position in range(length))
            for length in seq_lens.tolist()
        ], dtype=torch.int32),
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
    chunked = metadata["prefill"].chunked_context
    expected = [
        sum((position // interleave) % size == rank for position in range(length))
        for length in [17, 0, 8]
    ]
    torch.testing.assert_close(chunked.local_context_lens, torch.tensor(expected, dtype=torch.int32))
    assert chunked.actual_seq_lengths_kv == np.cumsum(expected).tolist()
    assert chunked.chunked_req_mask == [True, False, True]
    assert chunked.local_total_toks == sum(expected)
    assert chunked.chunk_seq_mask_filtered_indices.tolist() == [0, 1, 2, 8, 9, 10, 11, 12, 13]
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
    expected = [
        sum((position // 8) % 2 == rank for position in range(length))
        for length in [10, 18]
    ]
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
def test_dcp_split_uses_builder_config_without_current_context(is_consumer, is_producer, recompute, query_lens):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer, is_kv_producer=is_producer),
    )
    with (
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DCPMetadataBuilderMixin.__init__", return_value=None
        ),
        patch("vllm_ascend.attention.context_parallel.attention_cp.enable_dcp", return_value=True) as dcp,
    ):
        builder = AscendAttentionDCPMetadataBuilder()
    dcp.assert_called_once_with()
    builder.vllm_config = config
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
    num_decodes = sum(q <= 3 for q in query_lens) if is_consumer and not is_producer and recompute else 0
    num_decode_tokens = sum(query_lens[:num_decodes])
    assert actual == (num_decodes, 2 - num_decodes, num_decode_tokens, sum(query_lens) - num_decode_tokens)
