# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl


@pytest.mark.parametrize("tp_size,pcp_size", [(1, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("logical_pcp", [True, False], ids=["target", "replicated-draft"])
def test_query_gather_restores_only_distinct_tp_heads(tp_size, pcp_size, logical_pcp):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = tp_size * pcp_size
    impl.pcp_enabled = logical_pcp
    impl.pcp_group = SimpleNamespace(world_size=pcp_size)
    impl.tp_group = SimpleNamespace(
        all_gather=Mock(side_effect=lambda q, dim: torch.cat([q + 100 * rank for rank in range(tp_size)], dim=dim))
    )
    impl.dcp_group = SimpleNamespace(all_gather=Mock(side_effect=AssertionError("duplicated PCP query heads")))
    q_nope = torch.arange(12).float().view(2, 2, 3)
    q_pe = torch.arange(8).float().view(2, 2, 2)
    actual_nope, actual_pe = impl.reorg_decode_q(q_nope, q_pe)
    if tp_size == 1:
        assert actual_nope is q_nope and actual_pe is q_pe
        impl.tp_group.all_gather.assert_not_called()
    else:
        impl.tp_group.all_gather.assert_called_once()
        torch.testing.assert_close(actual_nope, torch.cat([q_nope + 100 * r for r in range(tp_size)], dim=1))
        torch.testing.assert_close(actual_pe, torch.cat([q_pe + 100 * r for r in range(tp_size)], dim=1))
    impl.dcp_group.all_gather.assert_not_called()


@pytest.mark.parametrize("tp_size,pcp_size,tp_rank,dcp_rank", [(1, 2, 0, 1), (4, 2, 2, 6), (2, 4, 1, 7)])
def test_current_attention_selects_tp_local_heads(tp_size, pcp_size, tp_rank, dcp_rank):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.num_heads = 2
    impl.dcp_size = tp_size * pcp_size
    impl.dcp_rank = dcp_rank
    impl.pcp_group = SimpleNamespace(world_size=pcp_size)
    impl.tp_group = SimpleNamespace(rank_in_group=tp_rank)
    q_nope = torch.arange(2 * tp_size * 3).float().view(1, 2 * tp_size, 3)
    q_pe = q_nope[..., :2]
    actual_nope, actual_pe = impl._local_decode_query(q_nope, q_pe)
    start = tp_rank * impl.num_heads
    torch.testing.assert_close(actual_nope, q_nope[:, start : start + 2])
    torch.testing.assert_close(actual_pe, q_pe[:, start : start + 2])
    assert actual_nope.shape[1] == impl.num_heads


@pytest.mark.parametrize("tp_size,pcp_size", [(1, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("defer", [False, True], ids=["full-kv", "history-current"])
def test_output_exchange_uses_sfa_tp_and_pcp_groups(tp_size, pcp_size, defer):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = tp_size * pcp_size
    impl.pcp_group = SimpleNamespace(world_size=pcp_size, unique_name="pcp")
    impl.tp_group = SimpleNamespace(world_size=tp_size, unique_name="tp")
    impl.dcp_group = SimpleNamespace(unique_name="physical-dcp")
    out = torch.randn(2, 2 * tp_size, 3)
    lse = torch.randn(2, 2 * tp_size, 1)
    result = object()
    with patch("torch.ops.vllm.dcp_a2a_fused", return_value=result) as exchange:
        assert impl._merge_dcp_attention_output(out, lse, defer_combine=defer) is result
    args = exchange.call_args.args
    assert args[0] is out and args[1] is lse
    assert args[2:] == (tp_size, 1, "tp", "pcp")
    assert exchange.call_args.kwargs == {"defer_combine": defer}


def test_local_decode_query_supports_one_gqa_query():
    from vllm_ascend.attention.context_parallel.common_cp import DCPImplMixin

    impl = DCPImplMixin.__new__(DCPImplMixin)
    impl.num_heads = 2
    impl.dcp_size = 8
    impl.dcp_rank = 5
    impl.pcp_group = SimpleNamespace(world_size=2)
    impl.tp_group = SimpleNamespace(rank_in_group=2)
    query = torch.arange(24).float().view(1, 8, 3)
    (local_query,) = impl._local_decode_query(query)
    torch.testing.assert_close(local_query, query[:, 4:6])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("defer", [False, True])
def test_merge_preserves_input_dtype(dtype, defer):
    from vllm_ascend.attention.context_parallel.common_cp import DCPImplMixin

    impl = DCPImplMixin.__new__(DCPImplMixin)
    impl.dcp_size = 1
    impl.pcp_group = SimpleNamespace(world_size=1)
    impl.dcp_group = SimpleNamespace(unique_name="unused")
    out = torch.ones(2, 2, 4, dtype=dtype)
    lse = torch.ones(2, 2, 1, dtype=torch.float32)
    with patch("torch.ops.vllm.dcp_a2a_fused", side_effect=lambda x, *args, **kwargs: x) as merge:
        actual = impl._merge_dcp_attention_output(out, lse, defer_combine=defer)
    assert actual is out and actual.dtype == dtype
    assert merge.call_args.args[0] is out
    assert merge.call_args.args[1] is lse
    assert merge.call_args.args[2:] == (1, 1, "")
    assert merge.call_args.kwargs == {"defer_combine": defer}


@pytest.mark.parametrize("dcp_enabled", [False, True])
@pytest.mark.parametrize("with_pcp_context", [False, True])
def test_capture_forwards_pcp_context_and_preserves_dcp_dummy_metadata(dcp_enabled, with_pcp_context):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder

    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.dcp_enabled = dcp_enabled
    builder.reorder_batch_threshold = 3
    expected = object()
    builder.build = Mock(return_value=expected)
    common = SimpleNamespace(attn_state=None, is_prefilling=None, num_reqs=3, num_actual_tokens=9, max_query_len=3)
    context = object()
    kwargs = {"pcp_context": context, "pcp_cache_group_idx": 2} if with_pcp_context else {}
    assert builder.build_for_cudagraph_capture(common, **kwargs) is expected
    args = builder.build.call_args.args
    call = builder.build.call_args.kwargs
    assert args[0] == 0
    captured = args[1]
    assert captured is not common
    assert captured.attn_state == AscendAttentionState.ChunkedPrefill
    assert common.attn_state is None and common.is_prefilling is None
    if dcp_enabled:
        assert captured.is_prefilling.shape == (3,)
        assert not captured.is_prefilling.any()
    else:
        assert captured.is_prefilling is None
    if with_pcp_context:
        assert call["pcp_context"] is context and call["pcp_cache_group_idx"] == 2
    else:
        assert "pcp_context" not in call


@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("local_histories", [[0, 7, 12], [3, 6], [0, 0]])
def test_chunk_reorg_maps_local_fragments_and_restores_logical_senders(interleave, local_histories):
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    from vllm_ascend.attention.context_parallel.mla_cp import DCPChunkedContextMetadata

    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 4
    impl.kv_lora_rank = 1
    impl.qk_rope_head_dim = 1
    impl.pcp_group = SimpleNamespace(world_size=2)
    # Physical HCCL senders [0,1,2,3], logical DCP order [0,2,1,3].
    collective_rank_order = [0, 2, 1, 3]
    global_histories = torch.tensor([4, 12], dtype=torch.int32)
    local = torch.tensor(local_histories, dtype=torch.int32)
    local_lengths = get_dcp_local_seq_lens(local, dcp_size=4, cp_kv_cache_interleave_size=interleave)
    pad = ((global_histories + 4 * interleave - 1) // (4 * interleave) * interleave).tolist()
    offsets = [0, pad[0], sum(pad)]
    table = [1] * len(local_histories)
    meta = DCPChunkedContextMetadata(
        cu_seq_lens=torch.tensor([0]),
        starts=torch.tensor([[0, 0]]),
        seq_tot=[sum(pad)],
        max_seq_lens=[max(local_histories, default=0)],
        workspace=torch.empty(0),
        chunk_seq_lens=local.view(1, -1),
        chunk_seq_lens_npu=local.view(1, -1),
        chunk_actual_seq_lengths_kv_list=[local.cumsum(0).tolist()],
        padded_local_chunk_seq_lens=[pad],
        local_context_lens_allranks=local_lengths.tolist(),
        cu_seq_lens_lst=[[0] + local.cumsum(0).tolist()],
        chunk_size=16,
        pcp_global_req_indices=table,
        padded_local_cu_seq_lens_lst=[offsets],
        kv_rank_offsets=[[rank * sum(pad) for rank in collective_rank_order]],
    )
    senders = []
    for physical_rank in range(4):
        logical_rank = collective_rank_order.index(physical_rank)
        rows = []
        for request, count in enumerate(pad):
            positions = torch.arange(count)
            values = (positions // interleave) * (4 * interleave) + logical_rank * interleave + positions % interleave
            latent = (request * 100 + values).float().view(count, 1, 1)
            rows.append(torch.cat((latent, latent + 1000), dim=-1))
        senders.append(torch.cat(rows))
    recv = torch.cat(senders)
    impl._dcp_all_gather = lambda packed, dim: recv
    latent, rope = impl._reorg_kvcache(
        torch.empty(sum(pad), 1, 1), torch.empty(sum(pad), 1, 1), chunked_context=meta, chunk_idx=0, toks=sum(pad)
    )
    expected = [
        100 + position
        for count in local_histories
        for rank in range(4)
        for position in range(count)
        if (position // interleave) % 4 == rank
    ]
    assert latent.flatten().tolist() == expected
    assert rope.flatten().tolist() == [x + 1000 for x in expected]


def test_pcp_context_exposes_upstream_request_mapping():
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext

    fields = AscendPCPAttentionContext.__dataclass_fields__
    assert "local_to_global_req_indices" in fields
    assert "local_batch" not in fields
    assert "prefill_context_lens_cpu" not in fields


def test_mla_dcp_inherits_standard_build_and_does_not_materialize_whole_cache():
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder

    assert "build" not in AscendMlaDCPMetadataBuilder.__dict__
    assert "_gather_prefill_cache_blocks" not in AscendMlaDCPImpl.__dict__


@pytest.mark.parametrize("pcp_enabled", [False, True])
def test_chunked_metadata_preserves_explicit_plan_and_workspace(pcp_enabled):
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder
    from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder

    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.pcp_enabled = pcp_enabled
    builder.dcp_enabled = True
    builder.num_decodes = 0
    builder.seq_lens = torch.tensor([6, 12], dtype=torch.int32)
    builder.query_lens = torch.tensor([3, 3], dtype=torch.int32)
    builder._get_pcp_prefill_kv_inputs = Mock(return_value=(torch.tensor([12]), [0, 0], torch.zeros(1, 1)))
    plan = torch.tensor([20], dtype=torch.int32)
    with patch.object(AscendMLAMetadataBuilder, "build_chunked_metadata", return_value=None) as base:
        result = builder.build_chunked_metadata(
            0, SimpleNamespace(num_reqs=2), chunk_plan_lens_cpu=plan, chunk_workspace_size=48
        )
    assert result is None
    assert base.call_args.kwargs["chunk_plan_lens_cpu"] is plan
    assert base.call_args.kwargs["chunk_workspace_size"] == 48
    if pcp_enabled:
        builder._get_pcp_prefill_kv_inputs.assert_called_once()
    else:
        builder._get_pcp_prefill_kv_inputs.assert_not_called()


@pytest.mark.parametrize(
    "pcp_size,dcp_size,max_num_seqs,interleave,expected",
    [
        (1, 1, 32, 128, 65536),
        (2, 1, 32, 128, 65536),
        (1, 16, 32, 128, 65536),
        (2, 2, 32, 128, 65536),
        (2, 16, 2, 128, 65536),
        (2, 16, 16, 128, 65536),
        (2, 16, 17, 128, 69632),
        (2, 16, 32, 128, 131072),
        (2, 16, 64, 128, 262144),
        (2, 16, 32, 256, 262144),
    ],
)
def test_pcp_dcp_workspace_fits_an_aligned_chunk_per_fragment(pcp_size, dcp_size, max_num_seqs, interleave, expected):
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8192),
        cache_config=SimpleNamespace(block_size=128),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
    )
    assert AscendMlaDCPMetadataBuilder.determine_chunked_prefill_workspace_size(config) == expected


@pytest.mark.parametrize("workspace_size", [65536, 131072])
def test_pcp_dcp_chunk_plan_rejects_insufficient_budget_and_fits_exact_boundary(workspace_size):
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder

    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.pcp_enabled = builder.dcp_enabled = builder.chunked_prefill_enabled = True
    builder.chunked_prefill_workspace_size = workspace_size
    builder.chunked_prefill_workspace = torch.empty(0)
    builder.block_size = builder.cp_virtual_block_size = 2048
    builder.cp_local_block_size = 128
    builder.dcp_size = 16
    builder.dcp_collective_rank_order = list(range(16))
    builder.device = torch.device("cpu")
    builder.num_decodes = 0
    builder.num_prefills = 64
    builder.query_lens = torch.full((64,), 8, dtype=torch.int32)
    builder.seq_lens = torch.tensor([4104, 4128] * 32, dtype=torch.int32)
    builder._get_pcp_prefill_kv_inputs = Mock(
        return_value=(torch.full((32,), 4128, dtype=torch.int32), [row for row in range(32) for _ in range(2)], None)
    )
    if workspace_size == 65536:
        with pytest.raises(
            ValueError,
            match=r"workspace_size=32768, num_prefills_with_context=32, block_size=2048, minimum_workspace_size=65536",
        ):
            builder.build_chunked_metadata(0, SimpleNamespace(num_reqs=64))
        return

    cpu_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return cpu_zeros(*args, **kwargs)

    with (
        patch.object(torch.Tensor, "pin_memory", lambda x: x),
        patch.object(torch.Tensor, "npu", lambda x: x, create=True),
        patch("torch.zeros", side_effect=zeros_without_pinning),
    ):
        metadata = builder.build_chunked_metadata(0, SimpleNamespace(num_reqs=64))
    assert builder.max_context_chunk == 2048
    assert builder.num_chunks == 3
    assert metadata.chunk_seq_lens.shape == (3, 64)
    assert metadata.seq_tot == [4096, 4096, 4096]
    assert metadata.chunk_seq_lens.sum(1).tolist() == [131072, 131072, 768]


def test_pcp_global_plan_keeps_local_fragment_mapping():
    import numpy as np

    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder, DCPChunkedContextMetadata
    from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder, ChunkedContextMetadata

    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.pcp_enabled = True
    builder.dcp_enabled = True
    builder.chunked_prefill_workspace_size = 64
    builder.dcp_collective_rank_order = [0, 2, 1, 3]
    builder.pcp_size = 2
    builder.dcp_size = 4
    builder.cp_local_block_size = 1
    builder.cp_virtual_block_size = 4
    builder.device = torch.device("cpu")
    builder.num_decodes = 0
    builder.num_prefills = 3
    builder.seq_lens = torch.tensor([6, 12, 24], dtype=torch.int32)
    builder.query_lens = torch.tensor([3, 3, 4], dtype=torch.int32)
    table = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    builder._pcp_context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=2,
            is_prefilling_np=np.array([True, True]),
            seq_lens_np=np.array([12, 24]),
        ),
        local_to_global_req_indices=(0, 0, 1),
        global_block_tables=(table,),
    )
    builder._pcp_cache_group_idx = 0
    lens = torch.tensor([3, 9, 20], dtype=torch.int32)
    chunk_lengths = torch.tensor([[3, 9, 16], [0, 0, 4]], dtype=torch.int32)
    builder.context_lens_cpu = lens
    builder.max_context_chunk = 16
    builder.num_chunks = 2
    builder.chunk_seq_lens = chunk_lengths
    builder.cu_seq_lens_cpu = torch.cat((torch.zeros(2, 1, dtype=torch.int32), chunk_lengths.cumsum(1)), dim=1)
    base_meta = ChunkedContextMetadata(
        cu_seq_lens=builder.cu_seq_lens_cpu,
        starts=torch.tensor([[0, 0, 0], [16, 16, 16]]),
        seq_tot=[28, 4],
        max_seq_lens=[16, 4],
        workspace=torch.empty(0),
        chunk_seq_lens=chunk_lengths,
        chunk_seq_lens_npu=chunk_lengths,
        chunk_actual_seq_lengths_kv_list=[[3, 12, 28], [0, 0, 4]],
    )
    cpu_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return cpu_zeros(*args, **kwargs)

    with (
        patch.object(AscendMLAMetadataBuilder, "build_chunked_metadata", return_value=base_meta) as base,
        patch.object(torch.Tensor, "pin_memory", lambda x: x),
        patch("torch.zeros", side_effect=zeros_without_pinning),
    ):
        result = builder.build_chunked_metadata(0, SimpleNamespace(num_reqs=3))
    assert isinstance(result, DCPChunkedContextMetadata)
    assert base.call_args.kwargs["chunk_plan_lens_cpu"].tolist() == [12, 24]
    assert base.call_args.kwargs["chunk_workspace_size"] == 32
    assert result.pcp_global_req_indices == [0, 0, 1]
    assert result.pcp_global_block_table.data_ptr() == table.data_ptr()
    torch.testing.assert_close(result.pcp_global_block_table, table)
    assert result.padded_local_chunk_seq_lens == [[3, 4], [0, 2]]
    assert result.chunk_actual_seq_lengths_kv_list == [[3, 12, 28], [0, 0, 4]]
    assert result.chunk_seq_lens.shape == (2, 3)
    assert result.seq_tot == [7, 2]
    assert result.padded_local_cu_seq_lens_lst == [[0, 3, 7], [0, 0, 2]]
    assert result.kv_rank_offsets == [[0, 14, 7, 21], [0, 4, 2, 6]]
    assert builder.num_prefills == 3 and builder.num_decodes == 0
