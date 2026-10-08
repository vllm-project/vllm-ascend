# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import fields
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel import mla_cp
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
    DCPChunkedContextMetadata,
    MLASplitAttentionKind,
)
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    AscendMLAPrefillMetadata,
    DecodeMLAPreprocessResult,
)


def test_mla_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendMlaDCPImpl, AscendMLAImpl)
    assert issubclass(
        AscendMlaDCPMetadataBuilder,
        AscendMLAMetadataBuilder,
    )
    assert AscendMlaDCPMetadataBuilder.decode_metadata_cls is (AscendMLADCPDecodeMetadata)
    base_fields = {field.name for field in fields(AscendMLADecodeMetadata)}
    dcp_fields = {field.name for field in fields(AscendMLADCPDecodeMetadata)}
    assert {"cp_seq_len", "dcp_mtp_attn_mask"}.isdisjoint(base_fields)
    assert {"cp_seq_len", "dcp_mtp_attn_mask"} <= dcp_fields


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_mla_dcp_passes_runner_v2_cp_compatibility(dcp_size) -> None:
    group = SimpleNamespace(world_size=dcp_size, rank_in_group=0)
    with patch("vllm.distributed.parallel_state.get_dcp_group", return_value=group):
        impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    assert impl.need_to_return_lse_for_decode == (dcp_size > 1)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )
    with patch(
        "vllm.v1.worker.cp_utils.get_layers_from_vllm_config",
        return_value={"attention": SimpleNamespace(impl=impl)},
    ):
        check_attention_cp_compatibility(config)


@pytest.mark.parametrize("num_decodes", [1, 2], ids=["v1-padding", "v2-empty-row"])
def test_mla_dcp_consumes_local_lengths_and_only_partitions_history(num_decodes) -> None:
    lengths = torch.tensor([20, 0][:num_decodes], dtype=torch.int32)
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4 * num_decodes),
        block_table=torch.ones((num_decodes, 2), dtype=torch.int32),
        seq_lens=lengths,
        max_seq_lens=20,
        seq_lens_list=lengths.tolist(),
        actual_seq_lengths_q=[4, 8],
    )
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = num_decodes
    builder.dcp_size = 2
    builder.dcp_rank = 0
    builder.cp_local_block_size = 4
    builder.seq_lens = lengths
    builder.query_lens = torch.tensor([4, 4], dtype=torch.int32)
    # A sentinel distinct from recomputing total lengths proves producer ownership.
    common = SimpleNamespace(dcp_local_seq_lens_cpu=torch.tensor([11, 0], dtype=torch.int32))
    with (
        patch.object(AscendMLAMetadataBuilder, "build_decode_metadata", return_value=decode),
        patch.object(mla_cp, "get_dcp_local_seq_lens", wraps=mla_cp.get_dcp_local_seq_lens) as partition,
    ):
        result = builder.build_decode_metadata(0, common)
    partition.assert_called_once()
    assert partition.call_args.args[0].tolist() == [16, 0][:num_decodes]
    assert result is decode
    assert result.cp_seq_len == [11, 0][:num_decodes]
    assert result.cp_history_seq_len == [8, 0]
    assert result.actual_seq_lengths_q == [4, 8]
    assert result.dcp_mtp_attn_mask is None


@pytest.mark.parametrize("num_prefills,dcp_size", [(1, 2), (31, 8)])
def test_mla_dcp_v2_mixed_batch_survives_base_decode_length_slice(num_prefills, dcp_size) -> None:
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 1
    builder.num_decode_tokens = 1
    builder.num_actual_tokens = 1 + 5 * num_prefills
    builder.dcp_size = dcp_size
    builder.dcp_rank = 0
    builder.cp_local_block_size = 1
    builder.seq_lens = torch.tensor([11] + [18] * num_prefills, dtype=torch.int32)
    builder.query_lens = torch.tensor([1] + [5] * num_prefills, dtype=torch.int32)
    builder.block_table = torch.ones((1 + num_prefills, 2), dtype=torch.int32)
    builder.graph_pad_size = -1
    builder.use_mla_rope = False
    builder.attn_mask_builder = Mock()
    builder.nope_zero_rope_cache = None
    common = SimpleNamespace(
        context_parallel_metadata=None,
        num_reqs=1 + num_prefills,
        dcp_local_seq_lens_cpu=torch.tensor(
            [(length + dcp_size - 1) // dcp_size for length in [11] + [18] * num_prefills],
            dtype=torch.int32,
        ),
        query_start_loc_cpu=torch.cat([torch.zeros(1, dtype=torch.int32), builder.query_lens.cumsum(0)]),
        positions=torch.arange(builder.num_actual_tokens),
    )

    # Exercise the real base builder, which slices seq_lens to decodes but
    # deliberately retains the complete mixed-batch query_lens tensor.
    result = builder.build_decode_metadata(0, common)

    assert result.cp_seq_len == [(11 + dcp_size - 1) // dcp_size]
    assert result.cp_history_seq_len == [(10 + dcp_size - 1) // dcp_size]
    assert result.actual_seq_lengths_q == [1]
    assert builder.seq_lens.tolist() == [11]
    assert builder.query_lens.tolist() == [1] + [5] * num_prefills


def test_mla_dcp_reorg_decode_query_gathers_fused_query() -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 2
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    q_nope = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    q_pe = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    group = SimpleNamespace(all_gather=lambda tensor, dim: torch.cat([tensor, tensor + 100], dim=dim))
    impl.dcp_group = group
    gathered_nope, gathered_pe = impl.reorg_decode_q(q_nope, q_pe)

    assert gathered_nope.shape == (1, 4, 3)
    assert gathered_pe.shape == (1, 4, 2)
    torch.testing.assert_close(gathered_nope[:, :2], q_nope)
    torch.testing.assert_close(gathered_pe[:, :2], q_pe)
    torch.testing.assert_close(gathered_nope[:, 2:], q_nope + 100)
    torch.testing.assert_close(gathered_pe[:, 2:], q_pe + 100)


def test_mla_dcp_uses_padded_local_chunk_lengths() -> None:
    padded_lengths = torch.tensor([[4, 2], [1, 0]], dtype=torch.int32)
    chunked = DCPChunkedContextMetadata(
        cu_seq_lens=torch.tensor([0, 2]),
        starts=torch.zeros(1, dtype=torch.int32),
        seq_tot=[6, 1],
        max_seq_lens=[4, 1],
        workspace=torch.empty(0),
        chunk_seq_lens=torch.empty(0, dtype=torch.int32),
        chunk_seq_lens_npu=torch.empty(0, dtype=torch.int32),
        chunk_actual_seq_lengths_kv_list=[[4, 6], [1, 1]],
        padded_chunk_seq_lens_npu=padded_lengths,
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=torch.tensor([2]),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=AscendMLAPrefillMetadata(
            attn_mask=None,
            query_lens=torch.tensor([2]),
            seq_lens=[2],
            context_lens=torch.tensor([0]),
            input_positions=torch.arange(2),
            query_start_loc=torch.tensor([0, 2]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked,
        ),
    )
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)

    torch.testing.assert_close(impl.get_context_seq_len_npu(1, metadata), padded_lengths[1])


@patch(
    "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
    SimpleNamespace(is_draft_model=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score")
def test_mla_dcp_mixed_cache_hit_batch_uses_decode_bsnd_metadata(mock_fia) -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 1
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl._merge_dcp_attention_output = lambda output, _lse: output
    impl._v_up_proj_batch_major = lambda output: output

    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        cp_seq_len=torch.tensor([10], dtype=torch.int32),
        dcp_mtp_attn_mask=torch.zeros((1, 1, 4, 4)),
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=102,
        slot_mapping=torch.arange(102),
        query_start_loc=torch.tensor([0, 4, 18, 32, 46, 60, 74, 88, 102]),
        seq_lens=torch.tensor([20, 14, 14, 14, 14, 14, 14, 14]),
        seq_lens_cpu=torch.tensor([20, 14, 14, 14, 14, 14, 14, 14]),
        block_tables=torch.ones((8, 2), dtype=torch.int32),
        num_decodes=1,
        num_decode_tokens=4,
        num_prefills=7,
        query_lens=[4, 14, 14, 14, 14, 14, 14, 14],
        attn_state=AscendAttentionState.PrefillCacheHit,
        decode=decode,
    )

    q_nope = torch.randn(4, 2, 3)
    q_pe = torch.randn(4, 2, 2)
    k_nope = torch.randn(2, 1, 2, 3)
    k_pe = torch.randn(2, 1, 2, 2)
    mock_fia.return_value = (
        torch.randn(1, 4, 2, 3),
        torch.randn(1, 2, 4, 1),
    )

    metadata.causal = False
    impl._forward_decode(
        DecodeMLAPreprocessResult(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
        ),
        2,
        metadata,
    )

    call_args = mock_fia.call_args.args
    call_kwargs = mock_fia.call_args.kwargs
    assert call_args[0].shape == (1, 4, 2, 3)
    assert call_kwargs["input_layout"] == "BSND"
    assert call_kwargs["actual_seq_lengths"] == [4]
    assert call_kwargs["block_table"].shape[0] == 1
    assert call_kwargs["actual_seq_lengths_kv"].tolist() == [10]


@patch(
    "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
    SimpleNamespace(is_draft_model=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score")
def test_mla_dcp_uses_native_global_query_heads_for_fia(mock_fia) -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 8
    impl.num_heads = 12
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)

    merged = {}

    def merge(output, softmax_lse):
        merged["output_shape"] = output.shape
        merged["softmax_lse_shape"] = softmax_lse.shape
        return output

    impl._merge_dcp_attention_output = merge
    impl._v_up_proj_batch_major = lambda output: output

    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        cp_seq_len=torch.tensor([10], dtype=torch.int32),
        dcp_mtp_attn_mask=torch.zeros((1, 1, 4, 4)),
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=4,
        slot_mapping=torch.arange(4),
        query_start_loc=torch.tensor([0, 4]),
        seq_lens=torch.tensor([20]),
        seq_lens_cpu=torch.tensor([20]),
        block_tables=torch.ones((1, 2), dtype=torch.int32),
        num_decodes=1,
        num_decode_tokens=4,
        num_prefills=0,
        query_lens=[4],
        attn_state=AscendAttentionState.DecodeOnly,
        decode=decode,
    )

    q_nope = torch.randn(4, 96, 3)
    q_pe = torch.randn(4, 96, 2)
    k_nope = torch.randn(2, 1, 2, 3)
    k_pe = torch.randn(2, 1, 2, 2)
    mock_fia.return_value = (
        torch.randn(1, 4, 96, 3),
        torch.randn(1, 96, 4, 1),
    )

    metadata.causal = False
    impl._forward_decode(
        DecodeMLAPreprocessResult(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
        ),
        2,
        metadata,
    )

    call_args = mock_fia.call_args.args
    call_kwargs = mock_fia.call_args.kwargs
    assert call_args[0].shape == (1, 4, 96, 3)
    assert call_kwargs["query_rope"].shape == (1, 4, 96, 2)
    assert call_kwargs["num_heads"] == 96
    assert merged["output_shape"] == (4, 96, 3)
    assert merged["softmax_lse_shape"] == (4, 96, 1)


@pytest.mark.parametrize(
    "dcp_size,dcp_rank,workspace_sizes,cached_size",
    [
        (1, 0, None, None),
        (2, 0, None, None),
        (2, 1, None, None),
        (16, 15, None, None),
        (2, 1, (64, 128), None),
        (2, 1, (128, 64), None),
        (2, 1, (64, 128), 256),
    ],
)
@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("history_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_split_decode_packs_on_main_overlapping_current_attention(
    dcp_size, dcp_rank, workspace_sizes, cached_size, history_dtype, pcp_size
):
    import vllm_ascend.attention.context_parallel.mla_cp as mla_cp

    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.scale = 0.5
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.dcp_size = dcp_size
    impl.dcp_rank = dcp_rank
    impl.dcp_device_group = object()
    impl.dcp_group = SimpleNamespace(unique_name="dcp-test")
    if pcp_size > dcp_size:
        pytest.skip("PCP ranks belong to the physical DCP group")
    tp_size = dcp_size // pcp_size
    tp_rank = dcp_rank % tp_size
    impl.pcp_group = SimpleNamespace(world_size=pcp_size, unique_name="pcp-test")
    impl.tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank, unique_name="tp-test")
    query_head_count = impl.num_heads * tp_size
    q_nope = torch.arange(2 * query_head_count * 4).float().view(2, query_head_count, 4)
    q_pe = torch.zeros(2, query_head_count, 2)
    current_k = torch.ones(2, 1, 4)
    current_pe = torch.ones(2, 1, 2)
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(2),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.tensor([4]),
        max_seq_lens=4,
        seq_lens_list=[4],
        actual_seq_lengths_q=[2],
        cp_history_seq_len=[2],
    )
    decode.attn_mask = torch.zeros(2, 2, dtype=torch.bool)
    history_output = torch.ones(2, query_head_count, 4, dtype=history_dtype)
    history_lse = torch.zeros(2, query_head_count, 1)
    current_output = torch.full((2, 2, 4), 3.0)
    current_lse = torch.zeros(2, 2, 1)
    transferred = torch.ones(dcp_size, 2, 2, 5)
    transferred[..., 4] = 0.0
    transferred[:, :, 1, :4] = float("nan")
    transferred[:, :, 1, 4] = -torch.inf
    current_lse[1] = torch.inf
    current_output[1] = float("nan")
    expected = torch.full((2, 2, 4), (dcp_size + 3.0) / (dcp_size + 1.0))
    expected[1] = 0.0
    events: list[object] = []
    active = ["main"]
    main = Mock()
    attn = Mock()

    def record_history_ready() -> str:
        events.append("history_ready")
        return "ready"

    def record_attn_done() -> str:
        events.append("attn_done")
        return "done"

    main.record_event.side_effect = record_history_ready
    attn.wait_event.side_effect = lambda event: events.append(("attn_wait", event))
    attn.record_event.side_effect = record_attn_done
    main.wait_event.side_effect = lambda event: events.append(("main_wait", event))

    @contextmanager
    def on_stream(stream):
        assert stream is attn
        active[0] = "attn"
        yield
        active[0] = "main"

    graph_params = SimpleNamespace(workspaces={2: torch.empty(cached_size, dtype=torch.uint8) if cached_size else None})
    workspace_query = Mock(
        side_effect=[torch.empty(n, dtype=torch.uint8) for n in workspace_sizes] if workspace_sizes else []
    )

    def attention(q, q_rope, k, k_rope, **kwargs):
        if workspace_sizes is not None:
            assert set(graph_params.workspaces) == {2}
            assert graph_params.workspaces[2].numel() == (cached_size or max(workspace_sizes))
        expected_stream = "main" if kwargs["attention_kind"] == MLASplitAttentionKind.HISTORY else "attn"
        assert active[0] == expected_stream
        events.append(kwargs["attention_kind"])
        if kwargs["attention_kind"] == MLASplitAttentionKind.HISTORY:
            torch.testing.assert_close(q, q_nope)
            torch.testing.assert_close(q_rope, q_pe)
            assert kwargs["actual_seq_lengths_kv"] == [2]
            assert kwargs["actual_seq_lengths"] == [2]
            assert kwargs["block_table"] is decode.block_table
            assert kwargs["block_size"] == 2
            assert kwargs["attn_mask"] is None
            assert kwargs["sparse_mode"] == 0
            return history_output, history_lse
        start = (tp_rank if pcp_size > 1 else dcp_rank) * impl.num_heads
        torch.testing.assert_close(q, q_nope[:, start : start + impl.num_heads])
        torch.testing.assert_close(q_rope, q_pe[:, start : start + impl.num_heads])
        torch.testing.assert_close(k, current_k)
        torch.testing.assert_close(k_rope, current_pe)
        assert kwargs["actual_seq_lengths"] == [2]
        assert kwargs["actual_seq_lengths_kv"] == [2]
        assert kwargs["attn_mask"] is decode.attn_mask
        assert kwargs["block_size"] == 0
        assert kwargs["block_table"] is None
        assert kwargs["sparse_mode"] == 3
        return current_output, current_lse

    def communicate(out, lse, size, scatter_dim, group_name, pcp_group_name=None, defer_combine=False):
        assert active[0] == "main"
        assert out is history_output and lse is history_lse
        assert size == tp_size and scatter_dim == 1 and defer_combine
        if pcp_size > 1:
            assert group_name == "tp-test" and pcp_group_name == "pcp-test"
        else:
            assert group_name == ("dcp-test" if dcp_size > 1 else "")
            assert pcp_group_name is None
        events.append("history_collective")
        return transferred

    def merge(partials, head_dim, scatter_dim, local_output, local_lse):
        assert active[0] == "main"
        assert events[-1] == ("main_wait", "done")
        assert partials is transferred
        assert head_dim == 4 and scatter_dim == 1
        assert local_output is current_output and local_lse is current_lse
        events.append("merge")
        # Independent reference counts every history rank and current KV once.
        history = partials.transpose(1, 2)
        lses = torch.cat((history[..., 4:], local_lse.unsqueeze(0)), dim=0)
        values = torch.cat((history[..., :4], local_output.unsqueeze(0)), dim=0)
        valid = torch.isfinite(lses)
        weights = torch.softmax(lses.masked_fill(~valid, -torch.inf), dim=0)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (torch.where(valid, values, 0.0) * weights).sum(0)

    impl._run_dcp_mtp_split_attention_op = attention
    impl._v_up_proj_batch_major = Mock(side_effect=lambda x: x)
    with (
        patch.object(
            mla_cp, "_EXTRA_CTX", SimpleNamespace(capturing=workspace_sizes is not None, is_draft_model=False)
        ),
        patch.object(mla_cp, "get_graph_params", return_value=graph_params),
        patch.object(mla_cp.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", workspace_query),
        patch.object(mla_cp, "_dcp_mtp_comm_stream", return_value=attn),
        patch.object(torch.npu, "current_stream", return_value=main),
        patch.object(torch.npu, "stream", side_effect=on_stream),
        patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
        patch("torch.ops.vllm.dcp_a2a_fused", side_effect=communicate) as history_update,
        patch.object(mla_cp, "fused_dcp_lse_combine", side_effect=merge) as update,
        patch("torch_npu.npu_attention_update", side_effect=AssertionError("unexpected NPU update")),
    ):
        metadata = SimpleNamespace(decode=decode, causal=True)
        assert impl._decode_requires_current_kv(metadata)
        assert not AscendMLAImpl.__new__(AscendMLAImpl)._decode_requires_current_kv(metadata)
        actual = impl._forward_decode(
            DecodeMLAPreprocessResult(
                ql_nope=q_nope,
                q_pe=q_pe,
                k_nope=torch.zeros(1, 1, 2, 4),
                k_pe=torch.zeros(1, 1, 2, 2),
                current_k_nope=current_k,
                current_k_pe=current_pe,
            ),
            2,
            metadata,
        )
    torch.testing.assert_close(actual, expected)
    assert workspace_query.call_count == (2 if workspace_sizes is not None and cached_size is None else 0)
    history_update.assert_called_once()
    update.assert_called_once()
    assert record_stream.call_count == 7
    assert events == [
        MLASplitAttentionKind.HISTORY,
        "history_ready",
        ("attn_wait", "ready"),
        MLASplitAttentionKind.CURRENT,
        "attn_done",
        "history_collective",
        ("main_wait", "done"),
        "merge",
    ]


class TestMLADCPQReplication(unittest.TestCase):
    @staticmethod
    def _make_impl(active=True, dcp=2, rank=0):
        impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
        impl.num_heads = 2
        impl.dcp_size = dcp
        impl.dcp_rank = rank
        impl.is_pcp_decode_sharded = False
        impl.dcp_q_replicate_enabled = active
        impl.q_projection_heads = 2 * (dcp if active else 1)
        impl.qk_nope_head_dim = 3
        impl.qk_rope_head_dim = 2
        impl.qk_head_dim = 5
        impl.kv_lora_rank = 2
        impl.v_head_dim = 3
        impl.num_kv_heads = 1
        impl.enable_mlapo = impl.fa_quant_layer = False
        return impl

    def test_decode_projection_matches_gathered_local_projections(self):
        for tokens, dcp in product([0, 1, 3], [2, 4]):
            with self.subTest(tokens=tokens, dcp=dcp):
                self._check_decode_projection_matches_gathered_local_projections(tokens, dcp)

    def _check_decode_projection_matches_gathered_local_projections(self, tokens, dcp):
        torch.manual_seed(12)
        group_heads = 2 * dcp
        x = torch.randn(tokens, 4, dtype=torch.float64)
        q_weight = torch.randn(group_heads * 5, 4, dtype=torch.float64)
        uk = torch.randn(group_heads, 3, 2, dtype=torch.float64)
        impl = self._make_impl(dcp=dcp)
        impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
        impl.W_UK_T_dcp_group = uk
        impl._dcp_all_gather_fragments = Mock(side_effect=AssertionError("Q gather must be skipped"))
        actual_nope, actual_pe = impl.reorg_decode_q(*impl._q_proj_and_k_up_proj(x))
        expected_nope, expected_pe = [], []
        for rank in range(dcp):
            q = torch.nn.functional.linear(x, q_weight[rank * 10 : (rank + 1) * 10]).view(tokens, 2, 5)
            expected_nope.append(torch.einsum("thp,hpl->thl", q[..., :3], uk[rank * 2 : (rank + 1) * 2]))
            expected_pe.append(q[..., 3:])
        torch.testing.assert_close(actual_nope, torch.cat(expected_nope, dim=1))
        torch.testing.assert_close(actual_pe, torch.cat(expected_pe, dim=1))
        impl._dcp_all_gather_fragments.assert_not_called()

    def test_query_gather_off_and_invalid_group_heads(self):
        impl = self._make_impl(active=False)
        q, pe = torch.randn(3, 2, 2), torch.randn(3, 2, 2)
        impl._dcp_all_gather_fragments = Mock(return_value=(q, pe))
        impl.reorg_decode_q(q, pe)
        impl._dcp_all_gather_fragments.assert_called_once_with(q, pe, dim=1)
        impl.dcp_q_replicate_enabled = True
        with self.assertRaisesRegex(ValueError, "complete group head"):
            impl.reorg_decode_q(q, pe)

    def test_group_k_weights_refresh_from_local_weights(self):
        for active in [False, True]:
            with self.subTest(active=active):
                self._check_group_k_weights_refresh_from_local_weights(active)

    def _check_group_k_weights_refresh_from_local_weights(self, active):
        impl = self._make_impl(active)
        weights = torch.arange(2 * 2 * 6, dtype=torch.float32).view(12, 2)
        impl.kv_b_proj = SimpleNamespace(weight=weights.clone(), quant_method=UnquantizedLinearMethod())
        impl.q_proj = Mock()
        remote = torch.arange(12, dtype=torch.float32).view(2, 3, 2) + 100
        gather = Mock(side_effect=lambda local, dim: torch.cat((local, remote), dim=dim))
        with (
            patch.object(mla_v1, "get_dcp_group", return_value=SimpleNamespace(all_gather=gather)),
            patch.object(mla_v1.torch_npu, "npu_format_cast", side_effect=lambda x, fmt: x),
            patch.object(mla_v1, "maybe_trans_nz", side_effect=lambda x: x),
        ):
            captured_group_weight = None
            for offset in (0, 0, 200):
                impl.kv_b_proj.weight.copy_(weights + offset)
                impl.process_weights_after_loading(torch.float32)
                local = (weights + offset).view(2, 6, 2)
                torch.testing.assert_close(impl.W_UK_T, local[:, :3, :])
                torch.testing.assert_close(impl.W_UV, local[:, 3:, :].transpose(1, 2))
                if active:
                    if captured_group_weight is None:
                        captured_group_weight = impl.W_UK_T_dcp_group
                    self.assertEqual(captured_group_weight.data_ptr(), impl.W_UK_T_dcp_group.data_ptr())
                    torch.testing.assert_close(impl.W_UK_T_dcp_group, torch.cat((local[:, :3, :], remote)))
        self.assertEqual(gather.call_count, 3 if active else 0)
        impl.q_proj.prepare_weights_for_processing.assert_not_called()

    def test_prefill_keeps_tokens_and_local_heads_in_mixed_batch(self):
        for decode_tokens, rank in product([0, 2], [0, 1]):
            with self.subTest(decode_tokens=decode_tokens, rank=rank):
                self._check_prefill_keeps_tokens_and_local_heads_in_mixed_batch(decode_tokens, rank)

    def _check_prefill_keeps_tokens_and_local_heads_in_mixed_batch(self, decode_tokens, rank):
        impl = self._make_impl(rank=rank)

        class Projection:
            group_size = 2
            rank_in_group = rank

            def __call__(self, x):
                raise AssertionError("Prefill must not run the group-width projection")

            def forward_local(self, x):
                return (x[:, rank * 10 : (rank + 1) * 10],)

        impl.q_proj = Projection()
        impl._get_num_prefill_kv_tokens = lambda _: 3
        impl.rope_single = lambda q, cos, sin: q
        impl.exec_kv_prefill = Mock(return_value=(torch.zeros(3, 1, 2), torch.zeros(3, 2)))
        impl.kv_b_proj = lambda x: (torch.zeros(3, 12),)
        q = torch.arange((decode_tokens + 3) * 20).view(decode_tokens + 3, 20)
        metadata = SimpleNamespace(
            num_decode_tokens=decode_tokens,
            num_actual_tokens=decode_tokens + 3,
            prefill=SimpleNamespace(cos=None, sin=None),
            slot_mapping=torch.arange(decode_tokens + 3),
        )
        result = impl.mla_preprocess_prefill(q, torch.zeros(decode_tokens + 3, 4), None, metadata)
        expected = q[decode_tokens:].view(3, 4, 5)[:, rank * 2 : rank * 2 + 2]
        torch.testing.assert_close(result.q_nope, expected[..., :3])
        torch.testing.assert_close(result.q_pe, expected[..., 3:])
        self.assertEqual(result.q_nope.shape, (3, 2, 3))
        torch.testing.assert_close(impl.exec_kv_prefill.call_args.args[4], metadata.slot_mapping[decode_tokens:])

    def test_replicated_decode_against_explicit_full_mla(self):
        for lengths in [(3, 4), (0, 7), (7, 0)]:
            with self.subTest(lengths=lengths):
                self._check_replicated_decode_against_explicit_full_mla(lengths)

    def _check_replicated_decode_against_explicit_full_mla(self, lengths):
        torch.manual_seed(21)
        q_weight = torch.randn(20, 4, dtype=torch.float64)
        x = torch.randn(1, 4, dtype=torch.float64)
        uk = torch.randn(4, 3, 2, dtype=torch.float64)
        uv = torch.randn(4, 2, 3, dtype=torch.float64)
        cache = torch.randn(sum(lengths), 2, dtype=torch.float64)
        position_k = torch.randn(sum(lengths), 2, dtype=torch.float64)
        impl = self._make_impl()
        impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
        impl.W_UK_T_dcp_group = uk
        absorbed, pe = impl._q_proj_and_k_up_proj(x)
        q = torch.nn.functional.linear(x, q_weight).view(1, 4, 5)
        full_k = torch.cat((torch.einsum("jl,hpl->jhp", cache, uk), position_k[:, None, :].expand(-1, 4, -1)), -1)
        full_v = torch.einsum("jl,hlv->jhv", cache, uv)
        scale = 5**-0.5
        expected = torch.einsum("thj,jhv->thv", (torch.einsum("thd,jhd->thj", q, full_k) * scale).softmax(-1), full_v)
        partial, lse, start = [], [], 0
        for length in lengths:
            end = start + length
            if length:
                scores = (
                    torch.einsum("thl,jl->thj", absorbed, cache[start:end])
                    + torch.einsum("thr,jr->thj", pe, position_k[start:end])
                ) * scale
                partial.append(torch.einsum("thj,jl->thl", scores.softmax(-1), cache[start:end]))
                lse.append(scores.logsumexp(-1))
            else:
                partial.append(torch.zeros_like(absorbed))
                lse.append(torch.full((1, 4), -torch.inf, dtype=torch.float64))
            start = end
        # Independent LSE merge reference, followed by the existing local V projection.
        factors = torch.stack(lse).softmax(0)
        merged = (torch.stack(partial) * factors[..., None]).sum(0)
        for rank in range(2):
            local = slice(rank * 2, rank * 2 + 2)
            impl.W_UV = uv[local]
            impl.num_heads = 2
            with patch.object(
                mla_v1.torch_npu,
                "npu_transpose_batchmatmul",
                side_effect=lambda a, b, perm_x1, perm_y: torch.bmm(a.permute(perm_x1), b).permute(perm_y),
            ):
                actual = impl._v_up_proj_batch_major(merged[:, local])
            torch.testing.assert_close(actual.view(1, 2, 3), expected[:, local], atol=1e-10, rtol=1e-10)

    def test_multi_query_split_decode_matches_causal_mla(self):
        for rank, dcp, q_lens in product([0, 1], [2, 4], [(1,), (3,), (1, 3)]):
            with self.subTest(rank=rank, dcp=dcp, q_lens=q_lens):
                self._check_multi_query_split_decode_matches_causal_mla(rank, dcp, q_lens)

    def _check_multi_query_split_decode_matches_causal_mla(self, rank, dcp, q_lens):
        torch.manual_seed(23)
        impl = self._make_impl(dcp=dcp, rank=rank)
        impl.scale = 5**-0.5
        impl.dcp_group = SimpleNamespace(unique_name="dcp-q-replicate-test")
        tokens, heads = sum(q_lens), 2 * dcp
        q_weight = torch.randn(heads * 5, 4, dtype=torch.float64)
        uk = torch.randn(heads, 3, 2, dtype=torch.float64)
        uv = torch.randn(heads, 2, 3, dtype=torch.float64)
        x = torch.randn(tokens, 4, dtype=torch.float64)
        impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
        impl.W_UK_T_dcp_group = uk
        impl._dcp_all_gather_fragments = Mock(side_effect=AssertionError("Unexpected Q gather"))
        q, pe = impl.reorg_decode_q(*impl._q_proj_and_k_up_proj(x))
        # Each request has one historical token on each DCP rank.
        history = torch.randn(len(q_lens), dcp, 2, dtype=torch.float64)
        history_pe = torch.randn_like(history)
        current = torch.randn(tokens, 1, 2, dtype=torch.float64)
        current_pe = torch.randn_like(current)
        history_outputs, history_lses, expected = [], [], []
        for shard in range(dcp):
            out, lse, start = [], [], 0
            for req, length in enumerate(q_lens):
                end = start + length
                scores = (q[start:end] @ history[req, shard] + pe[start:end] @ history_pe[req, shard]) * impl.scale
                out.append(history[req, shard].expand(length, heads, 2))
                lse.append(scores.unsqueeze(-1))
                if shard == 0:
                    for token in range(start, end):
                        keys = torch.cat((history[req], current[start : token + 1, 0]))
                        rope = torch.cat((history_pe[req], current_pe[start : token + 1, 0]))
                        full_q = torch.nn.functional.linear(x[token], q_weight).view(heads, 5)
                        full_k = torch.cat(
                            (torch.einsum("jl,hpl->jhp", keys, uk), rope[:, None].expand(-1, heads, -1)), -1
                        )
                        full_v = torch.einsum("jl,hlv->jhv", keys, uv)
                        probs = (torch.einsum("hd,jhd->hj", full_q, full_k) * impl.scale).softmax(-1)
                        expected.append(torch.einsum("hj,jhv->hv", probs, full_v))
                start = end
            history_outputs.append(torch.cat(out))
            history_lses.append(torch.cat(lse))

        local = slice(rank * 2, rank * 2 + 2)
        lengths = torch.tensor(q_lens).cumsum(0).tolist()
        decode = mla_cp.AscendMLADCPDecodeMetadata(
            input_positions=torch.arange(tokens),
            block_table=torch.zeros(len(q_lens), 1, dtype=torch.int32),
            seq_lens=torch.tensor(q_lens) + dcp,
            max_seq_lens=max(q_lens) + dcp,
            seq_lens_list=[n + dcp for n in q_lens],
            actual_seq_lengths_q=lengths,
            cp_history_seq_len=[1] * len(q_lens),
        )
        decode.attn_mask = torch.ones(tokens, tokens, dtype=torch.bool).triu(1)

        def attention(query, query_pe, key, key_pe, **kwargs):
            self.assertEqual(kwargs["actual_seq_lengths"], lengths)
            if kwargs["attention_kind"] == mla_cp.MLASplitAttentionKind.HISTORY:
                torch.testing.assert_close(query, q)
                self.assertEqual(kwargs["actual_seq_lengths_kv"], [1] * len(q_lens))
                self.assertIs(kwargs["attn_mask"], None)
                return history_outputs[rank], history_lses[rank]
            torch.testing.assert_close(query, q[:, local])
            torch.testing.assert_close(query_pe, pe[:, local])
            self.assertTrue(kwargs["attn_mask"] is decode.attn_mask and kwargs["sparse_mode"] == 3)
            self.assertEqual(kwargs["actual_seq_lengths_kv"], lengths)
            out, lse, start = [], [], 0
            for length in q_lens:
                for token in range(start, start + length):
                    scores = (
                        query[token] @ key[start : token + 1, 0].T + query_pe[token] @ key_pe[start : token + 1, 0].T
                    ) * impl.scale
                    out.append(scores.softmax(-1) @ key[start : token + 1, 0])
                    lse.append(scores.logsumexp(-1, keepdim=True))
                start += length
            return torch.stack(out), torch.stack(lse)

        def communicate(out, lse, size, scatter_dim, group_name, defer_combine):
            self.assertTrue(out is history_outputs[rank] and lse is history_lses[rank])
            self.assertTrue(size == dcp and scatter_dim == 1 and defer_combine)
            return object()

        def combine(_packed, _dim, scatter_dim, local_output, local_lse):
            values = torch.stack([v[:, local] for v in history_outputs] + [local_output])
            lses = torch.stack([v[:, local] for v in history_lses] + [local_lse])
            return (values * lses.softmax(0)).sum(0)

        impl._run_dcp_mtp_split_attention_op = attention
        impl._v_up_proj_batch_major = lambda out: torch.einsum("thl,hlv->thv", out, uv[local])
        stream = Mock()
        with (
            patch.object(mla_cp, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
            patch.object(mla_cp, "_dcp_mtp_comm_stream", return_value=stream),
            patch.object(torch.npu, "current_stream", return_value=stream),
            patch.object(torch.npu, "stream", side_effect=lambda _: nullcontext()),
            patch.object(torch.Tensor, "record_stream"),
            patch("torch.ops.vllm.dcp_a2a_fused", side_effect=communicate),
            patch.object(mla_cp, "fused_dcp_lse_combine", side_effect=combine),
        ):
            actual = impl._forward_decode_split_attention(
                q,
                pe,
                history[:, rank].contiguous(),
                history_pe[:, rank].contiguous(),
                current,
                current_pe,
                1,
                SimpleNamespace(decode=decode),
            )
        torch.testing.assert_close(actual, torch.stack(expected)[:, local], atol=1e-10, rtol=1e-10)
        impl._dcp_all_gather_fragments.assert_not_called()

    def test_graph_capture_and_update_preserve_group_and_local_heads(self):
        for draft_prefill, draft in product([False, True], [False, True]):
            with self.subTest(draft_prefill=draft_prefill, draft=draft):
                self._check_graph_capture_and_update_preserve_group_and_local_heads(draft_prefill, draft)

    def _check_graph_capture_and_update_preserve_group_and_local_heads(self, draft_prefill, draft):
        impl = self._make_impl()
        impl.scale = 0.5
        impl.layer_name = "model.layers.0.self_attn.attn"
        # Eight tokens represent a padded capture bucket, independently of head count.
        tokens = 8
        params = SimpleNamespace(
            attn_params=defaultdict(list),
            events=defaultdict(list),
            handles=defaultdict(list),
            workspaces={tokens: torch.empty(64)},
        )
        ctx = SimpleNamespace(capturing=True, is_draft_model=draft, is_draft_model_prefill=draft_prefill)
        stream, event = Mock(), Mock()
        op = Mock()
        steps = 2 if draft else 1
        with (
            patch.object(mla_cp, "_EXTRA_CTX", ctx),
            patch.object(mla_cp, "get_graph_params", return_value=params),
            patch.object(mla_cp, "get_draft_graph_params", return_value=params),
            patch.object(mla_cp, "get_draft_graph_prefill_params", return_value=params),
            patch.object(mla_cp, "weak_ref_tensors", side_effect=lambda x: x),
            patch.object(mla_cp.torch_npu, "npu_fused_infer_attention_score", op),
            patch.object(torch.npu, "current_stream", return_value=stream),
            patch.object(torch.npu, "ExternalEvent", return_value=event),
            patch.object(torch.npu, "stream", side_effect=lambda _: nullcontext()),
            patch.object(torch.npu, "graph_task_group_begin"),
            patch.object(torch.npu, "graph_task_group_end", return_value=object()),
            patch.object(torch.npu, "graph_task_update_begin"),
            patch.object(torch.npu, "graph_task_update_end"),
        ):
            for _ in range(steps):
                for kind, heads in [
                    (mla_cp.MLASplitAttentionKind.HISTORY, 4),
                    (mla_cp.MLASplitAttentionKind.CURRENT, 2),
                ]:
                    q = torch.zeros(tokens, heads, 2)
                    impl._run_dcp_mtp_split_attention_op(
                        q,
                        q.clone(),
                        torch.zeros(tokens, 1, 2),
                        torch.zeros(tokens, 1, 2),
                        attn_mask=None,
                        sparse_mode=0,
                        block_table=None,
                        block_size=0,
                        actual_seq_lengths=[4, 8],
                        actual_seq_lengths_kv=[4, 8],
                        attention_kind=kind,
                    )
            captured = list(op.out.call_args_list)
            self.assertEqual([c.kwargs["num_heads"] for c in captured], [4, 2] * steps)
            # Replay updates request lengths but retains the captured Q/output storage.
            metadatas = [
                {
                    impl.layer_name: SimpleNamespace(
                        decode=SimpleNamespace(
                            actual_seq_lengths_q=[2 + step, 8],
                            cp_history_seq_len=[5 + step, 0],
                            block_table=torch.zeros(2, 1, dtype=torch.int32),
                        )
                    )
                }
                for step in range(steps)
            ]
            op.out.reset_mock()
            impl.update_graph_params(
                stream, SimpleNamespace(attn_metadata=metadatas[0]), tokens, draft_attn_metadatas=metadatas
            )
            for i, call in enumerate(op.out.call_args_list):
                meta = metadatas[i // 2][impl.layer_name].decode
                self.assertIs(call.args[0], captured[i].args[0])
                self.assertIs(call.kwargs["out"][0], captured[i].kwargs["out"][0])
                self.assertEqual(call.kwargs["num_heads"], 4 if i % 2 == 0 else 2)
                self.assertEqual(call.kwargs["actual_seq_lengths"], meta.actual_seq_lengths_q)
                self.assertEqual(
                    call.kwargs["actual_seq_lengths_kv"],
                    meta.cp_history_seq_len if i % 2 == 0 else meta.actual_seq_lengths_q,
                )
            self.assertEqual(op.out.call_count, 2 * steps)

    def test_direct_and_low_rank_preprocess_preserve_multi_token_mixed_batch(self):
        for q_rank in [None, 4]:
            with self.subTest(q_rank=q_rank):
                self._check_direct_and_low_rank_preprocess_preserve_multi_token_mixed_batch(q_rank)

    def _check_direct_and_low_rank_preprocess_preserve_multi_token_mixed_batch(self, q_rank):
        torch.manual_seed(31)
        impl = self._make_impl(rank=1)
        impl.layerwise_kv_cache_hook = None
        hidden = torch.randn(7, 6)
        kv = torch.randn(7, 4)
        impl.q_lora_rank = q_rank
        impl.fused_qkv_a_proj = (lambda x: (torch.cat((x[:, :4], kv), -1),)) if q_rank else None
        impl.q_a_layernorm = lambda x: x * 2
        impl.kv_a_proj_with_mqa = lambda x: (kv,)
        q_c = hidden if q_rank is None else hidden[:, :4] * 2
        weight = torch.randn(20, q_c.shape[-1])

        class Projection:
            group_size = 2
            rank_in_group = 1

            def __call__(self, x):
                assert x.shape[0] == 4  # Only the decode rows use group heads.
                return (torch.nn.functional.linear(x, weight),)

            def forward_local(self, x):
                assert x.shape[0] == 3
                return (torch.nn.functional.linear(x, weight[10:]),)

        impl.q_proj = Projection()
        impl.W_UK_T_dcp_group = torch.randn(4, 3, 2)
        impl.rope_single = lambda x, cos, sin: x
        impl._dcp_all_gather_fragments = Mock(side_effect=AssertionError("Unexpected Q gather"))
        impl.exec_kv_decode = Mock(return_value=tuple(torch.zeros(4, 1, 2) for _ in range(4)))
        impl.exec_kv_prefill = Mock(return_value=(torch.zeros(3, 1, 2), torch.zeros(3, 2)))
        impl.kv_b_proj = lambda x: (torch.zeros(3, 12),)
        impl._get_num_prefill_kv_tokens = lambda _: 3
        metadata = SimpleNamespace(
            causal=True,
            num_decodes=2,
            num_prefills=1,
            num_decode_tokens=4,
            num_actual_tokens=7,
            slot_mapping=torch.arange(7),
            decode=SimpleNamespace(cos=None, sin=None),
            prefill=SimpleNamespace(cos=None, sin=None),
        )
        with (
            patch.object(mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
            patch.object(mla_v1, "wait_for_kv_layer_from_connector"),
            patch.object(mla_v1, "notify_kv_cache_written"),
        ):
            decode, prefill = impl._mla_preprocess("layer", hidden, None, metadata)
        projected = torch.nn.functional.linear(q_c, weight).view(7, 4, 5)
        expected = torch.einsum("thp,hpl->thl", projected[:4, :, :3], impl.W_UK_T_dcp_group)
        torch.testing.assert_close(decode.ql_nope, expected)
        torch.testing.assert_close(decode.q_pe, projected[:4, :, 3:])
        torch.testing.assert_close(prefill.q_nope, projected[4:, 2:, :3])
        torch.testing.assert_close(prefill.q_pe, projected[4:, 2:, 3:])
        self.assertIs(impl.exec_kv_decode.call_args.kwargs["return_current_kv"], True)
        torch.testing.assert_close(impl.exec_kv_decode.call_args.args[0], kv[:4])
        impl._dcp_all_gather_fragments.assert_not_called()

    def test_speculative_split_dispatch(self):
        for query_lens, (capturing, full_graph), draft in product(
            [[1, 1], [1, 3]], [(False, False), (True, False), (False, True)], [False, True]
        ):
            with self.subTest(query_lens=query_lens, capturing=capturing, full_graph=full_graph, draft=draft):
                impl = self._make_impl()
                metadata = SimpleNamespace(causal=True, num_decodes=2, query_lens=query_lens)
                with (
                    patch.object(mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=draft, capturing=capturing)),
                    patch.object(
                        mla_cp,
                        "get_forward_context",
                        return_value=SimpleNamespace(
                            cudagraph_runtime_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
                        ),
                    ),
                ):
                    self.assertIs(
                        impl._decode_requires_current_kv(metadata),
                        not draft or capturing or full_graph or (max(query_lens) > 1),
                    )
                    metadata.causal = False
                    self.assertFalse(impl._decode_requires_current_kv(metadata))
