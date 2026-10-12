# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
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


def test_mla_dcp_decode_metadata_separates_history_and_preserves_padded_queries() -> None:
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        actual_seq_lengths_q=[4, 8],
    )
    mtp_mask = torch.zeros((2, 8, 32), dtype=torch.bool)
    dcp_metadata = SimpleNamespace(
        draft_cp_seq_len=torch.tensor([12, 11], dtype=torch.int32),
        num_computed_tokens_of_dcp=[[12, 8]],
        dcp_mtp_attn_mask=mtp_mask,
    )
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 1
    builder.dcp_size = 2
    builder.dcp_rank = 0
    builder.cp_local_block_size = 4
    builder.query_lens = torch.tensor([4, 4])
    builder._require_dcp_metadata = lambda _metadata: dcp_metadata

    with patch.object(
        AscendMLAMetadataBuilder,
        "build_decode_metadata",
        return_value=decode,
    ):
        result = builder.build_decode_metadata(
            common_prefix_len=0,
            common_attn_metadata=SimpleNamespace(),
        )

    assert result is decode
    assert result.cp_seq_len.tolist() == [12]
    assert result.cp_history_seq_len == [8, 0]
    assert result.actual_seq_lengths_q == [4, 8]
    assert result.dcp_mtp_attn_mask is None


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
    impl._merge_dcp_attention_output = lambda output, _lse, _rank: output
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

    def merge(output, softmax_lse, _rank):
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
def test_split_decode_overlaps_history_communication(dcp_size, dcp_rank, workspace_sizes, cached_size):
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
    q_nope = torch.arange(2 * 2 * dcp_size * 4).float().view(2, 2 * dcp_size, 4)
    q_pe = torch.zeros(2, 2 * dcp_size, 2)
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
    history_output = torch.ones(2, 2 * dcp_size, 4)
    history_lse = torch.zeros(2, 2 * dcp_size, 1)
    current_output = torch.full((2, 2, 4), 3.0)
    current_lse = torch.zeros(2, 2, 1)
    transferred = torch.ones(2, 2, 5)
    transferred[..., 4] = torch.log(torch.tensor(float(dcp_size)))
    transferred[1, ..., :4] = float("nan")
    transferred[1, ..., 4] = -torch.inf
    current_lse[1] = torch.inf
    current_output[1] = float("nan")
    expected = torch.full((2, 2, 4), (dcp_size + 3.0) / (dcp_size + 1.0))
    expected[1] = 0.0
    events: list[object] = []
    active = ["main"]
    main = Mock()
    comm = Mock()

    def record_history_ready() -> str:
        events.append("history_ready")
        return "ready"

    def record_comm_done() -> str:
        events.append("comm_done")
        return "done"

    main.record_event.side_effect = record_history_ready
    comm.wait_event.side_effect = lambda event: events.append(("comm_wait", event))
    comm.record_event.side_effect = record_comm_done
    main.wait_event.side_effect = lambda event: events.append(("main_wait", event))

    @contextmanager
    def on_stream(stream):
        assert stream is comm
        active[0] = "comm"
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
        assert active[0] == "main"
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
        start = dcp_rank * impl.num_heads
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

    def communicate(out, lse, size, scatter_dim, group_name, return_lse):
        assert active[0] == "comm"
        assert out is history_output and lse is history_lse
        assert size == dcp_size and scatter_dim == 1 and return_lse
        assert group_name == ("dcp-test" if dcp_size > 1 else "")
        events.append("history_collective")
        return transferred

    def merge(partials, head_dim, scatter_dim):
        assert active[0] == "main"
        assert events[-1] == ("main_wait", "done")
        assert partials.shape == (2, 2, 2, 5)
        assert partials.dtype == torch.float32 and partials.is_contiguous()
        assert head_dim == 4 and scatter_dim == 0
        torch.testing.assert_close(partials[0], transferred, equal_nan=True)
        torch.testing.assert_close(partials[1, ..., :4], current_output, equal_nan=True)
        torch.testing.assert_close(partials[1, ..., 4:], current_lse)
        events.append("merge")
        # CPU reference for the fused kernel; invalid shards carry zero weight.
        lses = partials[..., 4:]
        valid = torch.isfinite(lses)
        weights = torch.softmax(lses.masked_fill(~valid, -torch.inf), dim=0)
        weights = torch.nan_to_num(weights, nan=0.0)
        outputs = torch.where(valid, partials[..., :4], 0.0)
        return (outputs * weights).sum(0)

    impl._run_dcp_mtp_split_attention_op = attention
    impl._v_up_proj_batch_major = Mock(side_effect=lambda x: x)
    with (
        patch.object(
            mla_cp, "_EXTRA_CTX", SimpleNamespace(capturing=workspace_sizes is not None, is_draft_model=False)
        ),
        patch.object(mla_cp, "get_graph_params", return_value=graph_params),
        patch.object(mla_cp.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", workspace_query),
        patch.object(mla_cp, "_dcp_mtp_comm_stream", return_value=comm),
        patch.object(torch.npu, "current_stream", return_value=main),
        patch.object(torch.npu, "stream", side_effect=on_stream),
        patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
        patch("torch.ops.vllm.sfa_dcp_a2a_fused", side_effect=communicate) as history_update,
        patch.object(mla_cp, "fused_sfa_dcp_lse_combine", side_effect=merge) as update,
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
    assert record_stream.call_count == 3
    assert events == [
        MLASplitAttentionKind.HISTORY,
        "history_ready",
        ("comm_wait", "ready"),
        "history_collective",
        "comm_done",
        MLASplitAttentionKind.CURRENT,
        ("main_wait", "done"),
        "merge",
    ]
