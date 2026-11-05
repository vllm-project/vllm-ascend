# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel import mla_cp
from vllm_ascend.attention.context_parallel.common_cp import CPKVScope
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
    DCPChunkedContextMetadata,
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
    # vLLM main resolves the DCP group through
    # ``get_dcp_world_size_and_rank`` and computes the LSE capability from
    # its result inside ``__new__``.
    with patch(
        "vllm.distributed.parallel_state.get_dcp_world_size_and_rank",
        return_value=(dcp_size, 0),
    ):
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
    SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score_v2")
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
    assert call_kwargs["actual_seq_qlen"] == [4]
    assert call_kwargs["block_table"].shape[0] == 1
    assert call_kwargs["actual_seq_kvlen"].tolist() == [10]


@patch(
    "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
    SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score_v2")
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
    assert call_kwargs["num_query_heads"] == 96
    assert merged["output_shape"] == (4, 96, 3)
    assert merged["softmax_lse_shape"] == (4, 96, 1)


@pytest.fixture
def fia_v2_boundary(monkeypatch):
    """Model the v2 schema and LSE outputs without accepting legacy kwargs."""
    calls = []
    workspace_calls = []

    def fia(
        query,
        key,
        value,
        *,
        query_rope,
        key_rope,
        num_query_heads,
        num_key_value_heads,
        input_layout,
        atten_mask,
        sparse_mode,
        softmax_scale,
        actual_seq_qlen,
        actual_seq_kvlen,
        return_softmax_lse,
        block_table=None,
        block_size=0,
        workspace=None,
        out=None,
    ):
        assert return_softmax_lse is True
        assert value is key
        assert num_key_value_heads == 1
        assert softmax_scale == 0.5
        lse_shape: tuple[int, ...]
        if input_layout == "TND":
            lse_shape = (query.shape[0], num_query_heads, 1)
        else:
            seq_len = query.shape[1] if input_layout == "BSND" else query.shape[2]
            lse_shape = (query.shape[0], num_query_heads, seq_len, 1)
        calls.append(
            SimpleNamespace(
                query=query,
                key=key,
                query_rope=query_rope,
                key_rope=key_rope,
                num_query_heads=num_query_heads,
                input_layout=input_layout,
                atten_mask=atten_mask,
                sparse_mode=sparse_mode,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                block_table=block_table,
                block_size=block_size,
                workspace=workspace,
                out=out,
            )
        )
        if out is not None:
            assert tuple(out[0].shape) == tuple(query.shape)
            assert tuple(out[1].shape) == lse_shape
            assert out[1].dtype == torch.float32
            out[0].fill_(1)
            out[1].zero_()
            return tuple(out)
        return torch.ones_like(query), torch.zeros(lse_shape, dtype=torch.float32)

    def get_workspace(*args, **kwargs):
        fia(*args, **kwargs)
        workspace_calls.append(calls.pop())
        return torch.empty(128, dtype=torch.uint8)

    monkeypatch.setattr(fia, "out", fia, raising=False)
    monkeypatch.setattr(mla_cp.torch_npu, "npu_fused_infer_attention_score_v2", fia)
    monkeypatch.setattr(
        mla_cp.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", get_workspace, raising=False
    )
    monkeypatch.setattr(
        mla_cp.torch_npu,
        "npu_fused_infer_attention_score",
        Mock(side_effect=AssertionError("DCP decode must use FIA v2")),
    )
    return SimpleNamespace(calls=calls, workspace_calls=workspace_calls)


@pytest.fixture
def mla_fia_graph(monkeypatch):
    num_tokens = 2
    graph = SimpleNamespace(
        events={num_tokens: []},
        handles={num_tokens: []},
        attn_params={num_tokens: []},
        workspaces={num_tokens: None},
    )
    monkeypatch.setattr(mla_cp, "get_graph_params", lambda: graph)
    monkeypatch.setattr(
        mla_cp, "update_graph_params_workspaces", lambda count, workspace: graph.workspaces.update({count: workspace})
    )
    monkeypatch.setattr(mla_cp, "weak_ref_tensors", lambda tensor: tensor)
    monkeypatch.setattr(mla_cp.torch_npu.npu, "current_stream", Mock())
    monkeypatch.setattr(torch.npu, "ExternalEvent", Mock())
    monkeypatch.setattr(torch.npu, "graph_task_group_begin", Mock())
    monkeypatch.setattr(torch.npu, "graph_task_group_end", Mock(return_value="fia-v2-handle"))
    return graph


def _make_page_strided_mla_cache():
    # Both components start inside a larger backing allocation, as with
    # component-major hybrid pages. The leading stride includes page padding.
    raw = torch.arange(64, dtype=torch.float32)
    nope = raw.as_strided((2, 2, 1, 4), (24, 4, 4, 1), storage_offset=3)
    rope = raw.as_strided((2, 2, 1, 2), (24, 2, 2, 1), storage_offset=11)
    return nope, rope


def _assert_cache_view_preserved(actual, original):
    assert actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert actual.storage_offset() == original.storage_offset()
    assert actual.stride(0) == original.stride(0)
    assert not actual.is_contiguous()
    torch.testing.assert_close(actual, original.view(actual.shape))


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
def test_mla_dcp_fia_v2_preserves_strided_cache_and_lse(monkeypatch, fia_v2_boundary, mla_fia_graph, capturing, layout):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.scale = 0.5
    impl.speculative_config = object() if layout == "BSND" else None
    impl._v_up_proj_batch_major = lambda output: output
    merged = []

    def merge(output, lse):
        assert output.shape == (2, 2, 4)
        assert lse.shape == (2, 2, 1)
        assert lse.dtype == torch.float32
        merged.append(lse)
        return output

    impl._merge_dcp_attention_output = merge
    num_decodes = 1 if layout == "BSND" else 2
    query_lens = [2] if layout == "BSND" else [1, 1]
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(2),
        block_table=torch.zeros((num_decodes, 2), dtype=torch.int32),
        seq_lens=torch.tensor([4] * num_decodes),
        max_seq_lens=4,
        seq_lens_list=[4] * num_decodes,
        cp_seq_len=[4] * num_decodes,
    )
    metadata = SimpleNamespace(
        external_flashmla=None,
        causal=False,
        decode=decode,
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=num_decodes,
        query_lens=query_lens,
    )
    monkeypatch.setattr(
        mla_cp,
        "_EXTRA_CTX",
        SimpleNamespace(capturing=capturing, is_draft_model=False, is_draft_model_prefill=False),
    )
    nope, rope = _make_page_strided_mla_cache()
    fused_query = torch.randn(2, 2, 6)
    q_nope, q_pe = fused_query.split([4, 2], dim=-1)
    result = impl._forward_decode(DecodeMLAPreprocessResult(q_nope, q_pe, nope, rope), 2, metadata)

    assert result.shape == (2, 2, 4)
    assert len(merged) == 1
    assert len(fia_v2_boundary.calls) == 1
    assert len(fia_v2_boundary.workspace_calls) == int(capturing)
    for call in fia_v2_boundary.calls + fia_v2_boundary.workspace_calls:
        _assert_cache_view_preserved(call.key, nope)
        _assert_cache_view_preserved(call.key_rope, rope)
        assert call.query.is_contiguous()
        assert call.query_rope.untyped_storage().data_ptr() == fused_query.untyped_storage().data_ptr()
        assert call.query_rope.storage_offset() == q_pe.storage_offset()
        assert call.input_layout == layout
        assert call.num_query_heads == 2
        assert call.actual_seq_qlen == (query_lens if layout == "BSND" else None)
        assert call.actual_seq_kvlen == decode.cp_seq_len
        assert call.block_table is decode.block_table
        assert call.block_size == 2
    call = fia_v2_boundary.calls[0]
    if capturing:
        assert call.workspace is mla_fia_graph.workspaces[2]
        assert call.out is not None
        assert len(mla_fia_graph.attn_params[2]) == 1
        assert mla_fia_graph.handles[2] == ["fia-v2-handle"]
    else:
        assert call.out is None
        assert not mla_fia_graph.attn_params[2]


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
@pytest.mark.parametrize("scope", [CPKVScope.HISTORY, CPKVScope.CURRENT])
def test_mla_dcp_split_fia_v2_returns_lse(monkeypatch, fia_v2_boundary, mla_fia_graph, capturing, scope):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.num_kv_heads = 1
    impl.scale = 0.5
    impl.layer_name = "model.layers.0.self_attn.attn"
    monkeypatch.setattr(
        mla_cp,
        "_EXTRA_CTX",
        SimpleNamespace(capturing=capturing, is_draft_model=False, is_draft_model_prefill=False),
    )
    mla_fia_graph.workspaces[2] = torch.empty(128, dtype=torch.uint8)
    q_nope, q_pe = torch.randn(2, 2, 4), torch.randn(2, 2, 2)
    if scope is CPKVScope.HISTORY:
        nope, rope = _make_page_strided_mla_cache()
        key, key_rope = nope.view(2, 1, 2, 4), rope.view(2, 1, 2, 2)
        block_table, block_size = torch.zeros(1, 2, dtype=torch.int32), 2
        mask, sparse_mode, kv_lengths = None, 0, [4]
    else:
        key, key_rope = torch.randn(2, 1, 4), torch.randn(2, 1, 2)
        block_table, block_size = None, 0
        mask, sparse_mode, kv_lengths = torch.zeros(2, 2, dtype=torch.bool), 3, [2]

    output, lse = impl._run_dcp_mtp_split_attention_op(
        q_nope,
        q_pe,
        key,
        key_rope,
        attn_mask=mask,
        sparse_mode=sparse_mode,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths=[2],
        actual_seq_lengths_kv=kv_lengths,
        attention_kind=scope,
    )

    assert output.shape == (2, 2, 4)
    assert lse.shape == (2, 2, 1)
    assert lse.dtype == torch.float32
    assert len(fia_v2_boundary.calls) == 1
    call = fia_v2_boundary.calls[0]
    assert call.key is key and call.key_rope is key_rope
    assert call.input_layout == "TND"
    assert call.actual_seq_qlen == [2]
    assert call.actual_seq_kvlen == kv_lengths
    assert call.block_table is block_table
    assert call.block_size == block_size
    assert call.atten_mask is mask
    assert call.sparse_mode == sparse_mode
    if scope is CPKVScope.HISTORY:
        _assert_cache_view_preserved(call.key, nope)
        _assert_cache_view_preserved(call.key_rope, rope)
    if capturing:
        assert call.workspace is mla_fia_graph.workspaces[2]
        params = mla_fia_graph.attn_params[2][0]
        assert params.attention_kind is scope
        assert params.layer_name == impl.layer_name
        assert params.attention_params[1] is key
        assert params.attention_params[3] is key_rope
        assert mla_fia_graph.handles[2] == ["fia-v2-handle"]
    else:
        assert call.out is None
        assert not mla_fia_graph.attn_params[2]


@pytest.mark.parametrize(
    "use_spec_decode,is_draft,is_prefill,query_ends,expected",
    [
        pytest.param(True, False, False, [1, 2], True, id="speculative-target"),
        pytest.param(True, True, False, [1, 2], False, id="draft-decode"),
        pytest.param(True, True, False, [2, 3], True, id="multi-token-draft"),
        pytest.param(False, True, True, [1, 2], True, id="draft-prefill"),
        pytest.param(False, False, False, [1, 2], False, id="decode"),
        pytest.param(False, False, False, [1, 4], True, id="multi-token-decode"),
    ],
)
def test_mla_split_selection_matches_shared_policy(use_spec_decode, is_draft, is_prefill, query_ends, expected):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.speculative_config = object() if use_spec_decode else None
    metadata = SimpleNamespace(
        external_flashmla=None, causal=True, decode=SimpleNamespace(actual_seq_lengths_q=query_ends)
    )
    with patch(
        "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
        SimpleNamespace(is_draft_model=is_draft, is_draft_model_prefill=is_prefill, capturing=True),
    ):
        assert impl._decode_requires_current_kv(metadata) is expected


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
    impl.speculative_config = None
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
        expected_stream = "main" if kwargs["attention_kind"] == CPKVScope.HISTORY else "attn"
        assert active[0] == expected_stream
        events.append(kwargs["attention_kind"])
        if kwargs["attention_kind"] == CPKVScope.HISTORY:
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
            mla_cp,
            "_EXTRA_CTX",
            SimpleNamespace(capturing=workspace_sizes is not None, is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch.object(mla_cp, "get_graph_params", return_value=graph_params),
        patch.object(
            mla_cp.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", workspace_query, create=True
        ),
        patch.object(mla_cp, "cp_decode_comm_stream", return_value=attn),
        patch.object(torch.npu, "current_stream", return_value=main),
        patch.object(torch.npu, "stream", side_effect=on_stream),
        patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
        patch("torch.ops.vllm.dcp_a2a_fused", side_effect=communicate) as history_update,
        patch.object(mla_cp, "fused_dcp_lse_combine", side_effect=merge) as update,
        patch("torch_npu.npu_attention_update", side_effect=AssertionError("unexpected NPU update")),
    ):
        metadata = SimpleNamespace(external_flashmla=None, decode=decode, causal=True)
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
    for index, call in enumerate(workspace_query.call_args_list):
        kwargs = call.kwargs
        assert kwargs["return_softmax_lse"] is True
        assert kwargs["num_query_heads"] == (query_head_count if index == 0 else impl.num_heads)
        assert kwargs["softmax_scale"] == impl.scale
        assert kwargs["actual_seq_qlen"] == [2]
        assert kwargs["actual_seq_kvlen"] == [2]
        assert {"softmax_lse_flag", "num_heads", "scale", "antiquant_mode", "antiquant_scale"}.isdisjoint(kwargs)
    history_update.assert_called_once()
    update.assert_called_once()
    assert record_stream.call_count == 7
    assert events == [
        CPKVScope.HISTORY,
        "history_ready",
        ("attn_wait", "ready"),
        CPKVScope.CURRENT,
        "attn_done",
        "history_collective",
        ("main_wait", "done"),
        "merge",
    ]
