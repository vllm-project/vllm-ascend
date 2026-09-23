# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.ops.dcp_linear import AscendDCPGroupColumnParallelLinear


def make_impl(active=True, dcp=2, rank=0):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.num_heads = 2
    impl.dcp_size = dcp
    impl.dcp_rank = rank
    impl.dcp_q_replicate = active
    impl.q_projection_heads = 2 * (dcp if active else 1)
    impl.qk_nope_head_dim = 3
    impl.qk_rope_head_dim = 2
    impl.qk_head_dim = 5
    impl.kv_lora_rank = 2
    impl.v_head_dim = 3
    impl.num_kv_heads = 1
    impl.enable_mlapo = impl.fa_quant_layer = False
    return impl


@pytest.mark.parametrize("dcp", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 3])
def test_decode_projection_matches_gathered_local_projections(dcp, tokens):
    torch.manual_seed(12)
    group_heads = 2 * dcp
    x = torch.randn(tokens, 4, dtype=torch.float64)
    q_weight = torch.randn(group_heads * 5, 4, dtype=torch.float64)
    uk = torch.randn(group_heads, 3, 2, dtype=torch.float64)
    impl = make_impl(dcp=dcp)
    impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
    impl.W_UK_T_dcp_qrep = uk
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


def test_query_gather_off_and_invalid_group_heads():
    impl = make_impl(active=False)
    q, pe = torch.randn(3, 2, 2), torch.randn(3, 2, 2)
    impl._dcp_all_gather_fragments = Mock(return_value=(q, pe))
    impl.reorg_decode_q(q, pe)
    impl._dcp_all_gather_fragments.assert_called_once_with(q, pe, dim=1)
    impl.dcp_q_replicate = True
    with pytest.raises(ValueError, match="complete group head"):
        impl.reorg_decode_q(q, pe)


@pytest.mark.parametrize("active", [False, True])
def test_group_k_weights_refresh_from_local_weights(active):
    impl = make_impl(active)
    weights = torch.arange(2 * 2 * 6, dtype=torch.float32).view(12, 2)
    impl.kv_b_proj = SimpleNamespace(weight=weights.clone(), quant_method=UnquantizedLinearMethod())
    remote = torch.arange(12, dtype=torch.float32).view(2, 3, 2) + 100
    gather = Mock(side_effect=lambda local, dim: torch.cat((local, remote), dim=dim))
    with (
        patch.object(mla_v1, "get_dcp_group", return_value=SimpleNamespace(all_gather=gather)),
        patch.object(mla_v1.torch_npu, "npu_format_cast", side_effect=lambda x, fmt: x),
        patch.object(mla_v1, "maybe_trans_nz", side_effect=lambda x: x),
    ):
        for offset in (0, 0, 200):
            impl.kv_b_proj.weight.copy_(weights + offset)
            impl.process_weights_after_loading(torch.float32)
            local = (weights + offset).view(2, 6, 2)
            torch.testing.assert_close(impl.W_UK_T, local[:, :3, :])
            torch.testing.assert_close(impl.W_UV, local[:, 3:, :].transpose(1, 2))
            if active:
                torch.testing.assert_close(impl.W_UK_T_dcp_qrep, torch.cat((local[:, :3, :], remote)))
    assert gather.call_count == (3 if active else 0)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("decode_tokens", [0, 2])
def test_prefill_keeps_tokens_and_local_heads_in_mixed_batch(rank, decode_tokens):
    impl = make_impl(rank=rank)

    class Projection:
        group_size = 2
        rank_in_group = rank
        _local_view = AscendDCPGroupColumnParallelLinear._local_view

        def __call__(self, x):
            return (x,)

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
    assert result.q_nope.shape == (3, 2, 3)
    torch.testing.assert_close(impl.exec_kv_prefill.call_args.args[4], metadata.slot_mapping[decode_tokens:])


@pytest.mark.parametrize("lengths", [(3, 4), (0, 7), (7, 0)])
def test_replicated_decode_against_explicit_full_mla(lengths):
    torch.manual_seed(21)
    q_weight = torch.randn(20, 4, dtype=torch.float64)
    x = torch.randn(1, 4, dtype=torch.float64)
    uk = torch.randn(4, 3, 2, dtype=torch.float64)
    uv = torch.randn(4, 2, 3, dtype=torch.float64)
    cache = torch.randn(sum(lengths), 2, dtype=torch.float64)
    position_k = torch.randn(sum(lengths), 2, dtype=torch.float64)
    impl = make_impl()
    impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
    impl.W_UK_T_dcp_qrep = uk
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


@pytest.mark.parametrize("q_lens", [(1,), (3,), (1, 3)])
@pytest.mark.parametrize("dcp", [2, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_multi_query_split_decode_matches_causal_mla(q_lens, dcp, rank):
    """Exercise the actual split path, replacing only device/collective boundaries."""
    from contextlib import nullcontext

    from vllm_ascend.attention.context_parallel import mla_cp

    torch.manual_seed(23)
    impl = make_impl(dcp=dcp, rank=rank)
    impl.scale = 5**-0.5
    impl.dcp_group = SimpleNamespace(unique_name="qrep-test")
    tokens, heads = sum(q_lens), 2 * dcp
    q_weight = torch.randn(heads * 5, 4, dtype=torch.float64)
    uk = torch.randn(heads, 3, 2, dtype=torch.float64)
    uv = torch.randn(heads, 2, 3, dtype=torch.float64)
    x = torch.randn(tokens, 4, dtype=torch.float64)
    impl.q_proj = lambda x: (torch.nn.functional.linear(x, q_weight),)
    impl.W_UK_T_dcp_qrep = uk
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
                    full_k = torch.cat((torch.einsum("jl,hpl->jhp", keys, uk), rope[:, None].expand(-1, heads, -1)), -1)
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
        assert kwargs["actual_seq_lengths"] == lengths
        if kwargs["attention_kind"] == mla_cp.MLASplitAttentionKind.HISTORY:
            torch.testing.assert_close(query, q)
            assert kwargs["actual_seq_lengths_kv"] == [1] * len(q_lens)
            assert kwargs["attn_mask"] is None
            return history_outputs[rank], history_lses[rank]
        torch.testing.assert_close(query, q[:, local])
        torch.testing.assert_close(query_pe, pe[:, local])
        assert kwargs["attn_mask"] is decode.attn_mask and kwargs["sparse_mode"] == 3
        assert kwargs["actual_seq_lengths_kv"] == lengths
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
        assert out is history_outputs[rank] and lse is history_lses[rank]
        assert size == dcp and scatter_dim == 1 and defer_combine
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
        patch("torch.ops.vllm.sfa_dcp_a2a_fused", side_effect=communicate),
        patch.object(mla_cp, "fused_sfa_dcp_lse_combine", side_effect=combine),
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


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("draft_prefill", [False, True])
def test_graph_capture_and_update_preserve_group_and_local_heads(draft, draft_prefill):
    from collections import defaultdict
    from contextlib import nullcontext

    from vllm_ascend.attention.context_parallel import mla_cp

    impl = make_impl()
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
            for kind, heads in [(mla_cp.MLASplitAttentionKind.HISTORY, 4), (mla_cp.MLASplitAttentionKind.CURRENT, 2)]:
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
        assert [c.kwargs["num_heads"] for c in captured] == [4, 2] * steps
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
            assert call.args[0] is captured[i].args[0]
            assert call.kwargs["out"][0] is captured[i].kwargs["out"][0]
            assert call.kwargs["num_heads"] == (4 if i % 2 == 0 else 2)
            assert call.kwargs["actual_seq_lengths"] == meta.actual_seq_lengths_q
            assert call.kwargs["actual_seq_lengths_kv"] == (
                meta.cp_history_seq_len if i % 2 == 0 else meta.actual_seq_lengths_q
            )
        assert op.out.call_count == 2 * steps


@pytest.mark.parametrize("q_rank", [None, 4])
def test_direct_and_low_rank_preprocess_preserve_multi_token_mixed_batch(q_rank):
    from vllm_ascend.attention.context_parallel import mla_cp

    torch.manual_seed(31)
    impl = make_impl(rank=1)
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
        _local_view = AscendDCPGroupColumnParallelLinear._local_view

        def __call__(self, x):
            return (torch.nn.functional.linear(x, weight),)

    impl.q_proj = Projection()
    impl.W_UK_T_dcp_qrep = torch.randn(4, 3, 2)
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
    expected = torch.einsum("thp,hpl->thl", projected[:4, :, :3], impl.W_UK_T_dcp_qrep)
    torch.testing.assert_close(decode.ql_nope, expected)
    torch.testing.assert_close(decode.q_pe, projected[:4, :, 3:])
    torch.testing.assert_close(prefill.q_nope, projected[4:, 2:, :3])
    torch.testing.assert_close(prefill.q_pe, projected[4:, 2:, 3:])
    assert impl.exec_kv_decode.call_args.kwargs["return_current_kv"] is True
    torch.testing.assert_close(impl.exec_kv_decode.call_args.args[0], kv[:4])
    impl._dcp_all_gather_fragments.assert_not_called()


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("capturing,full_graph", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("query_lens", [[1, 1], [1, 3]])
def test_speculative_split_dispatch(draft, capturing, full_graph, query_lens):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.attention.context_parallel import mla_cp

    impl = make_impl()
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
        assert impl._decode_requires_current_kv(metadata) is (
            not draft or capturing or full_graph or max(query_lens) > 1
        )
        metadata.causal = False
        assert not impl._decode_requires_current_kv(metadata)
