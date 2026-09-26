# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention import sfa_v1
from vllm_ascend.attention.context_parallel import sfa_cp
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl, AscendSFADCPMetadata, DCPGatherContext
from vllm_ascend.attention.utils import PreprocessType


def make_sparse_impl(active=True, dcp=2, rank=0):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.num_heads = impl.local_num_heads = 2
    impl.dcp_size, impl.dcp_rank = dcp, rank
    impl.dcp_q_replicate = active
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = 3, 2
    impl.qk_head_dim, impl.kv_lora_rank, impl.v_head_dim = 5, 2, 3
    impl.q_lora_rank = 4
    impl.q_proj = Mock()
    impl.scale = 5**-0.5
    impl.rl_weight_update_enabled = False
    impl.has_indexer = True
    impl.indexer = Mock()
    impl.enable_mlapo = impl.enable_sparse_sfa_c8 = impl.enable_sparse_li_c8 = False
    impl._resolve_preprocess_type = lambda _: PreprocessType.NATIVE
    impl.dcp_group = SimpleNamespace(unique_name="sparse-qrep-test")
    return impl


@pytest.mark.parametrize("active", [False, True])
def test_sparse_weights_reload_from_local_source(active):
    impl = make_sparse_impl(active)
    source = torch.arange(24, dtype=torch.float32).view(12, 2)
    impl.kv_b_proj = SimpleNamespace(weight=source.clone(), quant_method=None)
    remote = torch.randn(2, 3, 2)
    gather = Mock(side_effect=lambda x, dim: torch.cat((x, remote), dim))
    with (
        patch.object(sfa_v1, "get_dcp_group", return_value=SimpleNamespace(all_gather=gather)),
        patch.object(sfa_v1.torch_npu, "npu_format_cast", side_effect=lambda x, _: x),
        patch.object(sfa_v1, "maybe_trans_nz", side_effect=lambda x: x),
        patch.object(sfa_v1, "dispose_layer") as dispose,
    ):
        for offset in (0, 0, 200):
            impl.kv_b_proj.weight.copy_(source + offset)
            impl.process_weights_after_loading(torch.float32)
            local = (source + offset).view(2, 6, 2)
            torch.testing.assert_close(impl.W_UV, local[:, 3:].transpose(1, 2))
            torch.testing.assert_close(impl.W_UK_T, local[:, :3])
            if active:
                torch.testing.assert_close(impl.W_UK_T_dcp_qrep, torch.cat((local[:, :3], remote)))
        assert dispose.call_count == (0 if active else 3)
    assert gather.call_count == (3 if active else 0)
    assert impl.indexer.process_weights_after_loading.call_count == 3
    assert impl.q_proj.prepare_local_weight.call_count == (3 if active else 0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 4])
def test_sparse_projection_preserves_tokens_and_uses_group_uk(dtype, tokens):
    impl = make_sparse_impl()
    x = torch.randn(tokens, 4).to(dtype)
    weight = torch.randn(20, 4).to(dtype)
    impl.W_UK_T_dcp_qrep = torch.randn(4, 3, 2).to(dtype)
    impl.q_proj = SimpleNamespace(group_size=2)

    class Projection:
        group_size = 2

        def __call__(self, x):
            return (torch.nn.functional.linear(x, weight),)

    impl.q_proj = Projection()
    with patch.object(
        sfa_v1.torch_npu,
        "npu_transpose_batchmatmul",
        create=True,
        side_effect=lambda a, b, **kw: torch.bmm(a.transpose(0, 1), b).transpose(0, 1),
    ) as bmm:
        q, pe = impl._q_proj_and_k_up_proj(x)
    expected = torch.nn.functional.linear(x, weight).view(tokens, 4, 5)
    torch.testing.assert_close(q, torch.einsum("thp,hpl->thl", expected[..., :3], impl.W_UK_T_dcp_qrep))
    torch.testing.assert_close(pe, expected[..., 3:])
    assert bmm.call_count == int(dtype == torch.bfloat16)


@pytest.mark.parametrize("prefills", [1, 2])
@pytest.mark.parametrize("rank", [0, 1])
def test_sparse_prefill_and_mixed_batch_keep_local_heads_and_kv_gather(prefills, rank):
    impl = make_sparse_impl(rank=rank)
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = prefills
    metadata.num_decode_tokens = 1 if prefills == 1 else 0
    handle = Mock()
    packed = torch.randn(7, 1, 4)
    context = DCPGatherContext(packed, handle, None, (2, 2))
    metadata.dcp_context = SimpleNamespace(gather_context=context, kv_gather_block_table=object())
    q, pe, topk = torch.randn(3, 2, 2), torch.randn(3, 2, 2), torch.tensor([[0], [1], [2]])
    impl._start_dcp_query_gather = Mock(side_effect=AssertionError("Unexpected Q gather"))
    with patch.object(
        sfa_cp.DeviceOperator, "execute_sparse_flash_attention_process", return_value=object()
    ) as execute:
        impl._record_query_gather_context(q, pe, metadata)
        result = impl._execute_sparse_flash_attention_process(q, pe, (), topk, metadata, [1, 3], [4, 7])
    assert result is execute.return_value
    assert execute.call_args.args[1] is q
    assert execute.call_args.args[2] is pe
    assert execute.call_args.args[4] is topk
    assert execute.call_args.kwargs["return_lse"] is False
    assert execute.call_args.kwargs["sparse_mode"] == 3
    assert all(t.is_contiguous() for t in execute.call_args.args[3])
    handle.wait.assert_called_once()
    assert metadata.dcp_context.gather_context is None
    impl._start_dcp_query_gather.assert_not_called()


@pytest.mark.parametrize("dcp", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 128])
def test_sparse_decode_matches_explicit_selected_attention(dcp, interleave):
    torch.manual_seed(42)
    tokens, heads = 3, 2 * dcp
    x, weight = torch.randn(tokens, 4, dtype=torch.float64), torch.randn(heads * 5, 4, dtype=torch.float64)
    uk, uv = torch.randn(heads, 3, 2, dtype=torch.float64), torch.randn(heads, 2, 3, dtype=torch.float64)
    cache, rope = torch.randn(7, 2, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64)
    projected = torch.nn.functional.linear(x, weight).view(tokens, heads, 5)
    group_q = torch.einsum("thp,hpl->thl", projected[..., :3], uk)
    group_pe = projected[..., 3:]
    # Selected global positions are causal for query positions 4, 5, 6.
    topk = torch.tensor([[0, 1, -1], [1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    scale = 5**-0.5

    def sparse_attention(q, pe, kv, kr, indices):
        out, lses = [], []
        for t in range(tokens):
            ids = indices[t][indices[t] >= 0].long()
            if ids.numel() == 0:
                out.append(torch.zeros_like(q[t]))
                lses.append(torch.full((heads, 1), -torch.inf, dtype=q.dtype))
                continue
            scores = (q[t] @ kv[ids].T + pe[t] @ kr[ids].T) * scale
            out.append(scores.softmax(-1) @ kv[ids])
            lses.append(scores.logsumexp(-1, keepdim=True))
        return torch.stack(out), torch.stack(lses)

    partials, lses, impls, caches, indices = [], [], [], [], []
    for rank in range(dcp):
        impl = make_sparse_impl(dcp=dcp, rank=rank)
        impl._dcp_interleave_size = interleave
        impl._dcp_index_topk = 3
        impl._remap_order = torch.arange(3, dtype=torch.float32)
        impl._remap_invalid_index = torch.tensor(-1.0)
        local_ids = [i for i in range(7) if i // interleave % dcp == rank]
        local_cache = (cache[local_ids], rope[local_ids])
        mapped = impl._remap_sparse_indices(topk)
        out, lse = sparse_attention(group_q, group_pe, *local_cache, mapped)
        partials.append(out)
        lses.append(lse)
        impls.append(impl)
        caches.append(local_cache)
        indices.append(mapped)
    merged = (torch.stack(partials) * torch.stack(lses).softmax(0)).sum(0)
    expected = []
    for t in range(tokens):
        ids = topk[t][topk[t] >= 0].long()
        full_k = torch.cat((torch.einsum("jl,hpl->jhp", cache[ids], uk), rope[ids, None].expand(-1, heads, -1)), -1)
        full_v = torch.einsum("jl,hlv->jhv", cache[ids], uv)
        probs = (torch.einsum("hd,jhd->hj", projected[t], full_k) * scale).softmax(-1)
        expected.append(torch.einsum("hj,jhv->hv", probs, full_v))
    expected = torch.stack(expected)

    for rank, impl in enumerate(impls):
        local = slice(rank * 2, rank * 2 + 2)
        for active in (False, True):
            impl.dcp_q_replicate = active
            metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
            metadata.num_prefills = 0
            metadata.dcp_context = SimpleNamespace(
                gather_context=None, seq_lens=[len(caches[rank][0])], block_table=object()
            )
            query, pe = (group_q, group_pe) if active else (group_q[:, local], group_pe[:, local])
            handle = Mock()
            context = DCPGatherContext(torch.cat((group_q, group_pe), -1), handle, None, (2, 2))
            impl._start_dcp_query_gather = Mock(return_value=context)

            def execute(_impl, q, p, kv, idx, *_args, rank=rank, **kw):
                assert kw["sparse_mode"] == 0 and kw["return_lse"]
                torch.testing.assert_close(q, group_q)
                torch.testing.assert_close(p, group_pe)
                torch.testing.assert_close(idx, indices[rank])
                out, lse = sparse_attention(q, p, *kv, idx)
                return out, lse.transpose(0, 1), torch.ones_like(lse.transpose(0, 1))

            def merge(out, lse, size, dim, name, rank=rank, local=local):
                torch.testing.assert_close(out, partials[rank])
                torch.testing.assert_close(lse, lses[rank])
                assert size == dcp and dim == 1
                return merged[:, local]

            with (
                patch.object(sfa_cp.DeviceOperator, "execute_sparse_flash_attention_process", side_effect=execute),
                patch("torch.ops.vllm.sfa_dcp_a2a_fused", side_effect=merge) as collective,
                patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=False),
            ):
                impl._record_query_gather_context(query, pe, metadata)
                result = impl._execute_sparse_flash_attention_process(query, pe, caches[rank], topk, metadata, [3], [7])
            actual = torch.einsum("thl,hlv->thv", result, uv[local])
            torch.testing.assert_close(actual, expected[:, local], atol=1e-10, rtol=1e-10)
            assert impl._start_dcp_query_gather.call_count == int(not active)
            assert handle.wait.call_count == int(not active)
            collective.assert_called_once()
            assert metadata.dcp_context.gather_context is None


@pytest.mark.parametrize("case", ["valid", "tokens", "no_dcp", "direct", "nope", "mlapo", "sfa_c8", "li_c8"])
def test_sparse_qrep_backend_contract(case):
    impl = make_sparse_impl()
    impl.supports_dcp = case != "no_dcp"
    impl._parallel_query_gather_dim = lambda: 0 if case == "tokens" else 1
    if case == "direct":
        impl.q_lora_rank = None
    if case == "nope":
        impl.qk_rope_head_dim = 0
    impl.enable_mlapo = case == "mlapo"
    impl.enable_sparse_sfa_c8 = case == "sfa_c8"
    impl.enable_sparse_li_c8 = case == "li_c8"
    # Q-LoRA is validated by the existing native preprocessing path.
    if case in ("valid", "direct"):
        impl._validate_dcp_q_replicate()
    else:
        with pytest.raises(ValueError, match="dcp_q_replicate"):
            impl._validate_dcp_q_replicate()


@pytest.mark.parametrize("case", ["heads", "pending_gather"])
def test_sparse_qrep_rejects_inconsistent_decode_state(case):
    impl = make_sparse_impl()
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = 0
    metadata.dcp_context = SimpleNamespace(gather_context=object() if case == "pending_gather" else None)
    q = torch.zeros(2, 2 if case == "heads" else 4, 2)
    error = ValueError if case == "heads" else RuntimeError
    with pytest.raises(error, match="group head|pending Q gather"):
        impl._execute_sparse_flash_attention_process(q, q, (), None, metadata, None, None)


@pytest.mark.parametrize("dcp,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_sparse_prefill_projects_local_weights_before_bmm(dcp, rank, tokens, dtype):
    impl = make_sparse_impl(dcp=dcp, rank=rank)
    x = torch.randn(tokens, 4, dtype=dtype)
    weight = torch.randn(2 * dcp * 5, 4, dtype=dtype)
    local_weight = weight.chunk(dcp)[rank].contiguous()
    impl.W_UK_T = torch.randn(2, 3, 2, dtype=dtype)

    # No group UK buffer is available: prefill must not use it even transiently.
    class Projection:
        group_size = dcp

        def __call__(self, x):
            raise AssertionError("Prefill must not compute group-wide Q before slicing")

        def forward_local(self, x):
            return (torch.nn.functional.linear(x, local_weight),)

    impl.q_proj = Projection()
    with patch.object(
        sfa_v1.torch_npu,
        "npu_transpose_batchmatmul",
        create=True,
        side_effect=lambda a, b, **kw: torch.bmm(a.transpose(0, 1), b).transpose(0, 1),
    ):
        q, pe = impl._q_proj_and_k_up_proj(x, local_q=True)
    expected = torch.nn.functional.linear(x, local_weight).view(tokens, 2, 5)
    torch.testing.assert_close(q, torch.einsum("thp,hpl->thl", expected[..., :3], impl.W_UK_T))
    torch.testing.assert_close(pe, expected[..., 3:])
