# SPDX-License-Identifier: Apache-2.0
"""Compare DCP history/current assembly with dense attention on CPU.

The collectives/kernel are emulated; metadata and merge/projection orchestration
are the production Python functions. This does not qualify NPU kernel precision.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_helpers(combine):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/common_cp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {
            "merge_flash_attention_output",
            "exchange_flash_attention_output",
            "combine_flash_attention_output",
        }
    ]
    scope = {"torch": torch, "fused_sfa_dcp_lse_combine": combine}
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


def combine(recv, dim, scatter_dim, local_output=None, local_lse=None):
    assert scatter_dim == 1
    outputs = recv[..., :dim].transpose(1, 2)
    lses = recv[..., dim:].transpose(1, 2)
    if local_output is not None:
        outputs = torch.cat((outputs, local_output[None]), dim=0)
        lses = torch.cat((lses, local_lse[None]), dim=0)
    finite = torch.isfinite(lses)
    lses = torch.where(finite, lses, -torch.inf)
    maximum = lses.amax(0)
    weights = torch.where(finite, torch.exp(lses - maximum), 0)
    denominator = weights.sum(0).clamp_min(torch.finfo(torch.float32).tiny)
    outputs = torch.where(finite, outputs, 0)
    return (outputs * weights).sum(0) / denominator


def attention(q, k, v):
    """Single query, with the Flash operator's empty-KV sentinel."""
    if k.shape[0] == 0:
        return torch.zeros(q.shape[0], v.shape[-1]), torch.full((q.shape[0],), torch.inf)
    scores = torch.einsum("hd,shd->hs", q, k) / q.shape[-1] ** 0.5
    return torch.einsum("hs,shv->hv", scores.softmax(-1), v), scores.logsumexp(-1)


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 16])
@pytest.mark.parametrize("expanded", [False, True])
def test_history_current_matches_dense_with_empty_ranks_and_block_crossing(monkeypatch, dcp_size, interleave, expanded):
    torch.manual_seed(8)
    local_heads = 2
    heads = local_heads * dcp_size
    # First request crosses two 16-token blocks; another has no history.
    seq_lens, query_lens = [37, 3, 17], [4, 3, 1]
    tokens = sum(query_lens)
    dim, value_dim = 512, 128
    queries = torch.randn(tokens, heads, dim) * 0.2
    projection = torch.randn(heads, dim, value_dim) / dim**0.5
    keys = [torch.randn(length, heads, dim) * 0.2 for length in seq_lens]
    values = [torch.randn(length, heads, dim) for length in seq_lens]
    history_out = torch.empty(dcp_size, tokens, heads, dim)
    history_lse = torch.empty(dcp_size, heads, tokens)
    current_out = torch.empty(tokens, heads, value_dim if expanded else dim)
    current_lse = torch.empty(heads, tokens)
    expected = torch.empty_like(current_out)
    helpers = load_helpers(combine)

    offset = 0
    for request, (seq, qlen) in enumerate(zip(seq_lens, query_lens)):
        history = seq - qlen
        owners = torch.arange(history) // interleave % dcp_size
        for rank in range(dcp_size):
            indices = torch.nonzero(owners == rank).flatten()
            for position in range(qlen):
                out, lse = attention(queries[offset + position], keys[request][indices], values[request][indices])
                history_out[rank, offset + position] = out
                history_lse[rank, :, offset + position] = lse
        for position in range(qlen):
            end = history + position + 1
            value = values[request]
            if expanded:
                value = torch.einsum("shd,hdv->shv", value, projection)
            current_out[offset + position], current_lse[:, offset + position] = attention(
                queries[offset + position],
                keys[request][history:end],
                value[history:end],
            )
            expected[offset + position], _ = attention(queries[offset + position], keys[request][:end], value[:end])
        offset += qlen

    for rank in range(dcp_size):
        owned = slice(rank * local_heads, (rank + 1) * local_heads)

        def exchange(output, lse, size, scatter_dim, group, defer_combine, raw_row_words=257, rank=rank, owned=owned):
            torch.testing.assert_close(output, history_out[rank])
            torch.testing.assert_close(lse, history_lse[rank].T[..., None])
            assert size == dcp_size and scatter_dim == 1 and defer_combine
            return torch.cat((history_out[:, :, owned].transpose(1, 2), history_lse[:, owned, :, None]), dim=-1)

        monkeypatch.setattr(torch.ops.vllm, "sfa_dcp_a2a_fused", exchange, raising=False)
        result = helpers.merge_flash_attention_output(
            history_out[rank],
            history_lse[rank],
            SimpleNamespace(world_size=dcp_size, unique_name="cpu"),
            current_output=current_out[:, owned],
            current_lse=current_lse[owned],
            value_projection=projection[owned] if expanded else None,
        )
        torch.testing.assert_close(result, expected[:, owned], rtol=3e-5, atol=2e-6)


def test_noncausal_draft_merges_full_local_sequence_and_invalid_graph_rows(monkeypatch):
    torch.manual_seed(3)
    size, tokens, heads, dim = 4, 3, 8, 64
    q = torch.randn(tokens, heads, dim)
    k = torch.randn(5, heads, dim)
    v = torch.randn(5, heads, dim)
    outputs = torch.zeros(size, tokens, heads, dim)
    lses = torch.full((size, heads, tokens), torch.inf)
    expected = torch.zeros(tokens, heads, dim)
    for token in range(tokens - 1):
        expected[token], _ = attention(q[token], k, v)
        for rank in range(size):
            indices = torch.nonzero(torch.arange(5) // 4 % size == rank).flatten()
            outputs[rank, token], lses[rank, :, token] = attention(q[token], k[indices], v[indices])
    helpers = load_helpers(combine)
    for rank in range(size):
        owned = slice(rank * 2, (rank + 1) * 2)

        def exchange(*args, owned=owned, **kwargs):
            return torch.cat((outputs[:, :, owned].transpose(1, 2), lses[:, owned, :, None]), dim=-1)

        monkeypatch.setattr(torch.ops.vllm, "sfa_dcp_a2a_fused", exchange, raising=False)
        result = helpers.merge_flash_attention_output(
            outputs[rank], lses[rank], SimpleNamespace(world_size=size, unique_name="cpu")
        )
        torch.testing.assert_close(result, expected[:, owned], rtol=1e-5, atol=1e-6)
        assert torch.equal(result[-1], torch.zeros_like(result[-1]))
