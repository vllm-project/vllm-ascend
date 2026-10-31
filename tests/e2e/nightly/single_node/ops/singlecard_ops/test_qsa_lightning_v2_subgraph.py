# SPDX-License-Identifier: Apache-2.0
"""NPU QSA selector -> real GBSA comparison, without loading model weights.

pytest -o addopts= -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_qsa_lightning_v2_subgraph.py
This validates the prepared Indexer/attention boundary, not projection, cache
compression, a full model, or graph replay. Those need separate validation.
"""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401
from vllm_ascend.models.qwen4_exp.gbsa import qsa_gbsa_attention  # noqa: E402
from vllm_ascend.models.qwen4_exp.lightning_indexer import (  # noqa: E402
    qsa_select_paged_tokens_lightning,
    qsa_select_paged_tokens_lightning_v2,
)
from vllm_ascend.models.qwen4_exp.qsa import AscendQSAIndexer  # noqa: E402

from .test_qsa_lightning_v2_selection import (  # noqa: E402
    COMPRESS_RATIO,
    GROUP_TOPK,
    HEAD_DIM,
    OUTPUT_WIDTH,
    PACKED_WIDTH,
    TOKEN_TOPK,
    assert_selection,
    make_selection_case,
    row_reference_scores,
)

MAIN_PAGE_SIZE = 128
MAIN_QUERY_HEADS = 4
MAIN_KV_HEADS = 2
ATTENTION_ATOL = 0.03
ATTENTION_RTOL = 0.03
TAIL_VALUE = 256


def make_main_attention(case, uniform_tail):
    generator = torch.Generator().manual_seed(2031)
    pages_per_request = math.ceil(max(case.lengths_cpu) / MAIN_PAGE_SIZE)
    pages = len(case.lengths_cpu) * pages_per_request
    table = torch.randperm(pages, generator=generator).reshape(len(case.lengths_cpu), pages_per_request).to(torch.int32)
    keys = torch.zeros(pages, MAIN_PAGE_SIZE, MAIN_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    logical_keys = []
    logical_values = []
    for request, length in enumerate(case.lengths_cpu):
        k = torch.randn(length, MAIN_KV_HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
        v = torch.randn(length, MAIN_KV_HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
        if uniform_tail:
            # Uniform logits plus a conspicuous tail expose missing residual
            # tokens even in a ~2048-token attention window.
            request_positions = case.positions_cpu[case.requests_cpu == request]
            for position in request_positions.tolist():
                start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
                v[start:position + 1].fill_(TAIL_VALUE)
        logical_keys.append(k)
        logical_values.append(v)
        tokens = torch.arange(length)
        physical = table[request, tokens // MAIN_PAGE_SIZE].long()
        keys[physical, tokens % MAIN_PAGE_SIZE] = k
        values[physical, tokens % MAIN_PAGE_SIZE] = v
    shape = (len(case.query), MAIN_QUERY_HEADS, HEAD_DIM)
    query = torch.zeros(shape, dtype=torch.bfloat16) if uniform_tail else torch.randn(shape, generator=generator).to(torch.bfloat16)
    device = case.query.device
    return (query.to(device), keys.to(device), values.to(device), table.to(device),
            query, logical_keys, logical_values)


def reference_attention(case, query, logical_keys, logical_values):
    outputs = []
    heads = torch.arange(MAIN_QUERY_HEADS) // (MAIN_QUERY_HEADS // MAIN_KV_HEADS)
    for row, position in enumerate(case.positions_cpu.tolist()):
        request = int(case.requests_cpu[row])
        scores = row_reference_scores(case, row)
        groups = scores.topk(min(GROUP_TOPK, len(scores))).indices
        tail_start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
        tail = torch.arange(tail_start, position + 1)
        # Model the established GBSA budget explicitly, independently of its
        # implementation: a tail occupies one of the 512 sparse block slots.
        if len(tail) and len(groups) == GROUP_TOPK:
            groups = groups[:-1]
        tokens = torch.cat(((groups[:, None] * COMPRESS_RATIO + torch.arange(COMPRESS_RATIO)).flatten(), tail))
        assert len(tokens) and tokens.unique().numel() == len(tokens)
        assert (tokens <= position).all()
        k = logical_keys[request][tokens][:, heads].float()
        v = logical_values[request][tokens][:, heads].float()
        logits = torch.einsum("hd,shd->hs", query[row].float(), k) / math.sqrt(HEAD_DIM)
        outputs.append(torch.einsum("hs,shd->hd", logits.softmax(-1), v))
    return torch.stack(outputs)


@pytest.mark.parametrize("heads,page_size", [(4, 192), (12, 384)])
@pytest.mark.parametrize("uniform_tail", [False, True])
def test_old_and_native_selection_feed_same_gbsa(heads, page_size, uniform_tail):
    case = make_selection_case((3, 7, 2049, 2053), (1, 3, 2, 2), heads=heads, page_size=page_size)
    query, keys, values, table, query_cpu, logical_keys, logical_values = make_main_attention(case, uniform_tail)
    expected = reference_attention(case, query_cpu, logical_keys, logical_values)
    outputs = []
    for selector in (qsa_select_paged_tokens_lightning, qsa_select_paged_tokens_lightning_v2):
        selected = selector(*case.arguments())
        assert_selection(case, selected)
        output = torch.full_like(query, float("nan"))
        result = qsa_gbsa_attention(
            query, keys, values, selected[:, :PACKED_WIDTH], table,
            case.requests, case.positions, case.lengths, output,
        )
        assert result is output
        assert torch.isfinite(output).all()
        outputs.append(output.float().cpu())
    for actual in outputs:
        torch.testing.assert_close(actual, expected, atol=ATTENTION_ATOL, rtol=ATTENTION_RTOL)
    torch.testing.assert_close(outputs[0], outputs[1], atol=ATTENTION_ATOL, rtol=ATTENTION_RTOL)


def test_qsa_entry_padding_and_mtp_buffer_reuse(monkeypatch):
    case = make_selection_case((7, 2053), (2, 3))
    rows = len(case.query)
    extra_rows = 2
    device = case.query.device
    query = torch.cat((case.query, torch.zeros_like(case.query[:extra_rows])))
    metadata = SimpleNamespace(
        block_table=torch.cat((case.table, torch.zeros_like(case.table[:1]))),
        token_to_req=torch.cat((case.requests, torch.full((extra_rows,), 2, device=device))),
        logical_positions=torch.cat((case.positions, torch.full((extra_rows,), -1, device=device))),
        seq_lens=torch.cat((case.lengths, torch.ones(1, dtype=torch.int32, device=device))),
        query_start_loc=torch.cat((case.boundaries, case.boundaries[-1:] + extra_rows)),
        num_actual_tokens=rows + extra_rows,
        qsa_num_live_requests=len(case.lengths),
        qsa_num_live_tokens=rows,
    )
    # Exercise the actual _select method without constructing/loading a model.
    indexer = object.__new__(AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=case.cache)
    indexer.token_topk = TOKEN_TOPK
    indexer.compress_ratio = COMPRESS_RATIO
    buffer = torch.full((len(query), OUTPUT_WIDTH), 123, dtype=torch.int32, device=device)
    selected = indexer._select(query, metadata, buffer)
    assert selected is buffer
    assert_selection(case, selected[:rows])
    assert (selected[rows:] == -1).all()

    # Mirror MTP's target-row compaction and execute the real forward skip path.
    keep = torch.tensor([1, rows - 1], dtype=torch.int64, device=device)
    saved = selected.index_select(0, keep).clone()
    selected[:len(keep)].copy_(saved)
    reused = selected[:len(keep)]
    calls = []
    reuse_metadata = SimpleNamespace(num_actual_tokens=len(keep))
    monkeypatch.setattr(indexer, "_metadata", lambda: (reuse_metadata, reuse_metadata))
    monkeypatch.setattr(indexer, "project_qk", lambda hidden, positions: (hidden, hidden))
    monkeypatch.setattr(indexer, "_update_and_compress", lambda *args: calls.append("update_cache"))
    monkeypatch.setattr(indexer, "_select", lambda *args: pytest.fail("MTP reuse must not invoke selection"))
    indexer.skip_topk = True
    result = indexer.forward(torch.zeros(len(keep), HEAD_DIM, device=device), torch.arange(len(keep), device=device), reused)
    assert result is reused
    torch.testing.assert_close(result, saved)
    assert calls == ["update_cache"]
