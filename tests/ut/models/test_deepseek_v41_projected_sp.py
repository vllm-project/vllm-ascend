# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure TP projections must match projecting the gathered hidden states."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.projected_sp import project_and_gather, projected_sp_supported


@pytest.mark.parametrize("tokens", [0, 1, 7, 8])
@pytest.mark.parametrize("ratio,index_source", [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)])
def test_tp_projection_equivalence(tokens, ratio, index_source):
    tp = 4
    shard_rows = (tokens + tp - 1) // tp
    rng = torch.Generator().manual_seed(12)
    hidden = torch.randn(tp * shard_rows, 8, dtype=torch.bfloat16, generator=rng)

    def linear(width, dtype):
        weight = torch.randn(8, width, dtype=dtype, generator=rng)
        return lambda x: x @ weight

    attn = SimpleNamespace(
        role=SimpleNamespace(is_kv_source=ratio > 0, is_index_source=index_source, compress_ratio=ratio),
        wq_a=linear(4, torch.bfloat16),
        wkv=linear(2, torch.bfloat16),
        compressor=SimpleNamespace(
            wkv=linear(2, torch.float32 if ratio == 2 else torch.bfloat16), wgate=linear(2, torch.float32)
        ),
        indexer=SimpleNamespace(project_weights=lambda x: x[:, :1].float() * 0.125),
    )
    expected = {"q_a": attn.wq_a(hidden), "swa_kv": attn.wkv(hidden)}
    if ratio == 1:
        expected["compressor_kv"] = attn.compressor.wkv(hidden)
    if ratio == 2:
        expected["compressor_kv"] = attn.compressor.wkv(hidden.float())
        expected["compressor_score"] = attn.compressor.wgate(hidden.float())
    if index_source:
        expected["indexer_weights"] = attn.indexer.project_weights(hidden)

    # Each collective concatenates the TP ranks' rows in rank order.
    groups = {}
    for value in expected.values():
        groups.setdefault(value.dtype, []).append(value)
    packed = [torch.cat(group, dim=-1) for group in groups.values()]
    for rank in range(tp):
        rows = slice(rank * shard_rows, (rank + 1) * shard_rows)
        collectives = iter(packed)

        def gather(local, collectives=collectives, rows=rows):
            full = next(collectives)
            torch.testing.assert_close(local, full[rows])
            return full

        actual = project_and_gather(attn, hidden[rows], tokens, gather)
        assert next(collectives, None) is None
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            torch.testing.assert_close(value, expected[name][:tokens])


@pytest.mark.parametrize("pcp", [1, 2])
@pytest.mark.parametrize("draft", [False, True])
def test_dspark_target_only(pcp, draft):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=SimpleNamespace(use_dspark=lambda: True),
        lora_config=None,
    )
    assert projected_sp_supported(
        config, SimpleNamespace(dsv41_projected_sp=True), is_draft_layer=draft, batch_invariant=False
    ) is (not draft)
