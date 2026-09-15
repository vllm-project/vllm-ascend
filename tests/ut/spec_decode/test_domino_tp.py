# SPDX-License-Identifier: Apache-2.0
"""CPU checks of global-vocabulary correction with simulated TP shards."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ut.spec_decode.test_domino_alignment import _load_file
from tests.ut.spec_decode.test_domino_alignment import domino_modules as _domino_modules

domino_modules = _domino_modules


@pytest.mark.parametrize(
    "target,draft,pp,valid", [(1, 1, 1, True), (2, 2, 1, True), (2, None, 1, True), (2, 1, 1, False), (2, 2, 2, False)]
)
def test_parallel_config(domino_modules, target, draft, pp, valid):
    proposer_cls, _, _ = domino_modules
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=target, pipeline_parallel_size=pp),
        speculative_config=SimpleNamespace(draft_tensor_parallel_size=draft),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    if valid:
        proposer_cls(config, torch.device("cpu"))
    else:
        with pytest.raises(ValueError, match="parallel"):
            proposer_cls(config, torch.device("cpu"))


@pytest.mark.parametrize("vocab", [67, 151936, 248320])
def test_chunked_global_gi_table(domino_modules, monkeypatch, vocab):
    _, _, correction = domino_modules
    torch.manual_seed(11)
    weight = torch.randn(vocab, 4)
    gru_weight = torch.randn(6, 4)
    calls = []

    def embed(ids):
        calls.append(ids.numel())
        return weight[ids]

    # The local shard is deliberately unusable: only the global-ID API is valid.
    embedding = SimpleNamespace(tp_size=2, weight=torch.full((1, 4), float("nan")))
    model = SimpleNamespace(embed_input_ids=embed)
    table = correction._build_domino_gi_table(model, embedding, gru_weight, vocab)
    torch.testing.assert_close(table, torch.nn.functional.linear(weight, gru_weight))
    assert table.shape == (vocab, 6)
    assert sum(calls) == vocab and max(calls) <= correction._GI_TABLE_CHUNK_SIZE


@pytest.mark.parametrize("candidate_topk", [0, 64])
@pytest.mark.parametrize("scale,cap", [(1.0, None), (0.5, 2.0)])
def test_global_logits_and_rollout(domino_modules, monkeypatch, candidate_topk, scale, cap):
    _, _, correction = domino_modules
    helpers = _load_file("tp_helpers", "tests/ut/spec_decode/test_domino_correction.py")
    torch.manual_seed(15)
    reference = helpers.StubDominoModel(vocab_size=67).eval()
    hidden = torch.randn(8, reference.hidden_size)
    bonus = torch.tensor([4, 66])

    def transform(logits):
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits * scale

    reference.logits_processor = lambda head, h: transform(head(h))
    expected = correction.domino_correct(hidden, 4, reference, bonus, candidate_topk)
    padded = torch.nn.functional.pad(reference.lm_head.weight.detach(), (0, 0, 0, 5))
    # Padding scores are huge: they must be removed before top-k / argmax.
    local_logits = [torch.nn.functional.linear(hidden, shard) for shard in padded.chunk(2)]
    local_logits[-1][:, -5:] = 10000
    for rank in range(2):
        model = copy.deepcopy(reference)
        model.lm_head = nn.Linear(reference.hidden_size, 36, bias=False)
        model.lm_head.tp_size = 2
        model.lm_head.quant_method = SimpleNamespace(apply=lambda head, h, rank=rank: local_logits[rank].clone())
        model.logits_processor = SimpleNamespace(org_vocab_size=67, scale=scale, soft_cap=cap)
        distributed = ModuleType("vllm.distributed")

        def gather(logits, dim, rank=rank):
            assert torch.equal(logits, local_logits[rank])
            return torch.cat(local_logits, dim=dim)

        distributed.tensor_model_parallel_all_gather = gather
        monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
        actual = correction._domino_base_logits(model, hidden)
        torch.testing.assert_close(actual, transform(reference.lm_head(hidden)))
        got = correction.domino_correct(hidden, 4, model, bonus, candidate_topk)
        assert torch.equal(got, expected)
