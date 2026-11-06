# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import pytest
import torch
import xgrammar as xgr
from vllm.v1.core.sched.output import GrammarOutput

from vllm_ascend._310p.worker.v2.rejection_sampler import RejectionSampler310V2
from vllm_ascend._310p.worker.v2.structured_outputs import apply_grammar_bitmask

YES_NO_GRAMMAR = 'root ::= "yes" | "no"'


def test_bitmask_reorders_fixed_stride_for_mixed_variable_drafts():
    # Runner order differs from scheduler order; the plain request has no mask.
    batch = SimpleNamespace(
        req_ids=["plain", "short", "long"],
        cu_num_logits_np=np.array([0, 1, 2, 5], dtype=np.int32),
        num_draft_tokens=2,
    )
    masks: np.ndarray = np.zeros((6, 2), dtype=np.int32)
    for row, token in enumerate([31, 34, 0, 5, 6, 7]):
        masks.view(np.uint32)[row, token // 32] = np.uint32(1 << (token % 32))
    grammar = GrammarOutput(["long", "short"], masks)
    logits = torch.ones(5, 35, dtype=torch.float16)

    apply_grammar_bitmask(logits, batch, grammar)

    assert logits[0].eq(1).all()
    assert logits[1:].argmax(dim=-1).tolist() == [5, 31, 34, 0]
    assert logits[1:].isfinite().sum(dim=-1).tolist() == [1, 1, 1, 1]


def test_bitmask_without_mtp():
    batch = SimpleNamespace(req_ids=["grammar"], cu_num_logits_np=np.array([0, 1]), num_draft_tokens=0)
    logits = torch.ones(1, 3)
    apply_grammar_bitmask(logits, batch, GrammarOutput(["grammar"], np.full((1, 1), 2, dtype=np.int32)))
    assert logits.tolist() == [[-torch.inf, 1, -torch.inf]]


def test_rejection_stops_at_first_grammar_invalid_draft():
    num_drafts, limit = 3, 1
    sampler = SimpleNamespace(
        max_num_reqs=1,
        apply_sampling_params=lambda logits, *args, **kwargs: logits,
        _batch_max_num_logprobs=lambda _: None,
        _temperature_np=np.array([0.0]),
    )
    input_ids = torch.ones(num_drafts + 1, dtype=torch.int32)
    indices = torch.arange(num_drafts + 1)
    offsets = torch.tensor([0, num_drafts + 1], dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0], dtype=np.int32),
        expanded_idx_mapping=torch.zeros(num_drafts + 1, dtype=torch.int32),
        input_ids_cpu=input_ids,
        input_ids=input_ids,
        logits_indices_np=indices.numpy(),
        logits_indices=indices,
        cu_num_logits_np=offsets.numpy(),
        cu_num_logits=offsets,
        seq_lens_np=np.array([8]),
        prefill_len_np=np.array([4]),
        req_ids=["grammar"],
        num_draft_tokens=num_drafts,
    )
    logits = torch.ones(num_drafts + 1, 3)
    masks = np.full((num_drafts + 1, 1), 2, dtype=np.int32)
    masks[limit] = 4
    masks[limit + 1 :] = -1
    apply_grammar_bitmask(logits, batch, GrammarOutput(["grammar"], masks))

    with (
        patch("vllm_ascend._310p.worker.v2.rejection_sampler.is_pin_memory_available", return_value=False),
        patch.multiple(torch.npu, Stream=DEFAULT, Event=DEFAULT, current_stream=DEFAULT, stream=DEFAULT),
    ):
        runner = RejectionSampler310V2(
            sampler,
            MagicMock(num_speculative_tokens=num_drafts, rejection_sample_method="standard"),
            torch.device("cpu"),
        )
        output = runner(logits, batch, None)

    assert output.num_sampled.tolist() == [limit + 1]
    assert output.num_rejected.tolist() == [num_drafts - limit]
    assert output.sampled_token_ids[0].tolist() == [1] * limit + [2] + [-1] * (num_drafts - limit)


@pytest.fixture(scope="module")
def yes_no_xgrammar():
    vocab = ["<eos>", "yes", "no", "red", "tok0", "tok1"]
    info = xgr.TokenizerInfo(
        encoded_vocab=vocab,
        vocab_type=xgr.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[0],
    )
    compiled = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(YES_NO_GRAMMAR)
    bitmask = xgr.allocate_token_bitmask(1, len(vocab))
    assert xgr.GrammarMatcher(compiled).fill_next_token_bitmask(bitmask)
    return vocab, compiled, bitmask


def test_xgrammar_yes_no_mask_matches_and_decodes(yes_no_xgrammar):
    """Real xgrammar yes/no constraint through apply_grammar_bitmask + accept path."""
    vocab, compiled, bitmask = yes_no_xgrammar
    vocab_size = len(vocab)
    token = {piece: idx for idx, piece in enumerate(vocab)}
    masks = bitmask.detach().cpu().numpy().copy()

    words = masks.view(np.uint32)[0]
    allowed = [tid for tid in range(vocab_size) if words[tid // 32] & (np.uint32(1) << (tid % 32))]
    assert set(allowed) == {token["yes"], token["no"]}

    logits = torch.randn(1, vocab_size)
    expected = logits.float().clone()
    xgr.apply_token_bitmask_inplace(expected, torch.from_numpy(masks))
    batch = SimpleNamespace(req_ids=["g"], cu_num_logits_np=np.array([0, 1], dtype=np.int32), num_draft_tokens=0)
    apply_grammar_bitmask(logits, batch, GrammarOutput(["g"], masks))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert logits.argmax(dim=-1).item() in allowed

    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(token["yes"])
    assert matcher.accept_token(0)
    assert matcher.is_terminated()
