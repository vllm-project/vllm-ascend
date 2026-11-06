# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.thinking_budget import NPUThinkingBudgetStateHolder as ThinkingBudgetStateHolder

START = 90
END = 91
VOCAB_SIZE = 128


def make_holder(budget, num_spec_tokens=3):
    holder = ThinkingBudgetStateHolder(
        SimpleNamespace(reasoning_start_token_ids=[START], reasoning_end_token_ids=[END]),
        max_num_seqs=2,
        num_spec_tokens=num_spec_tokens,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    holder.sync_batch(
        SimpleNamespace(
            removed=[],
            moved=[],
            added=[(0, SamplingParams(thinking_token_budget=budget), [1, START], [])],
        )
    )
    return holder


def process(holder, output, draft):
    holder.update_state([output, []], [draft, [20]])
    sampling = SimpleNamespace(
        no_penalties=True,
        bad_words_token_ids={},
        output_token_ids=[output, []],
        spec_token_ids=[draft, [20]],
        allowed_token_ids_mask=None,
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        thinking_budget_state_holder=holder,
    )
    metadata = SimpleNamespace(num_draft_tokens=[len(draft), 1])
    sampler = AscendRejectionSampler.__new__(AscendRejectionSampler)
    logits = torch.zeros(len(draft) + 1, VOCAB_SIZE)
    return sampler.apply_logits_processors(logits, sampling, metadata)


@pytest.mark.parametrize("committed", [1, 2, 3])
def test_budget_forcing_tracks_committed_tokens_not_rejected_drafts(committed):
    holder = make_holder(3)
    # Two possible continuations are verified but only committed history on the
    # next call may consume budget. The unbudgeted second request stays intact.
    process(holder, [10], [11, 12, 13])
    actual = process(holder, [10] * committed, [11, 12, 13])
    assert actual[3, END] == 0
    assert actual[3 - committed, END] > 0


def test_zero_budget_forces_first_draft_without_touching_plain_request():
    holder = make_holder(0)
    logits = process(holder, [], [10, 11, 12])
    assert logits[0, END] > 0
    assert torch.count_nonzero(logits[3]) == 0


def test_natural_end_leaves_answer_logits_unchanged():
    holder = make_holder(16)
    logits = process(holder, [10, END, 11], [12, 13, 14])
    assert torch.count_nonzero(logits) == 0


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_forced_end_wins_greedy_and_random_sampling(seed):
    holder = make_holder(0)
    logits = process(holder, [], [10, 11, 12])
    assert logits[0].argmax().item() == END
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.multinomial(logits[0].softmax(-1), 32, replacement=True, generator=generator)
    assert samples.tolist() == [END] * 32


@pytest.mark.parametrize("draft_length,committed", [(k, n) for k in (1, 3, 5) for n in range(1, k + 2)])
def test_budget_boundary_forces_draft_or_bonus_not_other_request(draft_length, committed):
    # Draft verification follows at least one committed prefill sample.
    holder = make_holder(draft_length + 1, num_spec_tokens=draft_length)
    drafts = [[11] * draft_length, [20]]
    holder.update_state([[10] * committed, []], drafts)
    bonus = holder.apply_to_logits(torch.zeros(2, VOCAB_SIZE), predict_bonus_token=True, spec_token_ids=drafts)
    target = holder.apply_to_logits(
        torch.zeros(draft_length + 1, VOCAB_SIZE), predict_bonus_token=False, spec_token_ids=drafts
    )
    remaining = draft_length + 1 - committed
    if remaining == draft_length:
        assert bonus[0].argmax().item() == END
        assert torch.count_nonzero(target) == 0
    else:
        assert target[remaining].argmax().item() == END
        assert torch.count_nonzero(bonus) == 0
    assert torch.count_nonzero(bonus[1]) == 0
    assert torch.count_nonzero(target[-1]) == 0
