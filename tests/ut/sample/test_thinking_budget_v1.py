# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 thinking-budget forced end under speculative decoding.

Adapted from https://github.com/vllm-project/vllm/pull/61005.

The harness drives ``ThinkingBudgetStateHolder`` the way V1 calls it
(``update_state``, the bonus-row call in ``Sampler``, then the target-row call in
``RejectionSampler``) and verifies drafts greedily, so no GPU is needed.
"""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from vllm_ascend.sample.thinking_budget import NPUThinkingBudgetStateHolder as ThinkingBudgetStateHolder

THINK_START = 100
# A 6-token forced end sequence: a transition phrase plus the parser's marker.
END = [40, 41, 42, 43, 44, 200]
OFF_SCRIPT = 77  # the target's own preference where it disagrees with the phrase
FILLER = 5
VOCAB = 256
NUM_SPEC = 3


class _Config:
    reasoning_start_token_ids = [THINK_START]
    reasoning_end_token_ids = END


def _start(budget: int) -> tuple[ThinkingBudgetStateHolder, list[int]]:
    holder = ThinkingBudgetStateHolder(_Config(), 8, NUM_SPEC, torch.device("cpu"), False)
    output: list[int] = []
    params = SamplingParams(thinking_token_budget=budget)
    holder.sync_batch(BatchUpdate(batch_size=1, removed=(), added=[(0, params, None, output)], moved=()))
    output.append(THINK_START)
    return holder, output


def _phrase_progress(tokens: list[int]) -> int:
    """END tokens emitted since the first END[0], if the tail is exactly them."""
    if END[0] not in tokens:
        return -1
    tail = tokens[tokens.index(END[0]) :]
    n = 0
    while n < len(tail) and n < len(END) and tail[n] == END[n]:
        n += 1
    return n if n == len(tail) else -2


def _natural(breaks_at: int):
    """Target argmax: FILLER before the end sequence starts, then the phrase,
    except right after END[:breaks_at] where it prefers OFF_SCRIPT."""

    def natural(prefix: list[int]) -> int:
        n = _phrase_progress(prefix)
        if n == breaks_at:
            return OFF_SCRIPT
        if 0 <= n < len(END):
            return END[n]
        return FILLER

    return natural


def _phrase_drafter(output: list[int]) -> list[int]:
    """Proposes the next END tokens once the end sequence has started."""
    n = _phrase_progress(output)
    if n < 0:
        return [9] * NUM_SPEC
    nxt = END[n : n + NUM_SPEC]
    return nxt + [9] * (NUM_SPEC - len(nxt))


def _step(holder, output, drafts, natural):
    """One V1 verification step, greedy."""
    holder.update_state([output], [drafts])
    bonus = torch.zeros(1, VOCAB)
    bonus[0, natural(output + drafts)] = 1.0
    bonus = holder.apply_to_logits(bonus, True, [drafts])
    target = torch.zeros(len(drafts), VOCAB)
    for j in range(len(drafts)):
        target[j, natural(output + drafts[:j])] = 1.0
    target = holder.apply_to_logits(target, False, [drafts])
    for j, d in enumerate(drafts):
        t = int(torch.argmax(target[j]))
        output.append(t)
        if t != d:
            return
    output.append(int(torch.argmax(bonus[0])))


def _run_until_closed(holder, output, drafter, natural, max_steps=40):
    for _ in range(max_steps):
        if END[0] in output and END[-1] in output[output.index(END[0]) :]:
            return
        _step(holder, output, drafter(output), natural)
    raise AssertionError(f"end sequence never closed: {output}")


def _assert_exact_end(output):
    start = output.index(END[0])
    emitted = output[start : start + len(END)]
    assert emitted == END, f"forced end sequence corrupted: {output[start:]}"
    assert output[start:].count(END[0]) == 1, f"end sequence restarted: {output}"


def test_forced_end_survives_target_rejecting_a_matching_draft():
    holder, output = _start(budget=12)
    _run_until_closed(holder, output, _phrase_drafter, _natural(3))
    _assert_exact_end(output)


def test_forcing_from_inside_the_draft_window_covers_the_rest():
    """The budget runs out one position into the window while the drafter
    already proposes the phrase."""
    holder, output = _start(budget=10)

    def drafter(out):
        if END[0] in out:
            return _phrase_drafter(out)
        return [FILLER, END[0], END[1]] if len(out) - 1 >= 9 else [FILLER] * 2 + [9]

    _run_until_closed(holder, output, drafter, _natural(1))
    _assert_exact_end(output)


def test_forced_end_exact_when_no_draft_matches():
    """Control: no draft ever matches, so one end token is forced per step."""
    holder, output = _start(budget=12)
    _run_until_closed(holder, output, lambda o: [9] * NUM_SPEC, _natural(3))
    _assert_exact_end(output)


def test_forced_row_outside_the_logits_fails_loudly():
    """A forced draft position with no logits row must raise, not be skipped:
    skipping it drops an end token and corrupts the sequence again."""
    holder, output = _start(budget=2)
    output.extend([FILLER] * 4)
    drafts = [9] * NUM_SPEC
    holder.update_state([output], [drafts])

    fits = holder.apply_to_logits(torch.zeros(NUM_SPEC, VOCAB), False, [drafts])
    assert [int(torch.argmax(fits[j])) for j in range(NUM_SPEC)] == END[:NUM_SPEC]

    with pytest.raises(AssertionError, match="outside the logits"):
        holder.apply_to_logits(torch.zeros(NUM_SPEC - 1, VOCAB), False, [drafts])


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("bonus", [False, True])
def test_draftless_step_keeps_each_request_row(num_spec, bonus):
    holder = ThinkingBudgetStateHolder(_Config(), 4, num_spec, torch.device("cpu"), False)
    outputs = [[THINK_START, FILLER, FILLER] for _ in range(3)]
    holder.sync_batch(
        BatchUpdate(
            batch_size=3,
            removed=(),
            moved=(),
            added=[
                (i, SamplingParams(thinking_token_budget=budget), None, outputs[i])
                for i, budget in enumerate([2, None, 2])
            ],
        )
    )
    holder.update_state(outputs, None)
    logits = holder.apply_to_logits(torch.zeros(3, VOCAB), bonus, [[], [], []])
    assert holder.cu_num_tokens == {0: 0, 1: 1, 2: 2}
    assert logits[0, END[0]] == 1e9
    assert logits[2, END[0]] == 1e9
    assert torch.count_nonzero(logits[1]) == 0


@pytest.mark.parametrize("budget", [0, 1, 3, 7])
@pytest.mark.parametrize("num_spec", [0, 3])
def test_end_progress_survives_repeated_update_and_returns_to_answer(budget, num_spec):
    holder = ThinkingBudgetStateHolder(_Config(), 2, num_spec, torch.device("cpu"), False)
    output: list[int] = []
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), [THINK_START], output)],
        )
    )
    # No draft matches the end phrase, including the no-draft initial step.
    for step in range(budget + len(END)):
        drafts = [] if step == 0 else [9] * num_spec
        holder.update_state([output], [drafts])
        holder.update_state([output], [drafts])
        logits = torch.zeros(max(1, len(drafts)), VOCAB)
        logits[:, FILLER] = 1.0
        holder.apply_to_logits(logits, False, [drafts])
        output.append(int(logits[0].argmax()))
    assert output == [FILLER] * budget + END
    holder.update_state([output], [[]])
    answer = torch.randn(1, VOCAB)
    assert torch.equal(holder.apply_to_logits(answer.clone(), False, [[]]), answer)


@pytest.mark.parametrize("progress", [1, 3, 5])
def test_resumed_request_continues_committed_end_prefix(progress):
    holder = ThinkingBudgetStateHolder(_Config(), 2, NUM_SPEC, torch.device("cpu"), False)
    output = [FILLER] * 2 + END[:progress]
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=2), [THINK_START], output)],
        )
    )
    holder.update_state([output], [[]])
    logits = holder.apply_to_logits(torch.zeros(1, VOCAB), False, [[]])
    assert int(logits[0].argmax()) == END[progress]


def test_matching_suffix_before_budget_is_not_committed_end():
    holder = ThinkingBudgetStateHolder(_Config(), 2, NUM_SPEC, torch.device("cpu"), False)
    output = [FILLER] + END[:2]
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=3), [THINK_START], output)],
        )
    )
    holder.update_state([output], [[]])
    logits = holder.apply_to_logits(torch.zeros(1, VOCAB), False, [[]])
    assert int(logits[0].argmax()) == END[0]
