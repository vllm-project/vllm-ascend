# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.worker.v2.sample.rejection_greedy import (
    rejection_greedy_sample_pytorch,
    rejection_greedy_sample_spec_len_1_pytorch,
)

PLACEHOLDER = -1
TOKEN_CHOICES = 4  # Tiny "vocabulary" so draft/target collisions are frequent.


def _reference_spec_len_1(draft_token_ids, target_argmax, bonus_token_ids):
    """MTP1 reference: the target token always wins column 0."""
    return [
        [int(target), int(bonus) if draft == target else PLACEHOLDER]
        for draft, target, bonus in zip(draft_token_ids, target_argmax, bonus_token_ids)
    ]


def _reference_general(draft_token_ids, target_argmax, bonus_token_ids, draft_tokens_per_req, max_spec_len, is_greedy):
    """Index-free transcription of greedy verification, one request at a time.

    Drafts are accepted while they match the target argmax. At the first mismatch
    the target token replaces the draft, so copying ``target_argmax`` over the
    accepted prefix is equivalent. The bonus token is appended only when every
    draft was accepted.
    """
    batch_size = len(draft_tokens_per_req)
    if is_greedy is None:
        is_greedy = [True] * batch_size

    outputs = [[PLACEHOLDER] * (max_spec_len + 1) for _ in range(batch_size)]
    start = 0
    for req in range(batch_size):
        num_drafts = int(draft_tokens_per_req[req])
        assert num_drafts <= max_spec_len
        drafts = [int(t) for t in draft_token_ids[start : start + num_drafts]]
        targets = [int(t) for t in target_argmax[start : start + num_drafts]]
        if is_greedy[req]:
            first_mismatch = next((i for i in range(num_drafts) if drafts[i] != targets[i]), num_drafts)
            for pos in range(min(first_mismatch + 1, num_drafts)):
                outputs[req][pos] = targets[pos]
            if first_mismatch == num_drafts:
                outputs[req][num_drafts] = int(bonus_token_ids[req])
        start += num_drafts
    return outputs


def _run_general(
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    draft_tokens_per_req,
    max_spec_len,
    is_greedy=None,
    output_dtype=torch.int32,
    bonus_dtype=torch.int32,
):
    """Call ``rejection_greedy_sample_pytorch`` with production-shaped inputs."""
    batch_size = len(draft_tokens_per_req)
    output = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER, dtype=output_dtype)
    cu_num_draft_tokens = torch.tensor(
        [sum(draft_tokens_per_req[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
    )
    rejection_greedy_sample_pytorch(
        output,
        cu_num_draft_tokens,
        torch.tensor(draft_token_ids, dtype=torch.int64),
        torch.tensor(target_argmax, dtype=torch.int64),
        torch.tensor(bonus_token_ids, dtype=bonus_dtype).view(-1, 1),
        list(draft_tokens_per_req),
        max_spec_len,
        None if is_greedy is None else torch.tensor(is_greedy, dtype=torch.bool),
    )
    return output


def _build_tokens(counts, accept_pattern, seed=0):
    """Build draft/target tokens with a controlled accept pattern."""
    rng = torch.Generator().manual_seed(seed)
    num_tokens = sum(counts)
    drafts = torch.randint(0, TOKEN_CHOICES, (num_tokens,), generator=rng).tolist()
    if accept_pattern == "all":
        targets = list(drafts)
    elif accept_pattern == "none":
        targets = [(token + 1) % TOKEN_CHOICES for token in drafts]
    elif accept_pattern == "random":
        mismatches = torch.randint(0, 2, (num_tokens,), generator=rng).tolist()
        targets = [token if m == 0 else (token + 1) % TOKEN_CHOICES for token, m in zip(drafts, mismatches)]
    else:
        raise ValueError(f"unknown accept pattern: {accept_pattern}")
    return drafts, targets


@pytest.mark.parametrize(
    "draft_token_ids,target_argmax,bonus_token_ids",
    [
        # Every draft survives verification.
        ([5, 5, 5, 5], [5, 5, 5, 5], [7, 8, 9, 10]),
        # Every draft is rejected: column 0 still carries the target token.
        ([1, 2, 3, 4], [4, 3, 2, 1], [7, 8, 9, 10]),
        # Interleaved accept/reject, with bonus tokens that collide with drafts.
        ([0, 1, 2, 3], [0, 9, 9, 3], [1, 2, 3, 0]),
        # Single request, rejected.
        ([6], [2], [6]),
        # Single request, accepted.
        ([6], [6], [2]),
    ],
)
def test_spec_len_1_accepts_only_matching_drafts(draft_token_ids, target_argmax, bonus_token_ids):
    batch_size = len(draft_token_ids)
    output = torch.full((batch_size, 2), PLACEHOLDER, dtype=torch.int32)

    rejection_greedy_sample_spec_len_1_pytorch(
        output,
        torch.tensor(draft_token_ids, dtype=torch.int32),
        torch.tensor(target_argmax, dtype=torch.int32),
        torch.tensor(bonus_token_ids, dtype=torch.int32).view(-1, 1),
    )

    assert output.tolist() == _reference_spec_len_1(draft_token_ids, target_argmax, bonus_token_ids)
    # Rejected rows keep the placeholder in column 1, which is what the sampler
    # counts as "not sampled" and fills from the sampled path instead.
    for req, (draft, target) in enumerate(zip(draft_token_ids, target_argmax)):
        assert output[req, 0].item() == target
        if draft == target:
            assert output[req, 1].item() == bonus_token_ids[req]
        else:
            assert output[req, 1].item() == PLACEHOLDER


def test_spec_len_1_consumes_batch_column_bonus():
    """The contract is ``[batch, 1]``; the sampler passes ``.view(-1, 1)``."""
    draft_token_ids = [3, 4]
    target_argmax = [3, 0]
    bonus_token_ids = [8, 9]

    output = torch.full((2, 2), PLACEHOLDER, dtype=torch.int32)
    rejection_greedy_sample_spec_len_1_pytorch(
        output,
        torch.tensor(draft_token_ids, dtype=torch.int32),
        torch.tensor(target_argmax, dtype=torch.int32),
        torch.tensor(bonus_token_ids, dtype=torch.int32).view(-1, 1),
    )

    assert output.tolist() == [[3, 8], [0, PLACEHOLDER]]


@pytest.mark.parametrize(
    "counts,max_spec_len",
    [
        ([1], 1),
        ([1, 1, 1], 1),
        ([3], 3),
        ([3, 3, 3], 3),
        ([3, 2, 1], 3),
        ([0, 2, 0], 2),
        ([0, 0], 2),
        ([4, 1, 0, 2], 4),
    ],
)
@pytest.mark.parametrize("accept_pattern", ["all", "none", "random"])
@pytest.mark.parametrize("seed", [0, 1])
def test_general_matches_reference(counts, max_spec_len, accept_pattern, seed):
    """Ragged spec lengths, all accept patterns, zero-draft requests."""
    drafts, targets = _build_tokens(counts, accept_pattern, seed=seed)
    bonus_token_ids = [TOKEN_CHOICES + req for req in range(len(counts))]

    output = _run_general(drafts, targets, bonus_token_ids, counts, max_spec_len)

    expected = _reference_general(drafts, targets, bonus_token_ids, counts, max_spec_len, is_greedy=None)
    assert output.tolist() == expected
    # At most ``num_drafts + 1`` columns per request are written; the rest keep
    # the placeholder so the sampler can count sampled tokens.
    for req, num_drafts in enumerate(counts):
        assert output[req].numel() == max_spec_len + 1
        assert output[req, num_drafts + 1 :].tolist() == [PLACEHOLDER] * (max_spec_len - num_drafts)


@pytest.mark.parametrize("mismatch_pos", [0, 1, 2, 3])
def test_general_mismatch_position_copies_target_up_to_mismatch(mismatch_pos):
    """Only positions 0..mismatch are written; the tail stays untouched."""
    drafts = [0, 1, 2, 3]
    targets = [0, 1, 2, 3]
    targets[mismatch_pos] = (targets[mismatch_pos] + 1) % TOKEN_CHOICES
    bonus_token_ids = [7]

    output = _run_general(drafts, targets, bonus_token_ids, [4], max_spec_len=4)

    expected_row = [PLACEHOLDER] * 5
    for pos in range(mismatch_pos + 1):
        expected_row[pos] = targets[pos]
    assert output[0].tolist() == expected_row
    # The bonus token must not be consumed when a draft was rejected.
    assert bonus_token_ids[0] not in output[0, mismatch_pos + 1 :].tolist()
    assert output[0, mismatch_pos + 1 :].tolist() == [PLACEHOLDER] * (4 - mismatch_pos)


def test_general_ragged_counts_keep_placeholder_tail():
    """Two of three requests accept fewer drafts than ``max_spec_len``."""
    counts = [3, 2, 1]
    drafts = [10, 11, 12, 13, 14, 15]
    targets = list(drafts)
    bonus_token_ids = [20, 21, 22]

    output = _run_general(drafts, targets, bonus_token_ids, counts, max_spec_len=3)

    assert output.tolist() == [
        [10, 11, 12, 20],
        [13, 14, 21, PLACEHOLDER],
        [15, 22, PLACEHOLDER, PLACEHOLDER],
    ]
    # ``num_sampled`` in the sampler is ``(sampled != PLACEHOLDER).sum(dim=1)``.
    assert (output != PLACEHOLDER).sum(dim=1).tolist() == [4, 3, 2]


def test_general_zero_draft_requests_get_bonus_in_first_column():
    """Chunked-prefill-style requests contribute no drafts but still sample."""
    output = _run_general([], [], [30, 31, 32], [0, 0, 0], max_spec_len=2)

    assert output.tolist() == [
        [30, PLACEHOLDER, PLACEHOLDER],
        [31, PLACEHOLDER, PLACEHOLDER],
        [32, PLACEHOLDER, PLACEHOLDER],
    ]
    assert (output != PLACEHOLDER).sum(dim=1).tolist() == [1, 1, 1]


def test_general_max_spec_len_zero_writes_bonus_only():
    """The ``max_spec_len == 0`` guard must not build the empty position matrix."""
    output = _run_general([], [], [40, 41], [0, 0], max_spec_len=0)

    assert output.shape == (2, 1)
    assert output.tolist() == [[40], [41]]


def test_general_non_greedy_rows_are_untouched():
    """Non-greedy requests are filled by the sampled path, not here."""
    drafts = [1, 2, 3, 4, 5, 6]
    # Request 0 would mismatch at position 1 if it were greedy; it must stay empty.
    targets = [1, 9, 3, 4, 5, 6]
    counts = [3, 3]
    bonus_token_ids = [50, 51]

    output = _run_general(drafts, targets, bonus_token_ids, counts, max_spec_len=3, is_greedy=[False, True])

    assert output.tolist() == [
        [PLACEHOLDER] * 4,
        [4, 5, 6, 51],
    ]


def test_general_all_non_greedy_rows_write_nothing():
    output = _run_general([1], [1], [60], [1], max_spec_len=1, is_greedy=[False])

    assert output.tolist() == [[PLACEHOLDER, PLACEHOLDER]]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_general_preserves_output_dtype(dtype):
    """``target_argmax`` is cast on the copy path; the bonus path needs parity.

    ``output_token_ids`` and ``bonus_token_ids`` are int32 in the 310P sampler, and
    the bonus index-put is not cast, so the two must share a dtype. Both supported
    combinations are pinned here.
    """
    output = _run_general([1, 2], [1, 2], [8], [2], max_spec_len=2, output_dtype=dtype, bonus_dtype=dtype)

    assert output.dtype == dtype
    assert output.tolist() == [[1, 2, 8]]


def test_general_leaves_inputs_unmodified():
    drafts = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    targets = torch.tensor([1, 9, 3, 4], dtype=torch.int64)
    bonus = torch.tensor([[7], [8]], dtype=torch.int32)
    cu_num_draft_tokens = torch.tensor([2, 4], dtype=torch.int32)
    counts = [2, 2]

    drafts_copy, targets_copy = drafts.clone(), targets.clone()
    bonus_copy, cu_copy = bonus.clone(), cu_num_draft_tokens.clone()

    output = torch.full((2, 3), PLACEHOLDER, dtype=torch.int32)
    rejection_greedy_sample_pytorch(output, cu_num_draft_tokens, drafts, targets, bonus, counts, 2)

    assert output.tolist() == [[1, 9, PLACEHOLDER], [3, 4, 8]]
    torch.testing.assert_close(drafts, drafts_copy)
    torch.testing.assert_close(targets, targets_copy)
    torch.testing.assert_close(bonus, bonus_copy)
    torch.testing.assert_close(cu_num_draft_tokens, cu_copy)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_spec_len_1_matches_general_path(seed):
    """The MTP1 fast path must agree with the general path it replaces."""
    counts = [1, 1, 1, 1, 1]
    drafts, targets = _build_tokens(counts, "random", seed=seed)
    bonus_token_ids = [TOKEN_CHOICES + req for req in range(len(counts))]

    fast = torch.full((len(counts), 2), PLACEHOLDER, dtype=torch.int32)
    rejection_greedy_sample_spec_len_1_pytorch(
        fast,
        torch.tensor(drafts, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.int32),
        torch.tensor(bonus_token_ids, dtype=torch.int32).view(-1, 1),
    )

    general = _run_general(drafts, targets, bonus_token_ids, counts, max_spec_len=1)

    assert fast.tolist() == general.tolist()
