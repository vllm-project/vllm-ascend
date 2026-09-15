# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject distribution corruption and never compare divergent histories."""

import copy

import pytest

from tests.e2e.pull_request.pd_output import assert_distribution_match


def response(tokens, distributions=None):
    if distributions is None:
        distributions = [{"token_id:1": -0.7, "token_id:2": -0.8} for _ in tokens]
    return {"choices": [{"token_ids": tokens, "logprobs": {"top_logprobs": distributions}}]}


def no_replay(history):
    pytest.fail(f"Unexpected replay: {history}")


def test_exact_and_small_distribution_changes_pass():
    baseline = response([1, 1])
    actual = response([1, 1], [{"token_id:1": -0.71, "token_id:2": -0.79}] * 2)
    assert_distribution_match(baseline, baseline, [7], no_replay)
    assert_distribution_match(baseline, actual, [7], no_replay)


def test_replay_checks_every_reference_history_after_divergence():
    baseline = response([1, 1, 1])
    baseline["_pd_replay_references"] = [response([1]) for _ in range(3)]
    # These later distributions are intentionally invalid: they belong to a
    # different history and must never participate in the comparison.
    actual = response([2, 2, 2], [{"token_id:1": -0.8, "token_id:2": -0.7}, {}, {}])
    histories = []

    def replay(history):
        histories.append(history)
        return response([1])

    assert_distribution_match(baseline, actual, [7, 8], replay)
    assert histories == [[7, 8, 1], [7, 8, 1, 1]]


def test_replay_uses_an_independent_replay_control_not_continuous_decode():
    baseline = response([1, 1], [{"token_id:1": -0.7, "token_id:2": -0.8}, {}])
    baseline["_pd_replay_references"] = [response([1]), response([1])]
    actual = response([2, 2], [{"token_id:1": -0.8, "token_id:2": -0.7}, {}])
    assert_distribution_match(baseline, actual, [7], lambda history: response([1]))
    baseline["_pd_replay_references"][1] = response([1], [{"token_id:1": -2.7, "token_id:2": -2.8}])
    with pytest.raises(AssertionError, match="corruption"):
        assert_distribution_match(baseline, actual, [7], lambda history: response([1]))


@pytest.mark.parametrize("kind", ["missing", "reverse_missing", "nan", "mean", "maximum", "length"])
def test_corruption_remains_a_failure(kind):
    baseline = response([1, 1])
    actual = copy.deepcopy(baseline)
    choice = actual["choices"][0]
    if kind == "missing":
        del choice["logprobs"]["top_logprobs"][0]["token_id:1"]
    elif kind == "reverse_missing":
        choice["token_ids"][0] = 3
    elif kind == "nan":
        choice["logprobs"]["top_logprobs"][0]["token_id:1"] = float("nan")
    elif kind == "mean":
        choice["logprobs"]["top_logprobs"] = [{"token_id:1": -0.9, "token_id:2": -1.0}] * 2
    elif kind == "maximum":
        baseline = response([1] * 10)
        actual = response([1] * 10)
        actual["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:2"] = -1.9
    else:
        choice["token_ids"].pop()
    with pytest.raises(AssertionError):
        assert_distribution_match(baseline, actual, [7], no_replay)
