# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare quantized PD distributions under identical conditioning histories."""

import math
from collections.abc import Callable


def assert_distribution_match(
    reference: dict,
    actual: dict,
    prompt: list[int],
    replay: Callable[[list[int]], dict],
    *,
    maximum_delta: float = 1.0,
    mean_delta: float = 0.15,
) -> None:
    """Check all answer positions, replaying the reference history after a fork.

    The default limits match the existing Ascend sequence-parallel precision
    test (maximum delta < 1.0, mean delta < 0.15). Additionally require mutual
    top-5 greedy membership, as in upstream models/utils.py check_logprobs_close.
    Callers exercising lossy cache formats may provide explicit limits. Unlike
    stopping at the first greedy difference, replay checks every position against
    the same history. It never forces a generated answer token.
    """
    assert maximum_delta > 0 and mean_delta > 0
    expected = reference["choices"][0]
    observed = actual["choices"][0]
    tokens = expected["token_ids"]
    assert tokens and len(observed["token_ids"]) == len(tokens)
    assert len(expected["logprobs"]["top_logprobs"]) == len(tokens)
    assert len(observed["logprobs"]["top_logprobs"]) == len(tokens)
    deltas: list[float] = []
    largest = (0.0, -1, "")
    for position, token in enumerate(tokens):
        if observed["token_ids"][:position] == tokens[:position]:
            candidate_token = observed["token_ids"][position]
            distribution = observed["logprobs"]["top_logprobs"][position]
            target = expected["logprobs"]["top_logprobs"][position]
        else:
            conditioned = replay(prompt + tokens[:position])["choices"][0]
            assert len(conditioned["token_ids"]) == 1
            candidate_token = conditioned["token_ids"][0]
            distribution = conditioned["logprobs"]["top_logprobs"][0]
            # Both sides must use a fresh one-token request here. Comparing
            # a replayed prefill with the original continuous decode would
            # conflate the execution boundary with the transport under test.
            control = reference["_pd_replay_references"][position]["choices"][0]
            assert len(control["token_ids"]) == 1
            token = control["token_ids"][0]
            target = control["logprobs"]["top_logprobs"][0]
        assert f"token_id:{token}" in distribution, f"Reference greedy token outside top-5 at {position}"
        assert f"token_id:{candidate_token}" in target, f"PD greedy token outside reference top-5 at {position}"
        assert all(math.isfinite(value) and value <= 0 for value in (*target.values(), *distribution.values()))
        shared = target.keys() & distribution.keys()
        assert shared
        for key in shared:
            delta = abs(target[key] - distribution[key])
            deltas.append(delta)
            largest = max(largest, (delta, position, key))
    maximum, mean = max(deltas), sum(deltas) / len(deltas)
    print(f"PD top-5 conditional check: {len(tokens)} positions, max={maximum:.6f}, mean={mean:.6f}", flush=True)
    assert maximum < maximum_delta, (
        f"Conditional distribution corruption: max delta={maximum:.6f}, limit={maximum_delta:.6f}, at={largest[1:]}"
    )
    assert mean < mean_delta, f"Conditional distribution drift: mean delta={mean:.6f}, limit={mean_delta:.6f}"
