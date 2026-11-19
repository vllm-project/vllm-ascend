# SPDX-License-Identifier: Apache-2.0
"""Output validity and state-transfer assertions for the Flash PD test."""

import math
from collections import Counter, defaultdict


def check_completion(result, output_tokens):
    assert "error" not in result, result
    assert result["usage"]["completion_tokens"] == output_tokens
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length"
    logs = choice["logprobs"]
    assert len(logs["tokens"]) == len(logs["token_logprobs"]) == output_tokens, "Unexpected token/logprob count"
    assert all(math.isfinite(value) for value in logs["token_logprobs"]), "Non-finite token log probabilities"
    return logs


def check_state_pulls(workers, expected_pulls=2):
    """Require every observed KDA state to receive all PCP head shards."""
    groups = defaultdict(Counter)
    for worker_idx, worker in enumerate(workers):
        for record in worker["copies"]:
            assert record["pulls"] == expected_pulls, "KDA state did not include every PCP shard"
            assert record["bytes"] == record["expected_bytes"] > 0, "Missing or truncated KDA state copy"
            groups[(worker_idx, record["state"], record["block"])][record["offset"]] += 1
    assert groups, "No remote KDA state copies observed; local recomputation is not PD coverage"
    for offsets in groups.values():
        assert set(offsets) == set(range(expected_pulls)), "Missing PCP state shard"
        assert len(set(offsets.values())) == 1, "Unequal PCP state shard copy counts"
