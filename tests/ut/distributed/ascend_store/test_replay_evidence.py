# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from copy import deepcopy
from typing import Any

import pytest

from tests.SIT.cases.features.kv_pool.hisparse_ascendstore.replay_helpers import (
    COMPUTED_TOKENS,
    EXTERNAL_GETS,
    EXTERNAL_HITS,
    LOCAL_HITS,
    assert_external_replay,
    assert_matching_replay_config,
)


@pytest.mark.parametrize(
    ("hit", "external_hits", "computed", "local_hits", "get_keys", "valid"),
    [
        (128, 128, 1, 0, 8, True),
        (0, 0, 129, 0, 8, False),
        (64, 64, 65, 0, 8, False),  # Restored less than the local reference.
        (128, 128, 129, 0, 8, False),  # Lookup succeeded but the prefix was recomputed.
        (128, 128, 1, 128, 8, False),
        (128, 64, 1, 0, 8, False),
        (128, 128, 1, 0, 0, False),
    ],
)
def test_external_replay_requires_reading_the_whole_cached_prefix(
    hit, external_hits, computed, local_hits, get_keys, valid
):
    local = {"cached_tokens": 128}
    replay = {"cached_tokens": hit}
    counts = {EXTERNAL_HITS: external_hits, COMPUTED_TOKENS: computed, LOCAL_HITS: local_hits, EXTERNAL_GETS: get_keys}
    if valid:
        assert_external_replay(local, replay, counts, 129)
    else:
        with pytest.raises(AssertionError):
            assert_external_replay(local, replay, counts, 129)


@pytest.mark.parametrize("draft_tokens", [3, 4])
def test_reference_replay_checks_mtp_tokens_with_different_server_ports(draft_tokens):
    config: dict[str, dict[str, Any]] = {
        "model_config": {},
        "attention_config": {},
        "cache_config": {},
        "compilation_config": {},
        "parallel_config": {},
        "scheduler_config": {},
    }
    config["speculative_config"] = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "target_parallel_config": {"tensor_parallel_size": 2, "master_port": 29501},
        "draft_parallel_config": {"tensor_parallel_size": 2, "master_port": 29501},
    }
    reference = deepcopy(config)
    reference["speculative_config"]["target_parallel_config"]["master_port"] = 29502
    reference["speculative_config"]["draft_parallel_config"]["master_port"] = 29502
    reference["speculative_config"]["num_speculative_tokens"] = draft_tokens
    if draft_tokens == 3:
        assert_matching_replay_config(config, reference)
    else:
        with pytest.raises(AssertionError, match=r"speculative_config\.num_speculative_tokens differs"):
            assert_matching_replay_config(config, reference)
