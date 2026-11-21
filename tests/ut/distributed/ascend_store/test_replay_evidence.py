# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from tests.e2e.nightly.single_node.hisparse_ascendstore.replay_helpers import (
    COMPUTED_TOKENS,
    EXTERNAL_GETS,
    EXTERNAL_HITS,
    LOCAL_HITS,
    assert_external_replay,
)


@pytest.mark.parametrize(
    ("hits", "external_hits", "computed", "local_hits", "get_keys", "valid"),
    [
        ([128, 128], 256, 2, 0, 8, True),
        ([128, 0], 128, 130, 0, 8, False),  # One request missed despite a positive aggregate.
        ([128, 128], 256, 130, 0, 8, False),  # Lookup succeeded but part of the prefix was recomputed.
        ([128, 128], 128, 2, 128, 8, False),
        ([128, 128], 128, 2, 0, 8, False),
        ([128, 128], 256, 2, 0, 0, False),
    ],
)
def test_external_replay_requires_each_request_to_load_its_warm_prefix(
    hits, external_hits, computed, local_hits, get_keys, valid
):
    warm = [{"cached_tokens": 128}, {"cached_tokens": 128}]
    replay = [{"cached_tokens": count} for count in hits]
    counts = {EXTERNAL_HITS: external_hits, COMPUTED_TOKENS: computed, LOCAL_HITS: local_hits, EXTERNAL_GETS: get_keys}
    if valid:
        assert_external_replay(warm, replay, counts, 129)
    else:
        with pytest.raises(AssertionError):
            assert_external_replay(warm, replay, counts, 129)
