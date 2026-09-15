# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pool lookup or unrelated backend call must not prove Pool -> P -> D."""

import pytest

from tests.e2e.pull_request.pd_pool import pool_load_observed

ALLOCATED = (
    "[PD_0] KV pool load spec enabled req=cmpl-abc num_external_tokens=128 vllm_cached=0 kvpool_cached=128 groups=[0]"
)
COMPLETED = "[PD_0] KV pool worker backend get returned request=cmpl-abc token_len=128 groups=[0] keys=1"


def test_correlated_nonempty_prefill_pool_load():
    assert pool_load_observed([ALLOCATED, COMPLETED], "cmpl-abc")


@pytest.mark.parametrize(
    "lines",
    [
        [ALLOCATED],
        [COMPLETED],
        [ALLOCATED, COMPLETED.replace("cmpl-abc", "cmpl-other")],
        [ALLOCATED, COMPLETED.replace("keys=1", "keys=0")],
        [ALLOCATED.replace("vllm_cached=0", "vllm_cached=128"), COMPLETED],
        [ALLOCATED.replace("[PD_0]", "[PD_1]"), COMPLETED.replace("[PD_0]", "[PD_1]")],
        [ALLOCATED, COMPLETED, "[PD_0] Failed to get 1 keys out of 1. error_codes=[-1]"],
        [ALLOCATED, COMPLETED, "[PD_0] Failed to put 1 keys out of 1. error_codes=[-1]"],
    ],
)
def test_incomplete_or_unrelated_evidence_is_rejected(lines):
    assert not pool_load_observed(lines, "cmpl-abc")
