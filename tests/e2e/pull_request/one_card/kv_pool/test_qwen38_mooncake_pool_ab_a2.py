# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A2 mooncake KV-pool A/B cases for the hybrid-GDN Qwen3.8-27B W8A8 model.

Two mooncake scenarios (bulk tier-2 DRAM and layerwise tier-3 DRAM+SSD) run on
the shared a2-1 pool, where the NUMA cpuset binding keeps the A/B ratios
stable. The full scenario matrix and measurement criteria are documented in
``kv_pool_helpers.py``.
"""

from unittest.mock import patch

import pytest

from tests.e2e.pull_request.one_card.kv_pool.kv_pool_helpers import MODEL, run


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize(
    ("backend", "use_layerwise"),
    [
        ("mooncake", False),
        ("mooncake", True),
    ],
    ids=["mooncake_bulk_tier2", "mooncake_layerwise_tier3"],
)
@patch.dict("os.environ", {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_qwen38_mooncake_pool_ab_a2(backend: str, use_layerwise: bool):
    run(backend, use_layerwise)
