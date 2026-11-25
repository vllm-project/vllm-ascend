# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A2 memcache KV-pool A/B cases for the hybrid-GDN Qwen3.8-27B W8A8 model.

Two memcache scenarios (bulk tier-2 DRAM and layerwise tier-3 DRAM+SSD) run
on the dedicated a2-loop-1 single-card runner, which injects the raw UBS IO
loop device and (once reserved) hugepages that the memcache local service
needs. The full scenario matrix and measurement criteria are documented in
``kv_pool_helpers.py``.
"""

from unittest.mock import patch

import pytest

from tests.e2e.pull_request.one_card.kv_pool.kv_pool_helpers import MODEL, run


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@pytest.mark.parametrize(
    ("backend", "use_layerwise"),
    [
        ("memcache", False),
        ("memcache", True),
    ],
    ids=["memcache_bulk_tier2", "memcache_layerwise_tier3"],
)
@patch.dict("os.environ", {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_qwen38_memcache_pool_ab_a2(backend: str, use_layerwise: bool):
    run(backend, use_layerwise)
