# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in checks against a dedicated A5 TP8 C8/MTP=3 test server.

Set VLLM_ASCEND_HISPARSE_TEST_URL and enable the server-info and cache-reset
development endpoints. The test resets only the local cache, retaining the pool.
"""

import json
import os

import pytest

from tests.e2e.nightly.single_node.hisparse_ascendstore.replay_helpers import (
    DRAFTS,
    EXTERNAL_GETS,
    EXTERNAL_HITS,
    LOCAL_HITS,
    assert_external_replay,
    make_workloads,
    require_mtp3,
    reset_local_cache,
    run_arm,
    served_model,
    server_config,
)


@pytest.fixture(scope="module")
def serving_instance():
    url = os.getenv("VLLM_ASCEND_HISPARSE_TEST_URL", "").rstrip("/")
    if not url:
        pytest.skip("Requires a dedicated A5 HiSparse + AscendStore server")
    config = server_config(url)
    assert config["cache_config"]["cache_dtype"] == "fp8"
    assert config["attention_config"]["indexer_kv_dtype"] == "fp8"
    require_mtp3(config)
    transfer = config["kv_transfer_config"]
    assert transfer["kv_load_failure_policy"] == "fail"
    assert transfer["kv_connector"] == "MultiConnector" and transfer["kv_role"] == "kv_both"
    children = transfer["kv_connector_extra_config"]["connectors"]
    assert {child["kv_connector"] for child in children} == {"HiSparseConnector", "AscendStoreConnector"}
    store = next(child for child in children if child["kv_connector"] == "AscendStoreConnector")
    backend = store["kv_connector_extra_config"]["backend"]
    assert backend == "memcache"
    assert store["kv_connector_extra_config"]["load_async"] is True
    return url, served_model(url), backend


@pytest.mark.parametrize(("prompt_length", "concurrency"), [(257, 1), (8192, 2), (16384, 1)])
def test_c8_external_replay_matches_warm_prefix(serving_instance, tmp_path, prompt_length, concurrency):
    url, model, backend = serving_instance
    workloads = make_workloads(prompt_length, concurrency, seed=prompt_length)

    # Compare the same warm prefix restored from either cache.
    _, seed_counts = run_arm(url, model, workloads)
    assert seed_counts[LOCAL_HITS] == seed_counts[EXTERNAL_HITS] == seed_counts[EXTERNAL_GETS] == 0
    warm, warm_counts = run_arm(url, model, workloads)
    assert warm_counts[LOCAL_HITS] > 0
    if prompt_length >= 8192:
        assert warm_counts[EXTERNAL_HITS] == warm_counts[EXTERNAL_GETS] == 0
    # A two-block prompt may keep only part of its prefix locally.
    reset_local_cache(url)
    external, external_counts = run_arm(url, model, workloads)

    evidence = {
        "backend": backend,
        "prompt_length": prompt_length,
        "concurrency": concurrency,
        "seed_counts": seed_counts,
        "warm_counts": warm_counts,
        "external_counts": external_counts,
        "warm": warm,
        "external": external,
    }
    (tmp_path / "replay.json").write_text(json.dumps(evidence), encoding="utf-8")
    assert_external_replay(warm, external, external_counts, prompt_length)
    expected_cached_tokens = (prompt_length - 1) // 128 * 128
    assert all(choice["cached_tokens"] == expected_cached_tokens for choice in external)
    assert warm_counts[DRAFTS] > 1 and external_counts[DRAFTS] > 1
    for expected, actual in zip(warm, external, strict=True):
        assert actual["token_ids"] == expected["token_ids"]
        for field in ("token_logprobs", "top_logprobs"):
            assert actual["logprobs"][field] == expected["logprobs"][field]
    print(
        json.dumps(
            {
                key: evidence[key]
                for key in ("backend", "prompt_length", "concurrency", "warm_counts", "external_counts")
            }
        )
    )
