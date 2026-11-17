# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in checks against a dedicated A5 TP8 C8/MTP=3 test server.

Set VLLM_ASCEND_HISPARSE_TEST_URL and enable the server-info and cache-reset
development endpoints. The test resets only the local cache, retaining the pool.
"""

import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

_COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:ascend_store_load_get_keys_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:hisparse_cache_misses_total",
)


def _metrics(url):
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    totals = dict.fromkeys(_COUNTERS, 0.0)
    for line in response.text.splitlines():
        sample, _, value = line.rpartition(" ")
        name = sample.partition("{")[0]
        if name in totals:
            totals[name] += float(value)
    return totals


@pytest.fixture(scope="module")
def serving_instance():
    url = os.getenv("VLLM_ASCEND_HISPARSE_TEST_URL", "").rstrip("/")
    if not url:
        pytest.skip("Requires a dedicated A5 HiSparse + AscendStore server")
    response = requests.get(f"{url}/server_info", params={"config_format": "json"}, timeout=30)
    response.raise_for_status()
    config = response.json()["vllm_config"]
    assert config["cache_config"]["cache_dtype"] == "fp8"
    assert config["attention_config"]["indexer_kv_dtype"] == "fp8"
    assert config["speculative_config"]["method"] == "mtp"
    assert config["speculative_config"]["num_speculative_tokens"] == 3
    assert config["scheduler_config"]["async_scheduling"] is True
    transfer = config["kv_transfer_config"]
    assert transfer["kv_connector"] == "MultiConnector" and transfer["kv_role"] == "kv_both"
    children = transfer["kv_connector_extra_config"]["connectors"]
    assert {child["kv_connector"] for child in children} == {"HiSparseConnector", "AscendStoreConnector"}
    store = next(child for child in children if child["kv_connector"] == "AscendStoreConnector")
    backend = store["kv_connector_extra_config"]["backend"]
    assert backend in {"mooncake", "memcache"}
    assert store["kv_connector_extra_config"]["load_async"] is True
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    return url, response.json()["data"][0]["id"], backend


def _request(url, model, workload):
    prompt, salt = workload
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "cache_salt": salt,
            "seed": 42,
            "temperature": 0,
            "ignore_eos": True,
            "max_tokens": 32,
            "return_token_ids": True,
            "logprobs": 5,
        },
        timeout=(10, 900),
    )
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    assert result["usage"]["prompt_tokens"] == len(prompt)
    assert len(choice["token_ids"]) == len(choice["logprobs"]["token_logprobs"]) == 32
    return choice


def _reset_local_cache(url):
    for _ in range(30):
        response = requests.post(f"{url}/reset_prefix_cache", params={"reset_external": "false"}, timeout=30)
        response.raise_for_status()
        if response.json()["success"]:
            return
        time.sleep(0.2)
    pytest.fail("Local cache did not reset after the completed requests")


@pytest.mark.parametrize(("prompt_length", "concurrency"), [(257, 1), (8192, 2), (16384, 1)])
def test_c8_external_replay_matches_local_prefix(serving_instance, tmp_path, prompt_length, concurrency):
    url, model, backend = serving_instance
    rng = random.Random(prompt_length)
    workloads = [
        ([rng.randrange(1000, 31000) for _ in range(prompt_length)], uuid.uuid4().hex) for _ in range(concurrency)
    ]

    with ThreadPoolExecutor(concurrency) as executor:

        def run_arm():
            before = _metrics(url)
            choices = list(executor.map(lambda workload: _request(url, model, workload), workloads))
            after = _metrics(url)
            return choices, {key: after[key] - before[key] for key in before}

        # Populate the local prefix and external pool. Its output is not the
        # reference: this test isolates external restore from local prefix reuse.
        run_arm()
        local, local_counts = run_arm()
        assert local_counts[_COUNTERS[0]] > 0
        assert local_counts[_COUNTERS[1]] == local_counts[_COUNTERS[2]] == 0
        _reset_local_cache(url)
        external, external_counts = run_arm()

    evidence = {
        "backend": backend,
        "prompt_length": prompt_length,
        "concurrency": concurrency,
        "local_counts": local_counts,
        "external_counts": external_counts,
        "local": local,
        "external": external,
    }
    (tmp_path / "replay.json").write_text(json.dumps(evidence), encoding="utf-8")
    assert external_counts[_COUNTERS[0]] == 0
    assert external_counts[_COUNTERS[1]] > 0 and external_counts[_COUNTERS[2]] > 0
    assert local_counts[_COUNTERS[3]] > 1 and external_counts[_COUNTERS[3]] > 1
    for expected, actual in zip(local, external, strict=True):
        assert actual["token_ids"] == expected["token_ids"]
        for field in ("token_logprobs", "top_logprobs"):
            assert actual["logprobs"][field] == expected["logprobs"][field]
    print(
        json.dumps(
            {
                key: evidence[key]
                for key in ("backend", "prompt_length", "concurrency", "local_counts", "external_counts")
            }
        )
    )
