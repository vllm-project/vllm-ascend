# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in GLM-5.3 replay regression against a dedicated A5 serving instance.

Pass --hisparse-replay-url for an idle test server in dev mode with prefix
caching enabled. Select the configured connectors with --hisparse-replay-mode.
Use the same model, runner, and graph settings in all arms. The Store modes
reset the server's local cache during the test. Pass --hisparse-replay-mtp for
MTP=3 with async scheduling, or --hisparse-replay-long to include 128K inputs
on a server with a context window that leaves room for the output tokens.
"""

import concurrent.futures
import random
import uuid

import pytest
import requests

LOCAL_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
EXTERNAL_GETS = "vllm:ascend_store_load_get_keys_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"


def _metrics(url):
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    totals = dict.fromkeys((LOCAL_HITS, EXTERNAL_HITS, EXTERNAL_GETS, DRAFTS), 0.0)
    for line in response.text.splitlines():
        sample, _, value = line.rpartition(" ")
        name = sample.partition("{")[0]
        if name in totals:
            totals[name] += float(value)
    return totals


@pytest.fixture(scope="module")
def serving_instance(pytestconfig):
    url = pytestconfig.getoption("--hisparse-replay-url").rstrip("/")
    if not url:
        pytest.skip("Pass --hisparse-replay-url to run against a dedicated A5 test server")
    mode = pytestconfig.getoption("--hisparse-replay-mode")
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    if pytestconfig.getoption("--hisparse-replay-mtp"):
        response = requests.get(f"{url}/server_info", params={"config_format": "json"}, timeout=30)
        response.raise_for_status()
        config = response.json()["vllm_config"]
        assert config["speculative_config"]["method"] == "mtp"
        assert config["speculative_config"]["num_speculative_tokens"] == 3
        assert config["scheduler_config"]["async_scheduling"] is True
    return url, model, mode


def _request(url, model, workload, output_tokens):
    prompt, salt = workload
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "cache_salt": salt,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "max_tokens": output_tokens,
            "return_token_ids": True,
            "logprobs": 5,
        },
        timeout=(10, 900),
    )
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    assert result["usage"]["prompt_tokens"] == len(prompt)
    assert len(choice["token_ids"]) == len(choice["logprobs"]["token_logprobs"]) == output_tokens
    return choice["token_ids"], choice["logprobs"]


@pytest.mark.parametrize("prompt_length", [16383, 16384, 16385, 131072])
@pytest.mark.parametrize("concurrency", [1, 2])
def test_hisparse_prefix_replay(serving_instance, prompt_length, concurrency, pytestconfig):
    if prompt_length == 131072 and not pytestconfig.getoption("--hisparse-replay-long"):
        pytest.skip("Pass --hisparse-replay-long for the 128K case")
    url, model, mode = serving_instance
    output_tokens = pytestconfig.getoption("--hisparse-replay-output-tokens")
    rng = random.Random(16384)
    workloads = [
        ([rng.randrange(1000, 31000) for _ in range(prompt_length)], uuid.uuid4().hex) for _ in range(concurrency)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:

        def run_arm():
            before = _metrics(url)
            outputs = list(executor.map(lambda workload: _request(url, model, workload, output_tokens), workloads))
            after = _metrics(url)
            return outputs, {key: after[key] - before[key] for key in before}

        cold, cold_counts = run_arm()
        local, local_counts = run_arm()
        assert cold_counts[LOCAL_HITS] == cold_counts[EXTERNAL_HITS] == cold_counts[EXTERNAL_GETS] == 0
        assert local_counts[LOCAL_HITS] > 0
        mixed = bool(local_counts[EXTERNAL_HITS] or local_counts[EXTERNAL_GETS])
        if mode != "plain-store":
            assert not mixed, local_counts
        arms = [("local+external" if mixed else "local", local, local_counts)]
        if mode in {"store", "plain-store"}:
            response = requests.post(f"{url}/reset_prefix_cache", params={"reset_external": "false"}, timeout=30)
            response.raise_for_status()
            assert response.json()["success"]
            external, external_counts = run_arm()
            assert external_counts[LOCAL_HITS] == 0
            assert external_counts[EXTERNAL_HITS] > 0 and external_counts[EXTERNAL_GETS] > 0
            arms.append(("external", external, external_counts))

    mtp = pytestconfig.getoption("--hisparse-replay-mtp")
    tolerance = pytestconfig.getoption("--hisparse-replay-logprob-atol")
    if mtp:
        assert cold_counts[DRAFTS] > 1
    comparisons = [("cold", cold, name, replay) for name, replay, _ in arms]
    for name, replay, counts in arms:
        if mtp:
            assert counts[DRAFTS] > 1, name
        if name == "external":
            comparisons.append((arms[0][0], local, name, replay))

    failures = []
    for ref_name, reference, name, replay in comparisons:
        for index, ((ref_ids, ref_probs), (ids, probs)) in enumerate(zip(reference, replay, strict=True)):
            pair = f"request {index} {ref_name} vs {name}"
            mismatch = next(
                (
                    (position, expected, actual)
                    for position, (expected, actual) in enumerate(zip(ref_ids, ids, strict=True), start=1)
                    if expected != actual
                ),
                None,
            )
            if mismatch is not None:
                position, expected, actual = mismatch
                failures.append(f"{pair} differs at generated token {position} ({expected} → {actual})")
                continue
            if probs["token_logprobs"] != pytest.approx(ref_probs["token_logprobs"], rel=0, abs=tolerance) or any(
                actual != pytest.approx(expected, rel=0, abs=tolerance)
                for expected, actual in zip(ref_probs["top_logprobs"], probs["top_logprobs"], strict=True)
            ):
                failures.append(f"{pair} has different logprobs")
    if failures:
        pytest.fail("\n".join(failures))
