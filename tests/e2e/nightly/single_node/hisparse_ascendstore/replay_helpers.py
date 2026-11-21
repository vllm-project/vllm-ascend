# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""HTTP requests and counters shared by the A5 HiSparse replay checks."""

import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

LOCAL_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
EXTERNAL_GETS = "vllm:ascend_store_load_get_keys_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
COMPUTED_TOKENS = "local_compute"
HISPARSE_MISSES = "vllm:hisparse_cache_misses_total"


def _metrics(url):
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    totals = dict.fromkeys((LOCAL_HITS, EXTERNAL_HITS, EXTERNAL_GETS, DRAFTS, HISPARSE_MISSES), 0.0)
    for line in response.text.splitlines():
        sample, _, value = line.rpartition(" ")
        name = sample.partition("{")[0]
        if name in totals:
            totals[name] += float(value)
        elif name == "vllm:prompt_tokens_by_source_total" and 'source="local_compute"' in sample:
            totals[COMPUTED_TOKENS] = totals.get(COMPUTED_TOKENS, 0.0) + float(value)
    return totals


def _request(url, model, workload, output_tokens):
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
    assert len(choice["token_ids"]) == output_tokens
    for field in ("token_logprobs", "top_logprobs"):
        assert len(choice["logprobs"][field]) == output_tokens
    token_details = result["usage"].get("prompt_tokens_details") or {}
    choice["cached_tokens"] = token_details.get("cached_tokens")
    return choice


def reset_local_cache(url):
    for _ in range(30):
        response = requests.post(f"{url}/reset_prefix_cache", params={"reset_external": "false"}, timeout=30)
        response.raise_for_status()
        if response.json()["success"]:
            return
        time.sleep(0.2)
    pytest.fail("Local cache did not reset after the completed requests")


def make_workloads(prompt_length, concurrency, *, seed):
    rng = random.Random(seed)
    return [([rng.randrange(1000, 31000) for _ in range(prompt_length)], uuid.uuid4().hex) for _ in range(concurrency)]


def run_arm(url, model, workloads, *, output_tokens=32):
    before = _metrics(url)
    with ThreadPoolExecutor(max_workers=len(workloads)) as executor:
        futures = [executor.submit(_request, url, model, workload, output_tokens) for workload in workloads]
        choices = [future.result() for future in futures]
    after = _metrics(url)
    return choices, {key: after[key] - before[key] for key in before}


def assert_external_replay(warm, external, counts, prompt_length):
    assert counts[LOCAL_HITS] == 0 and counts[EXTERNAL_GETS] > 0, counts
    cached = []
    for reference, replay in zip(warm, external, strict=True):
        hit = replay["cached_tokens"]
        assert isinstance(hit, int) and hit > 0, "Enable --enable-prompt-tokens-details and verify each request hits"
        assert hit == reference["cached_tokens"], (hit, reference["cached_tokens"])
        cached.append(hit)
    assert counts[EXTERNAL_HITS] == sum(cached), counts
    assert counts[COMPUTED_TOKENS] == len(cached) * prompt_length - sum(cached), counts


def server_config(url):
    response = requests.get(f"{url}/server_info", params={"config_format": "json"}, timeout=30)
    response.raise_for_status()
    return response.json()["vllm_config"]


def require_mtp3(config):
    assert config["speculative_config"]["method"] == "mtp"
    assert config["speculative_config"]["num_speculative_tokens"] == 3
    assert config["scheduler_config"]["async_scheduling"] is True


def served_model(url):
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]
