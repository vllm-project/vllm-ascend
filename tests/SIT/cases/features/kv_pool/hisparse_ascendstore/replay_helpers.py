# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""HTTP requests and counters shared by the A5 HiSparse replay checks."""

import time

import pytest
import requests

LOCAL_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
EXTERNAL_GETS = "vllm:ascend_store_load_get_keys_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
COMPUTED_TOKENS = "local_compute"


def _metrics(url):
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    totals = dict.fromkeys((LOCAL_HITS, EXTERNAL_HITS, EXTERNAL_GETS, DRAFTS, COMPUTED_TOKENS), 0.0)
    for line in response.text.splitlines():
        sample, _, value = line.rpartition(" ")
        name = sample.partition("{")[0]
        if name in totals:
            totals[name] += float(value)
        elif name == "vllm:prompt_tokens_by_source_total" and 'source="local_compute"' in sample:
            totals[COMPUTED_TOKENS] += float(value)
    return totals


def _request(url, model, prompt, salt, output_tokens):
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


def run_replay(url, model, prompt, salt, *, output_tokens=32):
    before = _metrics(url)
    choice = _request(url, model, prompt, salt, output_tokens)
    after = _metrics(url)
    return choice, {key: after[key] - before[key] for key in before}


def assert_external_replay(local, external, counts, prompt_length):
    assert counts[LOCAL_HITS] == 0 and counts[EXTERNAL_GETS] > 0, counts
    hit = external["cached_tokens"]
    assert isinstance(hit, int) and hit > 0, "Enable --enable-prompt-tokens-details and verify external hits"
    assert hit == local["cached_tokens"], (hit, local["cached_tokens"])
    assert counts[EXTERNAL_HITS] == hit, counts
    assert counts[COMPUTED_TOKENS] == prompt_length - hit, counts


def server_config(url):
    response = requests.get(f"{url}/server_info", params={"config_format": "json"}, timeout=30)
    response.raise_for_status()
    return response.json()["vllm_config"]


def require_mtp3(config):
    assert config["speculative_config"]["method"] == "mtp"
    assert config["speculative_config"]["num_speculative_tokens"] == 3
    assert config["scheduler_config"]["async_scheduling"] is True


def assert_matching_replay_config(config, reference_config):
    # Core's embedded target and draft parallel configs also contain per-server
    # deployment settings. Compare the replay settings explicitly.
    for section, fields in (
        (
            "model_config",
            ("model", "model_weights", "revision", "code_revision", "dtype", "quantization", "enforce_eager"),
        ),
        ("attention_config", ("hisparse_config", "indexer_kv_dtype")),
        ("cache_config", ("block_size", "cache_dtype", "prefix_match_unit")),
        ("compilation_config", ("cudagraph_mode",)),
        ("parallel_config", ("tensor_parallel_size", "decode_context_parallel_size")),
        ("scheduler_config", ("async_scheduling", "max_num_batched_tokens")),
        (
            "speculative_config",
            (
                "method",
                "num_speculative_tokens",
                "num_speculative_tokens_per_batch_size",
                "model",
                "revision",
                "code_revision",
                "quantization",
                "enforce_eager",
                "draft_tensor_parallel_size",
                "draft_decode_context_parallel_size",
                "kv_cache_dtype",
                "draft_lm_head_quantization",
                "draft_sample_method",
                "rejection_sample_method",
                "parallel_drafting",
                "index_share_for_mtp_iteration",
                "use_local_argmax_reduction",
                "attention_backend",
                "moe_backend",
            ),
        ),
    ):
        settings = config[section]
        reference = reference_config[section]
        assert (settings is None) == (reference is None), f"{section} is enabled on only one server"
        if settings is not None:
            for field in fields:
                assert settings.get(field) == reference.get(field), f"{section}.{field} differs"


def served_model(url):
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]
