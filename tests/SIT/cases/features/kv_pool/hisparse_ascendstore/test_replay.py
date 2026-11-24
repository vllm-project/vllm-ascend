# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A5 C8/MTP=3 replay against two idle TP8 servers with matching model settings.

Enable VLLM_SERVER_DEV_MODE=1, prefix caching, and --enable-prompt-tokens-details
on both servers. Run from the repository root:

    pytest -sv tests/SIT/cases/features/kv_pool/hisparse_ascendstore \
        --hisparse-replay-url=http://store:8000 \
        --hisparse-replay-reference-url=http://reference:8000
"""

import random
import uuid

import pytest

from tests.SIT.cases.features.kv_pool.hisparse_ascendstore.replay_helpers import (
    DRAFTS,
    EXTERNAL_GETS,
    EXTERNAL_HITS,
    LOCAL_HITS,
    assert_external_replay,
    assert_matching_replay_config,
    require_mtp3,
    reset_local_cache,
    run_replay,
    served_model,
    server_config,
)

BLOCK_SIZE = 128
PROMPT_LENGTH = 16384
EXPECTED_HIT_TOKENS = PROMPT_LENGTH - BLOCK_SIZE


def _warm_local_prefix(url, model, prompt, salt):
    _, cold_counts = run_replay(url, model, prompt, salt)
    assert cold_counts[LOCAL_HITS] == cold_counts[EXTERNAL_HITS] == cold_counts[EXTERNAL_GETS] == 0
    local, counts = run_replay(url, model, prompt, salt)
    assert local["cached_tokens"] == counts[LOCAL_HITS] == EXPECTED_HIT_TOKENS
    assert counts[EXTERNAL_HITS] == counts[EXTERNAL_GETS] == 0, counts
    assert counts[DRAFTS] > 1, counts
    return local


def test_c8_mtp_external_replay(pytestconfig):
    store_url = pytestconfig.getoption("--hisparse-replay-url").rstrip("/")
    reference_url = pytestconfig.getoption("--hisparse-replay-reference-url").rstrip("/")
    if not store_url and not reference_url:
        pytest.skip("Requires dedicated A5 HiSparse + AscendStore and HiSparse-only servers")
    assert store_url and reference_url and store_url != reference_url, "Provide two distinct server URLs"

    config = server_config(store_url)
    reference_config = server_config(reference_url)
    assert_matching_replay_config(config, reference_config)
    assert config["parallel_config"]["tensor_parallel_size"] == 8
    assert config["cache_config"]["block_size"] == BLOCK_SIZE
    assert config["cache_config"]["cache_dtype"] == "fp8"
    assert config["attention_config"]["indexer_kv_dtype"] == "fp8"
    require_mtp3(config)
    assert reference_config["kv_transfer_config"]["kv_connector"] == "HiSparseConnector"
    transfer = config["kv_transfer_config"]
    assert transfer["kv_connector"] == "MultiConnector"
    assert transfer["kv_load_failure_policy"] == "fail"
    children = transfer["kv_connector_extra_config"]["connectors"]
    assert {child["kv_connector"] for child in children} == {"HiSparseConnector", "AscendStoreConnector"}
    store = next(
        child["kv_connector_extra_config"] for child in children if child["kv_connector"] == "AscendStoreConnector"
    )
    assert store["backend"] == "memcache"
    assert store["load_async"] is True

    rng = random.Random(PROMPT_LENGTH)
    prompt = [rng.randrange(1000, 31000) for _ in range(PROMPT_LENGTH)]
    salt = uuid.uuid4().hex
    model = served_model(store_url)
    reference = _warm_local_prefix(reference_url, served_model(reference_url), prompt, salt)
    store_local = _warm_local_prefix(store_url, model, prompt, salt)
    reset_local_cache(store_url)
    external, counts = run_replay(store_url, model, prompt, salt)
    assert_external_replay(store_local, external, counts, PROMPT_LENGTH)
    assert counts[DRAFTS] > 1, counts
    for name, replay in (("AscendStore local hit", store_local), ("AscendStore external hit", external)):
        assert replay["token_ids"] == reference["token_ids"], name
        for field in ("token_logprobs", "top_logprobs"):
            assert replay["logprobs"][field] == reference["logprobs"][field], (name, field)
