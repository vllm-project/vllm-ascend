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
def serving_instance(pytestconfig):
    url = pytestconfig.getoption("--hisparse-replay-url").rstrip("/")
    if not url:
        pytest.skip("Pass --hisparse-replay-url to run against a dedicated A5 test server")
    mode = pytestconfig.getoption("--hisparse-replay-mode")
    config = server_config(url)
    transfer = config["kv_transfer_config"]
    names = set()
    if transfer:
        connectors = (
            transfer["kv_connector_extra_config"]["connectors"]
            if transfer["kv_connector"] == "MultiConnector"
            else [transfer]
        )
        names = {child["kv_connector"] for child in connectors}
    expected = {
        "plain": set(),
        "only": {"HiSparseConnector"},
        "plain-store": {"AscendStoreConnector"},
        "store": {"HiSparseConnector", "AscendStoreConnector"},
    }
    assert names == expected[mode], f"Replay mode {mode!r} does not match server connectors {names}"
    if mode in {"store", "plain-store"}:
        assert transfer["kv_load_failure_policy"] == "fail"
    if pytestconfig.getoption("--hisparse-replay-mtp"):
        require_mtp3(config)
    return url, served_model(url), mode


@pytest.mark.parametrize("prompt_length", [16383, 16384, 16385, 131072])
@pytest.mark.parametrize("concurrency", [1, 2])
def test_hisparse_prefix_replay(serving_instance, prompt_length, concurrency, pytestconfig):
    if prompt_length == 131072 and not pytestconfig.getoption("--hisparse-replay-long"):
        pytest.skip("Pass --hisparse-replay-long for the 128K case")
    url, model, mode = serving_instance
    output_tokens = pytestconfig.getoption("--hisparse-replay-output-tokens")
    workloads = make_workloads(prompt_length, concurrency, seed=16384)

    cold, cold_counts = run_arm(url, model, workloads, output_tokens=output_tokens)
    local, local_counts = run_arm(url, model, workloads, output_tokens=output_tokens)
    assert cold_counts[LOCAL_HITS] == cold_counts[EXTERNAL_HITS] == cold_counts[EXTERNAL_GETS] == 0
    assert local_counts[LOCAL_HITS] > 0
    mixed = bool(local_counts[EXTERNAL_HITS] or local_counts[EXTERNAL_GETS])
    if mode != "plain-store":
        assert not mixed, local_counts
    arms = [("local+external" if mixed else "local", local, local_counts)]
    if mode in {"store", "plain-store"}:
        reset_local_cache(url)
        external, external_counts = run_arm(url, model, workloads, output_tokens=output_tokens)
        assert_external_replay(local, external, external_counts, prompt_length)
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
        for index, (expected_choice, actual_choice) in enumerate(zip(reference, replay, strict=True)):
            ref_ids, ref_probs = expected_choice["token_ids"], expected_choice["logprobs"]
            ids, probs = actual_choice["token_ids"], actual_choice["logprobs"]
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
