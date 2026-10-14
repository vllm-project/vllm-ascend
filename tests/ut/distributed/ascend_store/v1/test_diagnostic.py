# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the temporary diagnostic; these do not validate NPU output or KV transfer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.pull_request.one_card.ascendstore_v1_diagnostic import (
    AscendStoreDiagnosticWorkerExtension,
    compare_phases,
    first_divergence,
    request_index,
    snapshot_batch,
    token_batch_context,
)


def make_batch(computed: int = 0, scheduled: int = 8193) -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["cmpl-ascendstore-diag-v1-c8-natural-r1-cold-p6-0-uuid"],
        num_reqs=np.int64(1),
        num_reqs_after_padding=np.int64(2),
        num_tokens=np.int64(scheduled),
        num_tokens_after_padding=np.int64(scheduled + 1),
        idx_mapping_np=np.array([4, 99]),
        num_scheduled_tokens=np.array([scheduled, 99]),
        num_computed_tokens_np=np.array([computed, 99]),
        prefill_len_np=np.array([8193, 99]),
        query_start_loc_np=np.array([0, scheduled, 99]),
        is_prefilling_np=np.array([computed < 8193, False]),
        attn_state=SimpleNamespace(name="PrefillNoCache"),
    )


def test_batch_snapshot_copies_only_real_rows_and_is_json_serializable() -> None:
    batch = make_batch()
    record = snapshot_batch(batch, 2)
    batch.num_computed_tokens_np[0] = 99
    batch.req_ids[0] = "next-request"
    assert record["computed_tokens"] == [0]
    assert record["request_state_indices"] == [4]
    assert record["query_start_loc"] == [0, 8193]
    assert record["request_ids"][0].endswith("-p6-0-uuid")
    assert record["num_reqs_after_padding"] == 2
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize("computed,scheduled,position", ((0, 8193, 0), (8192, 1, 0), (8193, 1, 1), (8223, 1, 31)))
def test_batch_context_maps_cold_external_and_decode_steps(computed: int, scheduled: int, position: int) -> None:
    records = [snapshot_batch(make_batch(computed, scheduled), 0)]
    assert token_batch_context(records, 6, position)["model_row"] == 0
    assert token_batch_context(records, 6, position + 1) is None
    assert token_batch_context(records, 4, position) is None


@pytest.mark.parametrize("request_id", ("ascendstore-diag-none-c1-r1-cold-p6", "cmpl-ascendstore-diag-v1-c1-p6-0-uuid"))
def test_request_identity_survives_server_prefix_and_suffix(request_id: str) -> None:
    assert request_index(request_id) == 6


def test_unknown_request_identity_fails_instead_of_being_misattributed() -> None:
    with pytest.raises(AssertionError, match="Unrecognized"):
        request_index("another-client-p6")


def test_trace_hook_restores_worker_and_propagates_preparation_errors() -> None:
    worker = AscendStoreDiagnosticWorkerExtension()
    batch = make_batch()
    runner = SimpleNamespace(_prepare_inputs_impl=lambda: batch)
    worker.model_runner = runner
    original = runner._prepare_inputs_impl
    worker.start_ascendstore_diagnostic_trace()
    with pytest.raises(AssertionError, match="previous trace"):
        worker.start_ascendstore_diagnostic_trace()
    assert runner._prepare_inputs_impl() is batch
    result = worker.stop_ascendstore_diagnostic_trace()
    assert result["trace_restored"] is True and len(result["batches"]) == 1
    assert runner._prepare_inputs_impl is original
    with pytest.raises(AssertionError, match="No diagnostic trace"):
        worker.stop_ascendstore_diagnostic_trace()

    def fail() -> None:
        raise ValueError("input preparation failed")

    runner._prepare_inputs_impl = fail
    worker.start_ascendstore_diagnostic_trace()
    try:
        with pytest.raises(ValueError, match="input preparation"):
            runner._prepare_inputs_impl()
    finally:
        assert worker.stop_ascendstore_diagnostic_trace()["batches"] == []
    assert runner._prepare_inputs_impl is fail


def test_first_divergence_uses_token_ids_and_only_equal_prefix_scores() -> None:
    left = {"prompt_sha256": "same", "text": "same decoded text", "token_ids": [1, 2, 3], "logprobs": None}
    right = {
        **left,
        "token_ids": [1, 9, 8],
        "logprobs": {"top_logprobs": [{}, {"token_id:9": -0.1, "token_id:2": -0.11}, {}]},
    }
    result = first_divergence(left, right)
    assert result["token_position"] == 1 and result["shared_prefix_token_ids"] == [1]
    assert result["left"]["top_logprobs"] is None
    assert result["right"]["top_two_logprob_gap"] == pytest.approx(0.01)
    assert first_divergence(left, {**left, "text": "different presentation"}) is None
    with pytest.raises(AssertionError, match="Mismatched prompts"):
        first_divergence(left, {**right, "prompt_sha256": "different"})
    with pytest.raises(AssertionError, match="Truncated"):
        first_divergence(left, {**right, "token_ids": [1]})


def test_comparisons_keep_repeats_connectors_and_external_load_separate() -> None:
    phases = []
    for implementation in ("none", "production", "v1"):
        for source in ("cold", "external") if implementation != "none" else ("cold",):
            for repeat in (1, 2):
                token = 9 if implementation == "v1" and source == "external" else 1
                phases.append(
                    {
                        "id": f"{implementation}-c1-r{repeat}-{source}",
                        "implementation": implementation,
                        "case": "c1",
                        "repeat": repeat,
                        "source": source,
                        "batches": [snapshot_batch(make_batch(), 0)],
                        "observations": [{"request_index": 6, "prompt_sha256": "same", "token_ids": [token]}],
                    }
                )
    comparisons = compare_phases(phases)
    assert {row["contrast"] for row in comparisons} == {"repeat", "connector", "production-v1", "cold-external"}
    assert all(not row["differences"] for row in comparisons if row["contrast"] == "repeat")
    mismatch = next(row for row in comparisons if row["contrast"] == "cold-external" and row["right"].startswith("v1"))
    difference = mismatch["differences"][0]
    assert difference["request_index"] == 6 and difference["token_position"] == 0
    assert difference["right"]["batch"]["model_row"] == 0
