# SPDX-License-Identifier: Apache-2.0
"""Tests for the numerical comparison logic (pure CPU, synthetic dumps).

Fail-closed semantics: disjoint top-k sets, identical run identity, mismatched
checkpoint ids, non-finite tolerances and empty overlaps all fail.
"""

import json

import pytest

from tools.glm_reduced.compare import DUMP_FORMAT, Dump, compare_dumps, load_dump

META_BASE = {
    "format": DUMP_FORMAT,
    "model": "/models/glm-reduced",
    "checkpoint_id": "sha256:abc",
    "run_id": "run-A",
    "dtype": "bfloat16",
    "mode": "eager",
    "seed": 0,
    "prompt_count": 1,
    "output_tokens": 2,
    "runtime": {"vllm": "0.13.0"},
    "engine": {"enforce_eager": True},
}


def _record(token_ids=(5, 6), logprobs=((-0.5, -1.0), (-0.25, -2.0))):
    top = []
    for position, values in enumerate(logprobs):
        top.append(
            [
                {"token_id": token_ids[position], "logprob": values[0]},
                {"token_id": 999, "logprob": values[1]},
            ]
        )
    return {"prompt_index": 0, "prompt_token_ids": [1, 2, 3], "token_ids": list(token_ids), "top_logprobs": top}


def _write(path, meta, records):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


def _candidate_meta(**overrides):
    return {**META_BASE, "run_id": "run-B", "mode": "graph", "engine": {"enforce_eager": False}, **overrides}


def test_identical_values_pass_with_tolerances(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1e-3, rtol=1e-3)
    assert report.ok, report.problems
    assert report.compared_positions > 0


def test_small_drift_within_tolerance_passes(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    shifted = _record(logprobs=((-0.5005, -1.0), (-0.25, -2.0)))
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [shifted])
    assert compare_dumps(load_dump(baseline), load_dump(candidate), atol=1e-3, rtol=1e-3).ok
    assert not compare_dumps(load_dump(baseline), load_dump(candidate), atol=1e-6, rtol=1e-6).ok


def test_same_checkpoint_across_runtime_revisions_is_valid(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    meta = _candidate_meta(mode="eager", engine={"enforce_eager": True}, runtime={"vllm": "0.14.0"})
    candidate = _write(tmp_path / "c.jsonl", meta, [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1e-3, rtol=1e-3)
    assert report.ok, report.problems


def test_independent_aa_runs_with_identical_metadata_are_valid(tmp_path):
    # Two independent runs of the same checkpoint with identical settings are
    # valid noise characterization; only the run identity must differ.
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    meta = {**META_BASE, "run_id": "run-B"}
    candidate = _write(tmp_path / "c.jsonl", meta, [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1e-3, rtol=1e-3)
    assert report.ok, report.problems


def test_same_run_id_rejected(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    candidate = _write(tmp_path / "c.jsonl", dict(META_BASE), [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1.0, rtol=1.0)
    assert not report.ok
    assert any("same run_id" in problem for problem in report.problems)


def test_truncated_output_rejected_even_when_both_sides_truncate(tmp_path):
    # Independent-review reproduction: meta declares output_tokens=32 but only
    # 1 position is present, on BOTH sides. Must fail closed.
    meta = {**META_BASE, "output_tokens": 32, "prompt_count": 1}
    short_record = {
        "prompt_index": 0,
        "prompt_token_ids": [1],
        "token_ids": [2],
        "top_logprobs": [[{"token_id": 2, "logprob": -1.0}]],
    }
    baseline = Dump(meta, [short_record])
    candidate = Dump({**meta, "run_id": "run-B"}, [short_record])
    report = compare_dumps(baseline, candidate, atol=0.0, rtol=0.0)
    assert not report.ok
    assert any("truncated output" in problem for problem in report.problems)


def test_record_count_and_prompt_index_validation(tmp_path):
    meta = {**META_BASE, "prompt_count": 2}
    dump = Dump(meta, [_record()])  # 1 record for declared 2
    report = compare_dumps(dump, Dump({**meta, "run_id": "run-B"}, [_record()]), atol=1.0, rtol=1.0)
    assert not report.ok
    assert any("truncated dump" in problem for problem in report.problems)
    dup = Dump(META_BASE, [_record()])
    dup.records[0]["prompt_index"] = 5  # outside 0..prompt_count-1
    report = compare_dumps(dup, Dump({**META_BASE, "run_id": "run-B"}, [_record()]), atol=1.0, rtol=1.0)
    assert not report.ok
    assert any("prompt_index" in problem for problem in report.problems)


def test_checkpoint_id_mismatch_rejected(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(checkpoint_id="sha256:other"), [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1.0, rtol=1.0)
    assert not report.ok
    assert any("checkpoint_id mismatch" in problem for problem in report.problems)


def test_disjoint_topk_fails_closed(tmp_path):
    # Independent-review reproduction: identical token ids but disjoint top-k
    # must NOT yield a green comparison with zero compared positions.
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record(token_ids=(5, 6))])
    bad = _record(token_ids=(5, 6))
    bad["top_logprobs"] = [
        [{"token_id": 5, "logprob": -1.0}, {"token_id": 100, "logprob": -2.0}],
        [{"token_id": 6, "logprob": -0.5}, {"token_id": 101, "logprob": -3.0}],
    ]
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [bad])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=0.0, rtol=0.0)
    assert not report.ok


def test_compared_positions_zero_fails(tmp_path):
    record = _record()
    # Baseline top-k fully disjoint from candidate's (except sampled token
    # missing on both sides is caught separately) — force zero overlap.
    record["top_logprobs"] = [[{"token_id": 5, "logprob": -1.0}], [{"token_id": 6, "logprob": -1.0}]]
    cand = _record()
    cand["top_logprobs"] = [[{"token_id": 5, "logprob": -1.5}], [{"token_id": 6, "logprob": -1.5}]]
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [cand])
    baseline2 = _write(tmp_path / "b2.jsonl", META_BASE, [record])
    report = compare_dumps(load_dump(baseline2), load_dump(candidate), atol=0.0, rtol=0.0)
    assert not report.ok


def test_token_sequence_divergence_reported(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record(token_ids=(5, 6))])
    candidate_record = _record(token_ids=(5, 7))
    candidate_record["top_logprobs"][1][0]["token_id"] = 7
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [candidate_record])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1.0, rtol=1.0)
    assert not report.ok
    assert report.diverged_prompts == [0]
    assert any("position 1" in problem for problem in report.problems)


def test_non_finite_logprob_fails(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        bad = _record()
        bad["top_logprobs"][0][0]["logprob"] = bad_value
        candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [bad])
        report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1.0, rtol=1.0)
        assert not report.ok, bad_value
        assert any("non-finite" in problem for problem in report.problems)


@pytest.mark.parametrize("atol,rtol", [(float("nan"), 0.0), (0.0, float("inf")), (-1.0, 0.0)])
def test_non_finite_or_negative_tolerances_rejected(tmp_path, atol, rtol):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(), [_record()])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=atol, rtol=rtol)
    assert not report.ok
    assert any("tolerances" in problem for problem in report.problems)


def test_workload_mismatch_fails(tmp_path):
    baseline = _write(tmp_path / "b.jsonl", META_BASE, [_record()])
    long_record = _record(token_ids=(5, 6, 7, 8), logprobs=((-0.5, -1.0), (-0.25, -2.0), (-0.3, -1.5), (-0.4, -1.6)))
    candidate = _write(tmp_path / "c.jsonl", _candidate_meta(output_tokens=4), [long_record])
    report = compare_dumps(load_dump(baseline), load_dump(candidate), atol=1.0, rtol=1.0)
    assert not report.ok
    assert any("workload mismatch" in problem for problem in report.problems)


def test_missing_baseline_file_and_empty_dump(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dump(tmp_path / "nope.jsonl")
    empty = _write(tmp_path / "empty.jsonl", META_BASE, [])
    with pytest.raises(ValueError, match="truncated dump"):
        load_dump(empty)


def test_missing_format_header_and_meta_keys_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata header"):
        load_dump(path)
    path2 = tmp_path / "bad2.jsonl"
    meta = {k: v for k, v in META_BASE.items() if k != "checkpoint_id"}
    _write(path2, meta, [_record()])
    with pytest.raises(ValueError, match="checkpoint_id"):
        load_dump(path2)
