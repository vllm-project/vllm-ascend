"""AISBench timing inputs, concurrency analysis, and report outputs."""

import io
import json
import sqlite3

import numpy as np
import pytest

from tools.profiling.steady_state import (
    AisbenchTimingAdapter,
    RequestTiming,
    TimingDataUnavailable,
    analyze_steady_state,
    render_terminal,
    steady_state_summary,
    unavailable_steady_state,
)


@pytest.mark.parametrize("source", ["inline", "sqlite"])
def test_load_request_timings(tmp_path, source):
    record = {"uuid": "request-1", "success": True, "time_points": [100, 105, 120]}
    if source == "sqlite":
        database = tmp_path / "db_data" / "timings.db"
        database.parent.mkdir()
        blob = io.BytesIO()
        np.save(blob, np.array(record["time_points"]))
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE numpy_store (id INTEGER PRIMARY KEY, arr_blob BLOB)")
            connection.execute("INSERT INTO numpy_store VALUES (?, ?)", (1, blob.getvalue()))
        record.update(time_points={"__db_ref__": 1}, db_name="timings.db")
    (tmp_path / "load_details.jsonl").write_text(json.dumps(record))
    (tmp_path / "unrelated_details.jsonl").write_text("{}")
    result = AisbenchTimingAdapter(tmp_path, "load").load_request_timings()
    assert result.timings == [RequestTiming("request-1", 100, 120, True)]
    assert result.stats.records_read == result.stats.valid_timings == result.stats.successful_timings == 1
    assert result.stats.invalid_records == 0


def test_malformed_records_are_skipped_with_one_diagnostic(tmp_path, caplog):
    records = [
        {"id": "ok", "success": True, "time_points": [1, 2]},
        {"id": "failed", "success": False, "time_points": [2, 3]},
        {"time_points": [5, 4]},
        {"time_points": [1]},
        {"time_points": [0, float("inf")]},
    ]
    (tmp_path / "load_details.jsonl").write_text("\n".join(map(json.dumps, records)) + "\ninvalid json\n\n")
    result = AisbenchTimingAdapter(tmp_path, "load").load_request_timings()
    assert result.timings == [RequestTiming("ok", 1, 2, True), RequestTiming("failed", 2, 3, False)]
    assert (result.stats.records_read, result.stats.valid_timings, result.stats.invalid_records) == (6, 2, 4)
    assert result.stats.successful_timings == 1
    assert len(caplog.records) == 1
    assert "4 invalid" in caplog.text


def test_single_details_file_can_be_used_without_matching_dataset_name(tmp_path):
    (tmp_path / "renamed_details.jsonl").write_text('{"success":true,"time_points":[1,2]}')
    result = AisbenchTimingAdapter(tmp_path, "load").load_request_timings()
    assert result.stats.successful_timings == 1


@pytest.mark.parametrize("source", ["missing", "ambiguous", "missing_database"])
def test_unavailable_timing_sources_are_reported(tmp_path, source):
    if source == "ambiguous":
        for name in ("a", "b"):
            (tmp_path / f"{name}_details.jsonl").write_text("{}")
    elif source == "missing_database":
        (tmp_path / "load_details.jsonl").write_text(
            '{"time_points":{"__db_ref__":1},"db_name":"missing.db","success":true}'
        )
    with pytest.raises(TimingDataUnavailable):
        AisbenchTimingAdapter(tmp_path, "load").load_request_timings()


def test_window_spans_first_and_last_threshold_crossings_relative_to_benchmark_origin():
    requests = [
        RequestTiming("failed", 99, 200, False),
        RequestTiming("a", 100, 140, True),
        RequestTiming("b", 105, 130, True),
        RequestTiming("c", 110, 125, True),
        RequestTiming("d", 150, 170, True),
        RequestTiming("e", 150, 165, True),
        RequestTiming("f", 151, 160, True),
    ]
    result = analyze_steady_state(requests, target_concurrency=3, benchmark_zero=99)
    assert result.status == "found"
    assert (result.total_requests, result.successful_requests, result.observed_peak) == (7, 6, 3)
    assert (result.steady_start_s, result.steady_end_s, result.steady_duration_s) == (11, 61, 50)
    assert (result.completed_at_start, result.completed_at_end) == (0, 4)
    assert result.timeline[0].time_s == 0
    assert (result.timeline[-1].running_requests, result.timeline[-1].completed_requests) == (0, 6)
    assert result.warning is None


def test_simultaneous_completions_and_starts_do_not_create_a_concurrency_peak():
    result = analyze_steady_state(
        [RequestTiming("a", 0, 2, True), RequestTiming("b", 2, 4, True), RequestTiming("instant", 2, 2, True)],
        target_concurrency=2,
        benchmark_zero=0,
    )
    assert result.status == "not_found"
    assert result.observed_peak == 1
    assert result.steady_start_s is result.steady_end_s is None
    assert result.timeline[-1].completed_requests == 3


def test_short_window_is_reported_with_warning():
    result = analyze_steady_state(
        [RequestTiming("a", 100, 105, True), RequestTiming("b", 100, 105, True)],
        target_concurrency=4,
        threshold_ratio=0.5,
        benchmark_zero=100,
    )
    assert result.status == "found"
    assert result.threshold_concurrency == 2
    assert result.steady_duration_s == 5
    assert result.warning


@pytest.mark.parametrize("requests", [[], [RequestTiming("failed", 1, 2, False)]])
def test_no_successful_requests_produces_unavailable_result(requests):
    result = analyze_steady_state(requests, target_concurrency=2, benchmark_zero=1)
    assert result.status == "unavailable"
    assert result.successful_requests == 0
    assert result.steady_duration_s is None
    assert result.reason


@pytest.mark.parametrize("concurrency,ratio", [(0, 0.95), (2, 0), (2, 1.1)])
def test_analysis_rejects_invalid_parameters(concurrency, ratio):
    with pytest.raises(ValueError):
        analyze_steady_state([], concurrency, ratio, benchmark_zero=0)


@pytest.mark.parametrize("status", ["found", "not_found", "unavailable"])
def test_reports_expose_benchmark_identity_and_analysis_status(status):
    if status == "unavailable":
        result = unavailable_steady_state(target_concurrency=2, reason="timings missing")
    else:
        result = analyze_steady_state(
            [RequestTiming("a", 100, 130, True), RequestTiming("b", 105, 125, True)],
            target_concurrency=2 if status == "found" else 3,
            benchmark_zero=100,
        )
    summary = steady_state_summary("throughput", result)
    assert summary["benchmark_name"] == "throughput"
    assert summary["status"] == status
    assert "first formal AISBench request" in summary["time_origin"]
    if status == "found":
        assert summary["steady_state"]["duration_s"] == 20
    else:
        assert summary["steady_state"] is None
    output = render_terminal("throughput", result)
    assert "throughput" in output and status.upper() in output
    assert "Time zero:" in output
