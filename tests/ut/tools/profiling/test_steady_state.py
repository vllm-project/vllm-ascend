"""AISBench timing formats and the resulting concurrency reports."""

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
)


@pytest.mark.parametrize("database", [False, True])
def test_load_inline_and_database_timings(tmp_path, database):
    record = {"uuid": "request", "success": True, "time_points": [100, 105, 130]}
    if database:
        path = tmp_path / "db_data/timings.db"
        path.parent.mkdir()
        buffer = io.BytesIO()
        np.save(buffer, np.array(record["time_points"]))
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE numpy_store (id INTEGER PRIMARY KEY, arr_blob BLOB)")
            connection.execute("INSERT INTO numpy_store VALUES (?, ?)", (1, buffer.getvalue()))
        record.update(time_points={"__db_ref__": 1}, db_name="timings.db")
    (tmp_path / "perf_details.jsonl").write_text(json.dumps(record))
    (tmp_path / "other_details.jsonl").write_text("{}")
    loaded = AisbenchTimingAdapter(tmp_path, "perf").load_request_timings()
    assert loaded.timings == [RequestTiming("request", 100, 130, True)]
    assert loaded.stats.successful_timings == 1


def test_invalid_records_are_reported_without_losing_valid_requests(tmp_path, caplog):
    records = [
        {"id": "ok", "success": True, "time_points": [100, 130]},
        {"id": "failed", "success": False, "time_points": [90, 110]},
        {"time_points": [2, 1]},
        {"time_points": [0, float("inf")]},
    ]
    (tmp_path / "perf_details.jsonl").write_text("\n".join(map(json.dumps, records)) + "\ninvalid\n")
    loaded = AisbenchTimingAdapter(tmp_path, "perf").load_request_timings()
    assert (loaded.stats.records_read, loaded.stats.invalid_records, loaded.stats.successful_timings) == (5, 3, 1)
    assert len(loaded.timings) == 2
    assert len(caplog.records) == 1


@pytest.mark.parametrize("source", ["missing", "ambiguous", "missing_database"])
def test_unavailable_timing_sources(tmp_path, source):
    if source == "ambiguous":
        for name in ("a", "b"):
            (tmp_path / f"{name}_details.jsonl").write_text("{}")
    elif source == "missing_database":
        (tmp_path / "perf_details.jsonl").write_text(
            '{"success":true,"time_points":{"__db_ref__":1},"db_name":"absent.db"}'
        )
    with pytest.raises(TimingDataUnavailable):
        AisbenchTimingAdapter(tmp_path, "perf").load_request_timings()


def test_window_uses_first_successful_start_and_all_threshold_crossings():
    requests = [
        RequestTiming("failed", 90, 200, False),
        RequestTiming("b", 105, 130, True),
        RequestTiming("a", 100, 140, True),
        RequestTiming("c", 110, 125, True),
        RequestTiming("d", 150, 170, True),
        RequestTiming("e", 150, 165, True),
        RequestTiming("f", 151, 160, True),
    ]
    result = analyze_steady_state(requests, target_concurrency=3)
    assert (result.steady_start_s, result.steady_end_s, result.steady_duration_s) == (10, 60, 50)
    assert (result.completed_at_start, result.completed_at_end, result.observed_peak) == (0, 4, 3)
    assert (result.timeline[-1].time_s, result.timeline[-1].completed_requests) == (70, 6)
    assert result.warning and "failed requests" in result.warning
    summary = steady_state_summary("perf", result)
    assert summary["benchmark_name"] == "perf" and summary["status"] == "found"
    assert summary["steady_state"]["duration_s"] == 50
    report = render_terminal("perf", result)
    assert all(label in report for label in ("Concurrency Timeline", "Completed Requests", "Time zero:", "perf"))


def test_simultaneous_events_do_not_create_a_false_peak():
    result = analyze_steady_state(
        [RequestTiming("a", 0, 2, True), RequestTiming("b", 2, 4, True), RequestTiming("instant", 2, 2, True)],
        target_concurrency=2,
    )
    assert result.status == "not_found" and result.observed_peak == 1
    assert result.timeline[-1].completed_requests == 3
    assert steady_state_summary("perf", result)["steady_state"] is None


@pytest.mark.parametrize("requests", [[], [RequestTiming("failed", 1, 2, False)]])
def test_no_successful_requests_have_no_window(requests):
    result = analyze_steady_state(requests, target_concurrency=1)
    assert result.status == "unavailable" and result.steady_duration_s is None
    assert "UNAVAILABLE" in render_terminal("perf", result)


def test_short_window_warns():
    result = analyze_steady_state([RequestTiming("a", 100, 105, True)], target_concurrency=1)
    assert result.steady_duration_s == 5 and result.warning


@pytest.mark.parametrize("concurrency,ratio", [(0, 0.95), (1, 0), (1, 1.1)])
def test_invalid_analysis_parameters(concurrency, ratio):
    with pytest.raises(ValueError):
        analyze_steady_state([], concurrency, ratio)
