# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from tools.pd_results import aggregate_decode_windows


def _window(requests, duration_ms, latency_ms):
    latency = f"{latency_ms} ms"
    row = {"Stage": "total", "Average": latency, "Min": latency, "Max": latency, "N": requests}
    frame = pd.DataFrame([row, row], index=["TTFT", "TPOT"])
    metrics = {
        "Benchmark Duration": f"{duration_ms} ms",
        "Total Requests": requests,
        "Success Requests": requests,
        "Failed Requests": 0,
        "Total Input Tokens": requests * 16384,
        "Total Generated Tokens": requests * 1024,
        "Max Concurrency": requests,
    }
    return frame, {key: {"total": value} for key, value in metrics.items()}


def test_aggregate_measured_windows():
    frame, summary = aggregate_decode_windows([_window(4, 1000, 10), _window(2, 2000, 40)])

    assert summary["Benchmark Duration"]["total"] == "3000.0000 ms"
    assert summary["Total Requests"]["total"] == 6
    assert summary["Request Throughput"]["total"] == "2.0000 req/s"
    assert summary["Output Token Throughput"]["total"] == "2048.0000 token/s"
    assert summary["Input Token Throughput"]["total"] == "32768.0000 token/s"
    assert summary["Max Concurrency"]["total"] == 4
    assert frame.loc["TPOT", "Average"] == "20.0000 ms"
    assert frame.loc["TTFT", "Max"] == "40.0000 ms"
    assert frame.loc["TPOT", "N"] == 6
    assert "P99" not in frame.columns


def test_reject_incomplete_latency_samples():
    frame, summary = _window(4, 1000, 10)
    frame.loc["TPOT", "N"] = 3
    with pytest.raises(ValueError, match="TPOT sample count"):
        aggregate_decode_windows([(frame, summary)])
