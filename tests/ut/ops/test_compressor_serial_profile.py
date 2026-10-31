# SPDX-License-Identifier: Apache-2.0
import csv

import pytest

from benchmarks.summarize_compressor_serial import summarize


def write_trace(path, *, unknown=False, samples=10):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Task Start Time(us)", "Task Duration(us)", "Op Name"])
        clock = 0
        for _ in range(samples):
            for name, duration in [
                ("ArgMax", 1),
                ("serial_cast", 2),
                ("MatMul", 6),
                ("serial_pool", 1),
                ("unknown" if unknown else "serial_finalize", 1),
            ]:
                writer.writerow([clock, duration, name])
                clock += duration


def test_complete_sequence_sum_and_share(tmp_path):
    path = tmp_path / "summary.csv"
    write_trace(path)
    result = summarize(path, "serial", "MatMul", "serial_")
    assert result["total_us"] == [10] * 10
    assert result["median_cube_share"] == 0.6
    assert result["tasks_per_sample"] == 4


@pytest.mark.parametrize("unknown,samples", [(True, 10), (False, 9)])
def test_rejects_missing_or_unclassified_tasks(tmp_path, unknown, samples):
    path = tmp_path / "summary.csv"
    write_trace(path, unknown=unknown, samples=samples)
    with pytest.raises(ValueError):
        summarize(path, "serial", "MatMul", "serial_")
