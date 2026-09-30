# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("recorded_calls", [4, 5, 6])
def test_profile_boundaries_and_call_count(monkeypatch, tmp_path, recorded_calls):
    source = Path(__file__).resolve().parents[3] / "benchmarks" / "mhc_expand.py"
    spec = importlib.util.spec_from_file_location("mhc_expand_benchmark", source)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    events = []
    trace_dir = tmp_path / "trace"
    profiler = MagicMock()
    step = SimpleNamespace(step=lambda: events.append("step"))

    @contextmanager
    def profile(**kwargs):
        yield step
        output = trace_dir / "sample_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        (output / "op_statistic.csv").write_text(f"Count,Total Time(us)\n{recorded_calls},50\n")

    profiler.profile.side_effect = profile
    monkeypatch.setattr(benchmark, "torch_npu", SimpleNamespace(profiler=profiler))
    monkeypatch.setattr(
        benchmark, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync")))
    )
    fn = lambda: events.append("launch")
    if recorded_calls == benchmark.ACTIVE:
        assert benchmark.profile_case(fn, trace_dir) == 10
    else:
        with pytest.raises(RuntimeError, match=f"recorded {recorded_calls}"):
            benchmark.profile_case(fn, trace_dir)
    assert events == ["sync"] + ["launch", "sync", "step"] * (benchmark.WARMUP + benchmark.ACTIVE) + ["sync"]
