"""Formal request timing and successful-request progress, without AISBench installed."""

import asyncio
import importlib.util
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.profiling import capture


@pytest.fixture
def inferencer_class(tmp_path, monkeypatch):
    class GenInferencer:
        async def warmup(self, requests):
            return await asyncio.gather(*(self.do_request(request, None, None) for request in requests))

        async def do_request(self, data, token_bucket, session):
            if data == "failed":
                raise RuntimeError("request failed")
            return data

        def inference_with_shm(self, requests, after_requests=lambda: None):
            self.status_counter = SimpleNamespace(get_req=0)
            for request in requests:
                asyncio.run(self.do_request(request, None, None))
                self.status_counter.get_req += request != "unsuccessful"
            after_requests()
            return self.status_counter.get_req

    name = "ais_bench.benchmark.openicl.icl_inferencer"
    dependency = ModuleType(name)
    monkeypatch.setattr(dependency, "GenInferencer", GenInferencer, raising=False)
    monkeypatch.setitem(sys.modules, name, dependency)
    spec = importlib.util.spec_from_file_location(
        "profiling_request_marker_under_test", Path(capture.__file__).with_name("aisbench_request_marker.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setenv("ASCEND_PROFILE_REQUEST_MARKER", str(tmp_path / "first_request"))
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", "true")
    return module.TimedGenInferencer


@pytest.mark.parametrize("warmup", [[], ["warmup"]])
@pytest.mark.parametrize("first", ["success", "failed"])
def test_first_formal_attempt_defines_origin_once(tmp_path, monkeypatch, inferencer_class, warmup, first):
    marker = tmp_path / "first_request"
    now = Mock(return_value=50.0)
    monkeypatch.setattr(capture, "time", SimpleNamespace(perf_counter=now))
    inferencer = inferencer_class()
    asyncio.run(inferencer.warmup(warmup))
    assert not marker.exists()
    now.return_value = 100.0
    with pytest.raises(RuntimeError) if first == "failed" else nullcontext():
        asyncio.run(inferencer.do_request(first, None, None))
    now.return_value = 200.0
    asyncio.run(inferencer_class().do_request("another worker", None, None))
    assert float(marker.read_text()) == 100.0


@pytest.mark.parametrize("profiling", [False, True])
def test_marker_error_is_fatal_only_for_profiling(tmp_path, monkeypatch, inferencer_class, caplog, profiling):
    monkeypatch.setenv("ASCEND_PROFILE_REQUEST_MARKER", str(tmp_path / "missing/first_request"))
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", str(profiling))
    inferencer = inferencer_class()
    with pytest.raises(OSError) if profiling else nullcontext():
        for _ in range(2):
            asyncio.run(inferencer.do_request("success", None, None))
    if not profiling:
        assert len(caplog.records) == 1 and "time origin unavailable" in caplog.text


@pytest.mark.parametrize("failed", [False, True])
def test_progress_counts_formal_successes_and_survives_request_errors(tmp_path, monkeypatch, inferencer_class, failed):
    monkeypatch.setenv("ASCEND_PROFILE_START_AFTER_REQUESTS", "2")
    progress = tmp_path / "completed_requests"
    progress.mkdir()
    inferencer = inferencer_class()
    asyncio.run(inferencer.warmup(["warmup", "warmup"]))
    assert not list(progress.iterdir())

    def while_running():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(int(path.read_text()) == 2 for path in progress.glob("*.count")):
                return
            time.sleep(0.001)
        pytest.fail("No progress reported during inference")

    with pytest.raises(RuntimeError) if failed else nullcontext():
        inferencer.inference_with_shm(["success", "unsuccessful", "failed" if failed else "success"], while_running)
    assert [int(path.read_text()) for path in progress.glob("*.count")] == [1 if failed else 2]


@pytest.mark.parametrize("enabled,request_trigger", [(False, True), (True, False)])
def test_other_modes_do_not_publish_counts(tmp_path, monkeypatch, inferencer_class, enabled, request_trigger):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", str(enabled))
    if request_trigger:
        monkeypatch.setenv("ASCEND_PROFILE_START_AFTER_REQUESTS", "2")
    assert inferencer_class().inference_with_shm(["success", "unsuccessful"]) == 1
    assert not (tmp_path / "completed_requests").exists()
