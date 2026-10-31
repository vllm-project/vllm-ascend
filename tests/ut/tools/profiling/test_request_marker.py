"""The profiling adapter excludes warmup and publishes formal success counts."""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.profiling import capture


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    class GenInferencer:
        async def warmup(self, requests):
            for request in requests:
                await self.do_request(request, None, None)

        async def do_request(self, data, token_bucket, session):
            return data

        def inference_with_shm(self, requests, during=lambda: None):
            self.status_counter = SimpleNamespace(get_req=0)
            for success in requests:
                asyncio.run(self.do_request(success, None, None))
                self.status_counter.get_req += int(success)
            during()
            return self.status_counter.get_req

    name = "ais_bench.benchmark.openicl.icl_inferencer"
    dependency = ModuleType(name)
    monkeypatch.setattr(dependency, "GenInferencer", GenInferencer, raising=False)
    monkeypatch.setitem(sys.modules, name, dependency)
    spec = importlib.util.spec_from_file_location(
        "request_marker_test", Path(capture.__file__).with_name("aisbench_request_marker.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setenv("ASCEND_PROFILE_REQUEST_MARKER", str(tmp_path / "first_request"))
    return module.TimedGenInferencer


@pytest.mark.parametrize("warmup", [[], [True, True]])
def test_first_formal_request_records_one_shared_origin(tmp_path, monkeypatch, adapter, warmup):
    first = adapter()
    asyncio.run(first.warmup(warmup))
    marker = tmp_path / "first_request"
    assert not marker.exists()
    monkeypatch.setattr(capture, "time", SimpleNamespace(perf_counter=lambda: 100))
    asyncio.run(first.do_request(True, None, None))
    monkeypatch.setattr(capture, "time", SimpleNamespace(perf_counter=lambda: 200))
    asyncio.run(adapter().do_request(True, None, None))
    assert marker.read_text() == "100"


def test_marker_failure_is_not_silently_ignored(tmp_path, monkeypatch, adapter):
    monkeypatch.setenv("ASCEND_PROFILE_REQUEST_MARKER", str(tmp_path / "missing/marker"))
    with pytest.raises(OSError):
        asyncio.run(adapter().do_request(True, None, None))


def test_request_trigger_publishes_only_formal_successes(tmp_path, monkeypatch, adapter):
    monkeypatch.setenv("ASCEND_PROFILE_START_AFTER_REQUESTS", "2")
    progress = tmp_path / "completed_requests"
    progress.mkdir()
    inferencer = adapter()
    asyncio.run(inferencer.warmup([True, True]))

    def during():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(path.read_text() == "2" for path in progress.glob("*.count")):
                return
            time.sleep(0.001)
        pytest.fail("The worker did not publish progress during inference")

    assert inferencer.inference_with_shm([True, False, True], during) == 2
    assert [path.read_text() for path in progress.glob("*.count")] == ["2"]


def test_time_trigger_needs_no_request_counter_directory(tmp_path, adapter):
    assert adapter().inference_with_shm([True, False]) == 1
    assert not (tmp_path / "completed_requests").exists()
