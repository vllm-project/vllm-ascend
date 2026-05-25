import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts import utils


def test_wait_http_unready_returns_after_endpoint_stops(monkeypatch):
    states = iter([True, False])
    monkeypatch.setattr(utils, "_is_http_ready", lambda url: next(states))
    monkeypatch.setattr(utils.time, "sleep", lambda interval: None)

    utils.wait_http_unready("http://127.0.0.1:8000/health", timeout=10, interval=0)


def test_wait_http_unready_times_out_while_endpoint_stays_ready(monkeypatch):
    times = iter([0, 2])
    monkeypatch.setattr(utils.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(utils, "_is_http_ready", lambda url: True)

    with pytest.raises(TimeoutError, match="HTTP unready"):
        utils.wait_http_unready("http://127.0.0.1:8000/health", timeout=1, interval=0)
