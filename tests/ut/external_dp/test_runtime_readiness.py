import sys
import types

import pytest

aisbench_stub = types.ModuleType("tools.aisbench")
aisbench_stub.run_aisbench_cases = lambda *args, **kwargs: []
sys.modules.setdefault("tools.aisbench", aisbench_stub)

from tests.e2e.nightly.multi_node.external_dp.scripts import test_external_dp  # noqa: E402
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import EndpointResolver  # noqa: E402


def test_wait_all_endpoints_ready_checks_every_rank_each_round(monkeypatch, pd_config):
    endpoints = EndpointResolver(pd_config).resolve()
    checked_urls = []

    def fake_ready(url, timeout):
        checked_urls.append(url)
        return len(checked_urls) > len(endpoints)

    monkeypatch.setattr(test_external_dp, "_is_http_ready", fake_ready)
    monkeypatch.setattr(test_external_dp.time, "sleep", lambda interval: None)

    test_external_dp._wait_all_endpoints_ready(endpoints, timeout=10)

    expected_urls = [test_external_dp._endpoint_health_url(endpoint) for endpoint in endpoints]
    assert checked_urls[: len(endpoints)] == expected_urls
    assert checked_urls[len(endpoints) : len(endpoints) * 2] == expected_urls


def test_wait_all_endpoints_ready_fails_if_ready_rank_becomes_unhealthy(monkeypatch, pd_config):
    endpoints = EndpointResolver(pd_config).resolve()[:2]
    first_url = test_external_dp._endpoint_health_url(endpoints[0])
    first_url_checks = 0

    def fake_ready(url, timeout):
        nonlocal first_url_checks
        if url == first_url:
            first_url_checks += 1
            return first_url_checks == 1
        return False

    monkeypatch.setattr(test_external_dp, "_is_http_ready", fake_ready)
    monkeypatch.setattr(test_external_dp.time, "sleep", lambda interval: None)

    with pytest.raises(RuntimeError, match="became unhealthy after ready"):
        test_external_dp._wait_all_endpoints_ready(endpoints, timeout=10)
