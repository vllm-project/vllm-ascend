import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import EndpointResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import build_proxy_command, proxy_health_url


def test_generic_dp_proxy_command(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    command = build_proxy_command(generic_config, endpoints)
    assert "examples/external_online_dp/dp_load_balance_proxy_server.py" in command
    assert "--dp-hosts" in command
    assert command[command.index("--dp-hosts") + 1 : command.index("--dp-ports")] == [
        "10.0.0.1",
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.2",
    ]
    assert command[command.index("--dp-ports") + 1 :] == ["7100", "7101", "7100", "7101"]


def test_pd_proxy_command(pd_config):
    endpoints = EndpointResolver(pd_config).resolve()
    command = build_proxy_command(pd_config, endpoints)
    assert "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" in command
    assert command[command.index("--prefiller-hosts") + 1 : command.index("--prefiller-ports")] == [
        "10.0.0.1",
        "10.0.0.1",
    ]
    assert command[command.index("--decoder-hosts") + 1 : command.index("--decoder-ports")] == [
        "10.0.0.2",
        "10.0.0.2",
    ]


def test_generic_dp_requires_worker_group(generic_config):
    with pytest.raises(ValueError, match="worker endpoints"):
        build_proxy_command(generic_config, [])


def test_pd_requires_prefiller_decoder_group(pd_config):
    endpoints = EndpointResolver(pd_config).resolve()
    with pytest.raises(ValueError, match="prefiller and decoder"):
        build_proxy_command(pd_config, endpoints[:2])


def test_proxy_health_url(generic_config):
    assert proxy_health_url(generic_config) == "http://10.0.0.1:1999/healthcheck"
