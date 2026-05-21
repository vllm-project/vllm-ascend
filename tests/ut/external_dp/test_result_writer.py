import json

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import EndpointResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_utils import (
    CommandBuilder,
    build_benchmark_results,
    write_benchmark_results_json,
)


def _commands_for(config, endpoints):
    builder = CommandBuilder(config)
    return [builder.build(endpoint, config.templates[endpoint.config_index]) for endpoint in endpoints]


def test_external_dp_topology_summary(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    result = build_benchmark_results(
        config=generic_config,
        endpoints=endpoints,
        commands=_commands_for(generic_config, endpoints),
        results=[["csv", {"Output Token Throughput": {"total": "10 token/s"}}]],
    )
    summary = result["external_dp_topology"]["summary"]
    assert summary["routing_type"] == "generic_dp"
    assert summary["proxy"] == {"node_index": 0, "host": "10.0.0.1", "port": 1999}


def test_external_dp_topology_endpoints(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    result = build_benchmark_results(
        config=generic_config,
        endpoints=endpoints,
        commands=_commands_for(generic_config, endpoints),
        results=[["csv", {"Output Token Throughput": {"total": "10 token/s"}}]],
    )
    topology_endpoints = result["external_dp_topology"]["endpoints"]
    assert len(topology_endpoints) == 4
    assert topology_endpoints[0]["dp_rank"] == 0
    assert topology_endpoints[3]["host"] == "10.0.0.2"


def test_keep_existing_result_fields(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    result = build_benchmark_results(
        config=generic_config,
        endpoints=endpoints,
        commands=_commands_for(generic_config, endpoints),
        results=[["csv", {"Output Token Throughput": {"total": "10 token/s"}}]],
    )
    for key in ("model_name", "hardware", "dtype", "feature", "vllm_version", "tasks", "serve_cmd", "environment"):
        assert key in result
    assert result["model_name"] == generic_config.model
    assert result["tasks"][0]["pass_fail"] == "pass"


def test_write_result_json(generic_config, tmp_path):
    endpoints = EndpointResolver(generic_config).resolve()
    output_path = write_benchmark_results_json(
        config=generic_config,
        endpoints=endpoints,
        commands=_commands_for(generic_config, endpoints),
        results=[["csv", {"Output Token Throughput": {"total": "10 token/s"}}]],
        output_dir=tmp_path,
    )
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["external_dp_topology"]["summary"]["num_nodes"] == 2
