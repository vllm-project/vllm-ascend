import json

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import EndpointResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    CommandBuilder,
    build_benchmark_results,
    write_benchmark_results_json,
)


def _commands_for(config, endpoints):
    builder = CommandBuilder(config)
    return [builder.build(endpoint, config.templates[endpoint.config_index]) for endpoint in endpoints]


def _with_env(command, **env):
    return type(command)(
        cmd=command.cmd,
        env={**command.env, **env},
        display_cmd=command.display_cmd,
    )


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
    assert result["environment"] == {"VLLM_USE_MODELSCOPE": "true"}
    assert "external_dp_topology" not in result


def test_result_features_come_from_actual_command_env(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    commands = _commands_for(generic_config, endpoints)
    commands[-1] = _with_env(commands[-1], VLLM_ASCEND_ENABLE_FLASHCOMM1="1")
    result = build_benchmark_results(
        config=generic_config,
        endpoints=endpoints,
        commands=commands,
        results=[["csv", {"Output Token Throughput": {"total": "10 token/s"}}]],
    )
    assert "flashcomm1" in result["feature"]


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
    assert "external_dp_topology" not in data
