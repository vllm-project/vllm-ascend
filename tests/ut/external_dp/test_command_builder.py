import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import EndpointResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_utils import CommandBuilder


def test_render_server_cmd_template(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-0.6B"]
    assert command.cmd[command.cmd.index("--port") + 1] == "7100"
    assert command.cmd[command.cmd.index("--data-parallel-rank") + 1] == "0"


def test_render_envs(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.env["LOCAL_ENDPOINT"] == "10.0.0.1:7100"
    assert command.env["ASCEND_RT_VISIBLE_DEVICES"] == "0"


def test_replace_generated_variables(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[1]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[command.cmd.index("--port") + 1] == "7101"
    assert command.cmd[command.cmd.index("--data-parallel-rank") + 1] == "1"


def test_missing_variable_error(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    bad_template = type(template)(
        envs=template.envs,
        server_cmd_template=[*template.server_cmd_template, "${UNKNOWN_VAR}"],
    )
    with pytest.raises(KeyError, match="UNKNOWN_VAR"):
        CommandBuilder(generic_config).build(endpoint, bad_template)


def test_final_command_auto_prefix_vllm_serve_model(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[:3] == ["vllm", "serve", generic_config.model]


def test_request_model_variable_rendering(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[command.cmd.index("--served-model-name") + 1] == generic_config.request_model
