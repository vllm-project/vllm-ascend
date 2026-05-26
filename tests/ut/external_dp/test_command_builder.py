import json
from pathlib import Path

import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts import utils as external_dp_utils
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    EndpointResolver,
    ExternalDPConfigLoader,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    CommandBuilder,
    build_distributed_envs,
)


def test_render_server_cmd_template(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-0.6B"]
    assert command.cmd[command.cmd.index("--host") + 1] == "0.0.0.0"
    assert command.cmd[command.cmd.index("--port") + 1] == "7100"
    assert command.cmd[command.cmd.index("--data-parallel-rank") + 1] == "0"


def test_render_envs(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.env["SERVER_PORT"] == "7100"
    assert command.env["LOCAL_ENDPOINT"] == "10.0.0.1:7100"
    assert command.env["ASCEND_RT_VISIBLE_DEVICES"] == "0"


def test_build_distributed_envs(monkeypatch):
    monkeypatch.setattr(external_dp_utils, "get_net_interface", lambda ip: "eth0")
    envs = build_distributed_envs("10.0.0.1", "10.0.0.1")

    assert envs == {
        "HCCL_IF_IP": "10.0.0.1",
        "HCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "TP_SOCKET_IFNAME": "eth0",
        "LOCAL_IP": "10.0.0.1",
        "NIC_NAME": "eth0",
        "MASTER_IP": "10.0.0.1",
    }


def test_startup_template_can_include_distributed_envs(monkeypatch, generic_config):
    monkeypatch.setattr(external_dp_utils, "get_net_interface", lambda ip: "eth0")
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    startup_template = type(template)(
        envs={
            **template.envs,
            **build_distributed_envs("10.0.0.1", "10.0.0.1"),
        },
        server_cmd_template=template.server_cmd_template,
    )
    command = CommandBuilder(generic_config).build(endpoint, startup_template)

    assert command.env["LOCAL_IP"] == "10.0.0.1"
    assert command.env["MASTER_IP"] == "10.0.0.1"
    assert command.env["NIC_NAME"] == "eth0"
    assert command.env["HCCL_IF_IP"] == "10.0.0.1"
    assert command.env["GLOO_SOCKET_IFNAME"] == "eth0"
    assert command.env["HCCL_SOCKET_IFNAME"] == "eth0"
    assert command.env["TP_SOCKET_IFNAME"] == "eth0"


def test_distributed_envs_override_template_network_envs(monkeypatch, generic_config):
    monkeypatch.setattr(external_dp_utils, "get_net_interface", lambda ip: "eth0")
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    startup_template = type(template)(
        envs={
            **template.envs,
            "GLOO_SOCKET_IFNAME": "ib0",
            **build_distributed_envs("10.0.0.1", "10.0.0.1"),
        },
        server_cmd_template=template.server_cmd_template,
    )
    command = CommandBuilder(generic_config).build(endpoint, startup_template)

    assert command.env["GLOO_SOCKET_IFNAME"] == "eth0"


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


def test_host_is_not_a_config_template_variable(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    bad_template = type(template)(
        envs=template.envs,
        server_cmd_template=[*template.server_cmd_template, "${HOST}"],
    )
    with pytest.raises(KeyError, match="HOST"):
        CommandBuilder(generic_config).build(endpoint, bad_template)


def test_command_unbraced_variables_must_reference_env(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    bad_template = type(template)(
        envs=template.envs,
        server_cmd_template=[*template.server_cmd_template, "$PORT"],
    )
    with pytest.raises(KeyError, match="PORT"):
        CommandBuilder(generic_config).build(endpoint, bad_template)


def test_final_command_auto_prefix_vllm_serve_model(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[:3] == ["vllm", "serve", generic_config.model]


def test_command_can_render_from_env(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    command = CommandBuilder(generic_config).build(endpoint, generic_config.templates[0])
    assert command.cmd[command.cmd.index("--port") + 1] == command.env["SERVER_PORT"]


def test_cp_sp_template_variables_default_to_one(generic_config):
    endpoint = EndpointResolver(generic_config).resolve()[0]
    template = generic_config.templates[0]
    extended_template = type(template)(
        envs=template.envs,
        server_cmd_template=[*template.server_cmd_template, "${CP_SIZE}", "${SP_SIZE}"],
    )
    command = CommandBuilder(generic_config).build(endpoint, extended_template)
    assert command.cmd[-2:] == ["1", "1"]


def test_generic_smoke_uses_qwen3_moe_command():
    config_path = Path("tests/e2e/nightly/multi_node/external_dp/config/generic_dp_smoke.yaml")
    config = ExternalDPConfigLoader.from_yaml(
        str(config_path),
        cluster_ips=["10.0.0.1", "10.0.0.2"],
    )
    endpoint = EndpointResolver(config).resolve()[0]
    command = CommandBuilder(config).build(endpoint, config.templates[0])

    assert command.cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-30B-A3B"]
    assert command.cmd[command.cmd.index("--max-model-len") + 1] == "4096"
    assert command.cmd[command.cmd.index("--data-parallel-size") + 1] == "4"
    assert command.cmd[command.cmd.index("--data-parallel-rank") + 1] == "0"
    assert "--quantization" not in command.cmd
    assert "--enable-expert-parallel" in command.cmd
    assert command.env["HCCL_BUFFSIZE"] == "1024"
    assert command.env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"
    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in command.env


def test_disaggregated_smoke_kv_transfer_config_uses_multiline_json():
    config_path = Path("tests/e2e/nightly/multi_node/external_dp/config/disaggregated_prefill_smoke.yaml")
    config = ExternalDPConfigLoader.from_yaml(
        str(config_path),
        cluster_ips=["10.0.0.1", "10.0.0.2"],
    )
    endpoints = EndpointResolver(config).resolve()
    producer_command = CommandBuilder(config).build(endpoints[0], config.templates[0])
    consumer_command = CommandBuilder(config).build(endpoints[-1], config.templates[1])

    assert '"kv_connector": "MooncakeConnectorV1",\n' in config_path.read_text(encoding="utf-8")

    for command, kv_role in ((producer_command, "kv_producer"), (consumer_command, "kv_consumer")):
        raw_config = command.cmd[command.cmd.index("--kv-transfer-config") + 1]
        kv_config = json.loads(raw_config)
        assert kv_config["kv_role"] == kv_role
        assert kv_config["kv_connector_extra_config"] == {
            "prefill": {"dp_size": 2, "tp_size": 1},
            "decode": {"dp_size": 2, "tp_size": 1},
        }


def test_deepseek_pd_kv_transfer_config_uses_multiline_json():
    config_path = Path("tests/e2e/nightly/multi_node/external_dp/config/DeepSeek-V3.1-external-dp-pd.yaml")
    config = ExternalDPConfigLoader.from_yaml(
        str(config_path),
        cluster_ips=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
    )
    endpoints = EndpointResolver(config).resolve()
    endpoint_by_config_index = {endpoint.config_index: endpoint for endpoint in endpoints}
    expected = {
        0: ("kv_producer", "30000", "0"),
        1: ("kv_producer", "30100", "1"),
        2: ("kv_consumer", "30200", "2"),
    }

    assert '"kv_connector": "MooncakeConnectorV1",\n' in config_path.read_text(encoding="utf-8")

    for config_index, (kv_role, kv_port, engine_id) in expected.items():
        endpoint = endpoint_by_config_index[config_index]
        command = CommandBuilder(config).build(endpoint, config.templates[config_index])
        raw_config = command.cmd[command.cmd.index("--kv-transfer-config") + 1]
        kv_config = json.loads(raw_config)
        assert kv_config["kv_role"] == kv_role
        assert kv_config["kv_port"] == kv_port
        assert kv_config["engine_id"] == engine_id
        assert kv_config["kv_connector_extra_config"] == {
            "prefill": {"dp_size": 2, "tp_size": 8},
            "decode": {"dp_size": 32, "tp_size": 1},
        }
