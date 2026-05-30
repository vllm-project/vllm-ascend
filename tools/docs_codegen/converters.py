from __future__ import annotations

import json
import shlex
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tools.docs_codegen.errors import make_docs_codegen_error
from tools.docs_codegen.scanner import ModelCodeBlock
from tools.docs_codegen.utils import (
    ScalarValue,
    parse_command_tokens,
    render_cli_command,
    require_indexed_mapping,
    require_mapping,
    require_non_empty_string,
    require_scalar_mapping,
    substitute_template_positionals,
    trim_blank_edges,
)
from tools.docs_codegen.yaml_loader import LoadedYaml


@dataclass(frozen=True)
class GeneratedScript:
    """A converter output ready to be persisted as an artifact."""

    content: str
    language: str = "shell"


class BaseConverter(ABC):
    """Minimal contract shared by all converter plugins."""

    name: str

    @abstractmethod
    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        """Convert one loaded YAML document into one generated artifact."""


def build_default_converters() -> dict[str, BaseConverter]:
    converters: dict[str, BaseConverter] = {}
    for converter in (
        SingleNodeConverter(),
        MultiNodeConverter(),
        ExternalDpTemplateConverter(),
        ExternalDpLaunchConverter(),
        ExternalDpProxyConverter(),
    ):
        converters[converter.name] = converter
    return converters


def get_converter(
    converters: Mapping[str, BaseConverter],
    tag: str,
    *,
    block: ModelCodeBlock | None = None,
) -> BaseConverter:
    converter = converters.get(tag)
    if converter is None:
        supported = ", ".join(sorted(converters))
        raise make_docs_codegen_error(
            f"converter_tag '{tag}' is not registered; supported converters: {supported}",
            block=block,
            converter_tag=tag,
        )
    return converter


# ============================================================================
# Shell Rendering Helpers
# ============================================================================


def _join_shell_sections(*sections: Sequence[str]) -> str:
    rendered_lines: list[str] = []
    for section in sections:
        normalized = trim_blank_edges(section)
        if not normalized:
            continue
        if rendered_lines and rendered_lines[-1] != "":
            rendered_lines.append("")
        rendered_lines.extend(normalized)
    return "\n".join(rendered_lines).rstrip() + "\n"


def _render_env_export_lines(
    envs: Mapping[str, ScalarValue],
    *,
    defaults: Mapping[str, ScalarValue] | None = None,
) -> list[str]:
    exports: OrderedDict[str, ScalarValue] = OrderedDict((str(key), value) for key, value in envs.items())
    if defaults is not None:
        for key, value in defaults.items():
            exports[str(key)] = value
    return [f"export {name}={_format_export_value(value)}" for name, value in exports.items()]


def _build_shell_script(
    envs: Mapping[str, ScalarValue],
    command_tokens: Sequence[str],
    *,
    block: ModelCodeBlock,
    env_defaults: Mapping[str, ScalarValue] | None = None,
) -> GeneratedScript:
    content = _join_shell_sections(
        _render_env_export_lines(envs, defaults=env_defaults),
        format_vllm_serve_command(command_tokens, block=block),
    )
    return GeneratedScript(content=content)


def format_vllm_serve_command(tokens: Sequence[str], *, block: ModelCodeBlock) -> list[str]:
    if len(tokens) < 3 or tokens[0] != "vllm" or tokens[1] != "serve":
        raise make_docs_codegen_error(
            "generated command must start with 'vllm serve <model>'",
            block=block,
        )

    model = _format_command_token(tokens[2])
    command_lines = [f"vllm serve {model}"]
    option_lines: list[str] = []
    token_index = 3

    while token_index < len(tokens):
        token = tokens[token_index]
        if not token.startswith("-"):
            raise make_docs_codegen_error(
                f"generated command contains an unsupported positional argument '{token}'",
                block=block,
            )

        if token_index + 1 < len(tokens) and not tokens[token_index + 1].startswith("-"):
            value = tokens[token_index + 1]
            if token == "--kv-transfer-config":
                stripped = value.strip()
                if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(parsed, (dict, list)):
                            value = json.dumps(parsed, indent=4, ensure_ascii=False)

            option_lines.append(f"{token} {_format_command_token(value)}")
            token_index += 2
            continue

        option_lines.append(token)
        token_index += 1

    if not option_lines:
        return command_lines

    command_lines[0] = command_lines[0] + " \\"
    for index, line in enumerate(option_lines):
        suffix = " \\" if index < len(option_lines) - 1 else ""
        indented_line = line.replace("\n", "\n  ")
        command_lines.append(f"  {indented_line}{suffix}")

    return command_lines


def _format_export_value(value: ScalarValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)

    needs_quote = text != "" and (
        any(char.isspace() for char in text) or any(char in text for char in "'\"`;|&<>*?[]{}")
    )
    if not needs_quote:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _format_command_token(token: str) -> str:
    if not token:
        return '""'
    # Quote whitespace, embedded double quotes (JSON), and JSON-like containers
    # so e.g. a space-free '--profiler-config {"a":"b"}' value is not mangled by
    # the shell. Plain shell expansions like ``$SERVER_PORT`` / ``${NODE_0_IP}``
    # start with ``$`` and are intentionally left unquoted.
    needs_quote = (
        any(char.isspace() for char in token)
        or '"' in token
        or (token[:1] in "{[" and token[-1:] in "}]")
    )
    if needs_quote:
        return shlex.quote(token)
    return token


# ============================================================================
# Single Node Converter
# ============================================================================

SINGLE_NODE_DEFAULT_SERVER_PORT = "8000"
SINGLE_NODE_AUTO_SERVER_PORT = "DEFAULT_PORT"


def _resolve_single_node_server_port(envs: Mapping[str, ScalarValue]) -> ScalarValue:
    server_port = envs.get("SERVER_PORT")
    if server_port is None or server_port == SINGLE_NODE_AUTO_SERVER_PORT:
        return SINGLE_NODE_DEFAULT_SERVER_PORT
    return server_port


def _convert_single_node_case(
    loaded_yaml: LoadedYaml,
    *,
    block: ModelCodeBlock,
) -> GeneratedScript:
    test_case = require_indexed_mapping(
        loaded_yaml.yaml_root,
        collection_name="test_cases",
        option_name="case_index",
        field_name="test_case",
        block=block,
        default_index=0,
    )
    envs = require_scalar_mapping(test_case.get("envs"), field_name="envs", block=block)
    model = require_non_empty_string(test_case.get("model"), field_name="model", block=block)
    server_cmd = parse_command_tokens(test_case.get("server_cmd"), field_name="server_cmd", block=block)
    server_cmd_extra = []
    if test_case.get("server_cmd_extra") is not None:
        server_cmd_extra = parse_command_tokens(
            test_case.get("server_cmd_extra"),
            field_name="server_cmd_extra",
            block=block,
        )
    return _build_shell_script(
        envs,
        ["vllm", "serve", model, *server_cmd, *server_cmd_extra],
        block=block,
        env_defaults={"SERVER_PORT": _resolve_single_node_server_port(envs)},
    )


class SingleNodeConverter(BaseConverter):
    name = "single_node"

    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        return _convert_single_node_case(loaded_yaml, block=block)


# ============================================================================
# Multi Node Converter
# ============================================================================


def _convert_multi_node_host(
    loaded_yaml: LoadedYaml,
    *,
    block: ModelCodeBlock,
) -> GeneratedScript:
    deployment_item = require_indexed_mapping(
        loaded_yaml.yaml_root,
        collection_name="deployment",
        option_name="host_index",
        field_name="deployment_item",
        block=block,
    )
    envs = require_scalar_mapping(deployment_item.get("envs"), field_name="envs", block=block)
    server_cmd = parse_command_tokens(deployment_item.get("server_cmd"), field_name="server_cmd", block=block)
    return _build_shell_script(envs, server_cmd, block=block)


class MultiNodeConverter(BaseConverter):
    name = "multi_node"

    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        return _convert_multi_node_host(loaded_yaml, block=block)


# ============================================================================
# External DP Converters
#
# These read the external-DP YAML schema directly (``model`` / ``routing`` /
# ``config`` / ``templates``) used by
# tests/e2e/nightly/multi_node/external_dp/config/*.yaml. They are tightly
# coupled to that schema by design.
# ============================================================================

LAUNCH_ONLINE_DP_SCRIPT = "launch_online_dp.py"
PROXY_SCRIPT = "load_balance_proxy_server_example.py"
ROUTING_DISAGGREGATED_PREFILL = "disaggregated_prefill"

# Mirror tests/e2e/nightly/multi_node/external_dp/scripts/external_dp_config.py
# (proxy runs on node 0, port 1999); these are not part of the YAML.
EXTERNAL_DP_PROXY_NODE_INDEX = 0
EXTERNAL_DP_PROXY_PORT = 1999

# Maps external-DP ``${VAR}`` template variables to the positional shell
# parameters that ``launch_online_dp.py`` forwards to ``run_dp_template.sh``
# (``$1=visible_devices`` ... ``$7=tp_size``). Used so generated template
# snippets read like the hand-written ``run_dp_template.sh`` instead of leaking
# raw ``${DP_SIZE}`` placeholders.
RUN_DP_TEMPLATE_POSITIONALS: dict[str, str] = {
    "VISIBLE_DEVICES": "$1",
    "PORT": "$2",
    "DP_SIZE": "$3",
    "DP_RANK": "$4",
    "DP_ADDRESS": "$5",
    "DP_RPC_PORT": "$6",
    "TP_SIZE": "$7",
}

# Maps each launch_online_dp.py flag to the config[] field that feeds it.
LAUNCH_ARG_BY_FIELD: tuple[tuple[str, str], ...] = (
    ("dp_size", "--dp-size"),
    ("tp_size", "--tp-size"),
    ("dp_size_local", "--dp-size-local"),
    ("dp_rank_start", "--dp-rank-start"),
    ("dp_address", "--dp-address"),
    ("dp_rpc_port", "--dp-rpc-port"),
    ("port_start", "--vllm-start-port"),
)


def _node_ip_placeholder(node_index: int) -> str:
    return f"${{NODE_{node_index}_IP}}"


def _require_node_list(yaml_root: object, *, block: ModelCodeBlock) -> list[dict]:
    if not isinstance(yaml_root, dict):
        raise make_docs_codegen_error(
            f"YAML root must be a mapping, got {type(yaml_root).__name__}",
            block=block,
        )
    config = yaml_root.get("config")
    if not isinstance(config, list) or not config:
        raise make_docs_codegen_error("YAML field 'config' must be a non-empty list", block=block)
    return [require_mapping(node, field_name=f"config[{index}]", block=block) for index, node in enumerate(config)]


def _require_node_field(node: Mapping[str, object], field: str, *, node_index: int, block: ModelCodeBlock) -> object:
    if node.get(field) is None:
        raise make_docs_codegen_error(
            f"config[{node_index}] is missing required field '{field}'",
            block=block,
        )
    return node[field]


# ----------------------------------------------------------------------------
# Template converter (per node): env exports + ``vllm serve`` command.
# ----------------------------------------------------------------------------


def _convert_external_dp_template(loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
    template = require_indexed_mapping(
        loaded_yaml.yaml_root,
        collection_name="templates",
        option_name="host_index",
        field_name="template",
        block=block,
    )
    model = require_non_empty_string(loaded_yaml.yaml_root.get("model"), field_name="model", block=block)

    raw_envs = require_scalar_mapping(template.get("envs"), field_name="envs", block=block)
    envs = {
        key: (
            substitute_template_positionals(value, positionals=RUN_DP_TEMPLATE_POSITIONALS)
            if isinstance(value, str)
            else value
        )
        for key, value in raw_envs.items()
    }

    raw_server_cmd = parse_command_tokens(
        template.get("server_cmd_template"),
        field_name="server_cmd_template",
        block=block,
    )
    server_cmd = [
        substitute_template_positionals(token, positionals=RUN_DP_TEMPLATE_POSITIONALS) for token in raw_server_cmd
    ]

    return _build_shell_script(envs, ["vllm", "serve", model, *server_cmd], block=block)


class ExternalDpTemplateConverter(BaseConverter):
    name = "external_dp_template"

    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        return _convert_external_dp_template(loaded_yaml, block=block)


# ----------------------------------------------------------------------------
# Launch converter (whole cluster): one ``python launch_online_dp.py`` line per
# config node, single-line, separated by a blank line.
# ----------------------------------------------------------------------------


def _convert_external_dp_launch(loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
    nodes = _require_node_list(loaded_yaml.yaml_root, block=block)
    commands: list[str] = []
    for node_index, node in enumerate(nodes):
        options = [
            (flag, [str(_require_node_field(node, field, node_index=node_index, block=block))])
            for field, flag in LAUNCH_ARG_BY_FIELD
        ]
        commands.append(render_cli_command(["python", LAUNCH_ONLINE_DP_SCRIPT], options, multiline=False).rstrip())
    return GeneratedScript(content="\n\n".join(commands) + "\n")


class ExternalDpLaunchConverter(BaseConverter):
    name = "external_dp_launch"

    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        return _convert_external_dp_launch(loaded_yaml, block=block)


# ----------------------------------------------------------------------------
# Proxy converter (whole cluster): the load-balance proxy launch command.
# ----------------------------------------------------------------------------


def _expand_proxy_group(
    indices: object,
    nodes: list[dict],
    *,
    group_name: str,
    block: ModelCodeBlock,
) -> tuple[list[str], list[str]]:
    if not isinstance(indices, list) or not indices:
        raise make_docs_codegen_error(
            f"routing.groups.{group_name} must be a non-empty list",
            block=block,
        )
    hosts: list[str] = []
    ports: list[str] = []
    for raw_index in indices:
        node_index = int(raw_index)
        if node_index < 0 or node_index >= len(nodes):
            raise make_docs_codegen_error(
                f"routing.groups.{group_name} index {node_index} is out of range for 'config' "
                f"with {len(nodes)} items",
                block=block,
            )
        node = nodes[node_index]
        dp_size_local = int(_require_node_field(node, "dp_size_local", node_index=node_index, block=block))
        port_start = int(_require_node_field(node, "port_start", node_index=node_index, block=block))
        for local_rank in range(dp_size_local):
            hosts.append(_node_ip_placeholder(node_index))
            ports.append(str(port_start + local_rank))
    return hosts, ports


def _convert_external_dp_proxy(loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
    nodes = _require_node_list(loaded_yaml.yaml_root, block=block)
    routing = require_mapping(loaded_yaml.yaml_root.get("routing"), field_name="routing", block=block)

    routing_type = routing.get("type")
    if routing_type != ROUTING_DISAGGREGATED_PREFILL:
        raise make_docs_codegen_error(
            f"converter_tag 'external_dp_proxy' only supports routing.type "
            f"'{ROUTING_DISAGGREGATED_PREFILL}', got {routing_type!r}",
            block=block,
        )

    groups = require_mapping(routing.get("groups"), field_name="routing.groups", block=block)
    prefiller_hosts, prefiller_ports = _expand_proxy_group(
        groups.get("prefiller"), nodes, group_name="prefiller", block=block
    )
    decoder_hosts, decoder_ports = _expand_proxy_group(
        groups.get("decoder"), nodes, group_name="decoder", block=block
    )

    options = [
        ("--host", [_node_ip_placeholder(EXTERNAL_DP_PROXY_NODE_INDEX)]),
        ("--port", [str(EXTERNAL_DP_PROXY_PORT)]),
        ("--prefiller-hosts", prefiller_hosts),
        ("--prefiller-ports", prefiller_ports),
        ("--decoder-hosts", decoder_hosts),
        ("--decoder-ports", decoder_ports),
    ]
    content = render_cli_command(["python", PROXY_SCRIPT], options, multiline=True, expand_values=True)
    return GeneratedScript(content=content)


class ExternalDpProxyConverter(BaseConverter):
    name = "external_dp_proxy"

    def convert(self, loaded_yaml: LoadedYaml, *, block: ModelCodeBlock) -> GeneratedScript:
        return _convert_external_dp_proxy(loaded_yaml, block=block)
