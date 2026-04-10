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
    require_indexed_mapping,
    require_non_empty_string,
    require_scalar_mapping,
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
    for converter in (SingleNodeConverter(), MultiNodeConverter()):
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
    if any(char.isspace() for char in token):
        return shlex.quote(token)
    return token


# ============================================================================
# Single Node Converter
# ============================================================================


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
        env_defaults={"SERVER_PORT": "8000"},
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
