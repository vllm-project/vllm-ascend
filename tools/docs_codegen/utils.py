from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any, cast

import regex as re

from tools.docs_codegen.errors import make_docs_codegen_error

ScalarValue = str | int | float | bool | None

# Braced ``${VAR}`` template variables, mirroring runtime.py:TEMPLATE_VAR_RE.
TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def trim_blank_edges(lines: Sequence[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return list(lines[start:end])


def require_mapping(value: Any, *, field_name: str, block: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise make_docs_codegen_error(
            f"converter field '{field_name}' must be a mapping, got {type(value).__name__}",
            block=block,
        )
    return {str(key): item for key, item in value.items()}


def require_non_empty_string(value: Any, *, field_name: str, block: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise make_docs_codegen_error(
            f"converter field '{field_name}' must be a non-empty string",
            block=block,
        )
    return value.strip()


def require_block_index(
    *,
    block: Any,
    option_name: str,
    default: int | None = None,
) -> int:
    raw_index = block.get_option(option_name)
    if raw_index is None:
        if default is not None:
            return default
        raise make_docs_codegen_error(
            f"model-code block with converter_tag '{block.converter_tag}' requires {option_name}",
            block=block,
        )

    if not raw_index.isdecimal():
        raise make_docs_codegen_error(
            f"{option_name} must be a non-negative integer, got '{raw_index}'",
            block=block,
        )
    return int(raw_index)


def require_indexed_mapping(
    yaml_root: Any,
    *,
    collection_name: str,
    option_name: str,
    field_name: str,
    block: Any,
    default_index: int | None = None,
) -> dict[str, Any]:
    index = require_block_index(block=block, option_name=option_name, default=default_index)

    if not isinstance(yaml_root, dict):
        raise make_docs_codegen_error(
            f"YAML root must be a mapping, got {type(yaml_root).__name__}",
            block=block,
        )

    collection = yaml_root.get(collection_name)
    if not isinstance(collection, list):
        raise make_docs_codegen_error(
            f"YAML field '{collection_name}' must be a list",
            block=block,
        )

    if index >= len(collection):
        raise make_docs_codegen_error(
            f"{option_name} {index} is out of range for '{collection_name}' with {len(collection)} items",
            block=block,
        )

    return require_mapping(collection[index], field_name=field_name, block=block)


def require_scalar_mapping(
    value: Any,
    *,
    field_name: str,
    block: Any,
) -> dict[str, ScalarValue]:
    mapping = require_mapping(value, field_name=field_name, block=block)
    normalized: dict[str, ScalarValue] = {}
    for key, item in mapping.items():
        if isinstance(item, (dict, list)):
            raise make_docs_codegen_error(
                f"converter field '{field_name}.{key}' must be a scalar value",
                block=block,
            )
        normalized[str(key)] = cast(ScalarValue, item)
    return normalized


def parse_command_tokens(value: Any, *, field_name: str, block: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return shlex.split(value, posix=True)
        except ValueError as exc:
            raise make_docs_codegen_error(
                f"converter field '{field_name}' contains an invalid shell string: {exc}",
                block=block,
            ) from exc

    if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
        return [str(item) for item in value]

    raise make_docs_codegen_error(
        f"converter field '{field_name}' must be a shell string or a flat token list",
        block=block,
    )


def substitute_template_positionals(
    value: str,
    *,
    positionals: Mapping[str, str],
) -> str:
    """Replace braced ``${VAR}`` template variables with positional shell params.

    Only keys present in ``positionals`` are replaced; unknown braced variables
    and unbraced references like ``$SERVER_PORT`` are left untouched.
    """

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return positionals.get(key, match.group(0))

    return TEMPLATE_VAR_RE.sub(repl, value)


def render_cli_command(
    prefix: Sequence[str],
    options: Sequence[tuple[str, Sequence[str]]],
    *,
    multiline: bool,
    expand_values: bool = False,
) -> str:
    """Render a CLI command from a prefix and ``(flag, values)`` option groups.

    Supports multi-value flags (e.g. ``--prefiller-hosts h1 h2``). With
    ``multiline=False`` the whole command is rendered on one line. With
    ``multiline=True`` each option starts on its own backslash-continued line;
    when ``expand_values`` is also set, a multi-value flag is placed on its own
    line followed by each value on its own indented line (single-value flags
    stay inline). The returned string always ends with a newline.
    """
    prefix_str = " ".join(prefix)

    if not multiline:
        rendered = [" ".join([flag, *[str(value) for value in values]]) for flag, values in options]
        return " ".join([prefix_str, *rendered]).rstrip() + "\n"

    # Each entry is a logical line rendered without its trailing backslash.
    entries: list[str] = [prefix_str]
    for flag, values in options:
        str_values = [str(value) for value in values]
        if expand_values and len(str_values) > 1:
            entries.append(f"  {flag}")
            entries.extend(f"    {value}" for value in str_values)
        else:
            entries.append(f"  {' '.join([flag, *str_values])}")

    lines = [entry + (" \\" if index < len(entries) - 1 else "") for index, entry in enumerate(entries)]
    return "\n".join(lines) + "\n"
