# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register Kimi K3's upstream XTML structural tag on vLLM d02."""

from typing import Any

from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    _get_function_parameters,
    register_vllm_structural_tag,
)
from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
)

from vllm_ascend.patch.platform.kimi_k3_protocol import (
    ARGUMENT_END,
    CALL_END,
    CLOSE_TOKEN,
    MESSAGE_END,
    OPEN_TOKEN,
    RESPONSE_END,
    RESPONSE_START,
    SEP_TOKEN,
    TOOLS_END,
    TOOLS_START,
)

_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}
_STRING_ATOM = r"(?:[^<]|<[^|])"


def _escape_attr(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _bounded_string_regex(prop: dict[str, Any]) -> str | None:
    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if not isinstance(max_len, int) or max_len < 0 or max_len > 4096:
        return None
    if not isinstance(min_len, int) or min_len < 0 or min_len > max_len:
        min_len = 0
    return _STRING_ATOM + f"{{{min_len},{max_len}}}"


def _argument_tag(
    key: str,
    schema: dict[str, Any],
    root_defs: dict[str, Any] | None = None,
) -> TagFormat | None:
    prop = schema if isinstance(schema, dict) else {}
    json_type = prop.get("type")
    xtml_type = _JSON_TO_XTML_TYPE.get(json_type) if isinstance(json_type, str) else None
    if xtml_type is None:
        return None

    begin = f'{OPEN_TOKEN}argument key="{_escape_attr(key)}" type="{xtml_type}"{SEP_TOKEN}'
    if xtml_type == "string":
        enum_values = prop.get("enum")
        if enum_values is None and isinstance(prop.get("const"), str):
            enum_values = [prop["const"]]
        if (
            isinstance(enum_values, list)
            and enum_values
            and len(enum_values) <= 256
            and all(isinstance(value, str) for value in enum_values)
            and not any("<|" in value for value in enum_values)
        ):
            branches = [ConstStringFormat(value=value) for value in enum_values]
            content: Any = branches[0] if len(branches) == 1 else OrFormat(elements=branches)
        elif (bounded := _bounded_string_regex(prop)) is not None:
            content = RegexFormat(pattern=bounded)
        else:
            content = AnyTextFormat(excludes=[CLOSE_TOKEN])
    else:
        embedded = prop
        if root_defs:
            embedded = dict(prop)
            for defs_key, defs_value in root_defs.items():
                embedded.setdefault(defs_key, defs_value)
        content = JSONSchemaFormat(json_schema=embedded)
    return TagFormat(
        begin=begin,
        content=content,
        end=ARGUMENT_END,
    )


def _permissive_argument_tag() -> TagFormat:
    return TagFormat(
        begin=OPEN_TOKEN + "argument ",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^<]*" + SEP_TOKEN.replace("|", r"\|")),
                AnyTextFormat(excludes=[CLOSE_TOKEN]),
            ]
        ),
        end=ARGUMENT_END,
    )


def _arguments_block(parameters: dict[str, Any] | bool) -> Any:
    if not isinstance(parameters, dict):
        return StarFormat(content=_permissive_argument_tag())
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not properties:
        return StarFormat(content=_permissive_argument_tag())
    root_defs = {
        defs_key: parameters[defs_key]
        for defs_key in ("$defs", "definitions")
        if isinstance(parameters.get(defs_key), dict)
    }
    tags: list[TagFormat] = []
    for key, prop in properties.items():
        tag = _argument_tag(key, prop, root_defs)
        tags.append(tag if tag is not None else _permissive_argument_tag())
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=list(tags))
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _call_tag(tool: FunctionToolParam) -> TagFormat:
    function = tool.function
    parameters = _get_function_parameters(function)
    begin = f'{OPEN_TOKEN}call tool="{_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[0-9]+"),
                ConstStringFormat(value=f'"{SEP_TOKEN}'),
                _arguments_block(parameters),
            ]
        ),
        end=CALL_END,
    )


def _response_prefix() -> list[Any]:
    return [
        OptionalFormat(content=ConstStringFormat(value=RESPONSE_START)),
        TagFormat(
            begin="",
            content=AnyTextFormat(),
            end=RESPONSE_END,
        ),
    ]


def _tools_channel(tools: list[FunctionToolParam]) -> TagFormat:
    return TagFormat(
        begin=TOOLS_START,
        content=TagsWithSeparatorFormat(
            tags=[_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=TOOLS_END,
    )


@register_vllm_structural_tag("kimi_k3")
def get_kimi_k3_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    """Build the upstream K3 grammar with an optional message trailer."""

    del builtin_tools, reasoning
    trailer = OptionalFormat(content=ConstStringFormat(value=MESSAGE_END))

    if not tools:
        return StructuralTag(format=SequenceFormat(elements=[*_response_prefix(), trailer]))

    if tool_choice == "auto":
        tools_part: Any = OptionalFormat(content=_tools_channel(tools))
    elif tool_choice == "forced":
        tools_part = _tools_channel(tools[:1])
    else:
        tools_part = _tools_channel(tools)

    return StructuralTag(format=SequenceFormat(elements=[*_response_prefix(), tools_part, trailer]))
