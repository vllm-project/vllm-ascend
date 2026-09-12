# SPDX-License-Identifier: Apache-2.0
"""V4.1 tool constraints: raw string parameters and JSON for other values."""

import copy
import json

from vllm import envs
from vllm.tool_parsers.structural_tag_registry import (
    _any_tool_strict,
    _dump_tool_choice_for_xgrammar,
    _dump_tool_for_xgrammar,
)
from xgrammar import Grammar, StructuralTag, normalize_tool_choice
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    GrammarFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

PARAMETER_END = "</｜DSML｜ parameter>"


def _parameter_formats(name, schema, definitions):
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/") or ref[8:] not in definitions:
            raise ValueError(f"Unsupported V4.1 tool parameter reference: {ref}")
        schema = {**definitions[ref[8:]], **{key: value for key, value in schema.items() if key != "$ref"}}
    if "anyOf" in schema:
        return [item for branch in schema["anyOf"] for item in _parameter_formats(name, branch, definitions)]
    types = schema.get("type", ["string", "number", "boolean", "null", "array", "object"])
    if isinstance(types, str):
        types = [types]
    formats = []
    for value_type in types:
        value_schema = {**schema, "type": value_type, "$defs": definitions}
        if value_type == "string":
            # Use xgrammar's raw XML string constraints (enum/pattern/length),
            # changing only its terminator, never user-provided schema literals.
            grammar = str(
                Grammar.from_structural_tag(
                    StructuralTag(format=JSONSchemaFormat(json_schema=value_schema, style="deepseek_xml"))
                )
            )
            old_end = json.dumps("</｜DSML｜parameter>")
            grammar = grammar.replace(f"excludes=({old_end})", f"excludes=({json.dumps(PARAMETER_END)})")
            content = GrammarFormat(grammar=grammar)
        else:
            content = JSONSchemaFormat(json_schema=value_schema)
        formats.append(
            TagFormat(
                begin=f'<｜DSML｜ parameter name="{name}" string="{str(value_type == "string").lower()}">',
                content=content,
                end=PARAMETER_END + "\n",
            )
        )
    return formats


def _tool_tag(tool):
    function = tool.function
    schema = copy.deepcopy(function.parameters or {"type": "object", "properties": {}})
    # Cross-property constraints cannot be enforced by independent parameter
    # grammars. Reject them rather than silently weakening strict tool schemas.
    supported = {"type", "properties", "required", "additionalProperties", "$defs", "title", "description", "$schema"}
    if set(schema) - supported or schema.get("type", "object") != "object":
        raise ValueError("V4.1 strict tools require an object schema with properties and required fields")
    if schema.get("additionalProperties") not in (None, False):
        raise ValueError("V4.1 strict tools require named properties (additionalProperties=false)")
    parameters = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not set(required).issubset(properties):
        raise ValueError("V4.1 required parameters must be declared in properties")
    for name, value_schema in properties.items():
        if '"' in name:
            raise ValueError("V4.1 DSML parameter names cannot contain quotes")
        alternatives = _parameter_formats(name, value_schema, schema.get("$defs", {}))
        parameter = alternatives[0] if len(alternatives) == 1 else OrFormat(elements=alternatives)
        if name not in required:
            parameter = OptionalFormat(content=parameter)
        parameters.append(parameter)
    return TagFormat(
        begin=f'<｜DSML｜ invoke name="{function.name}">\n',
        content=SequenceFormat(elements=parameters) if parameters else ConstStringFormat(value="\n"),
        end="</｜DSML｜ invoke>\n",
    )


def get_structural_tag(request, *, reasoning):
    if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING or not request.tools or request.tool_choice == "none":
        return None
    if request.tool_choice == "auto" and not _any_tool_strict(request.tools):
        return None
    tools, builtin_tools, choice = normalize_tool_choice(
        [_dump_tool_for_xgrammar(tool) for tool in request.tools],
        _dump_tool_choice_for_xgrammar(request.tool_choice),
    )
    if builtin_tools:
        raise ValueError("V4.1 supports function tools only")
    tags = [_tool_tag(tool) for tool in tools]
    calls = TagFormat(
        begin="<｜DSML｜ calls>\n",
        content=tags[0] if choice == "forced" else TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
        end="</｜DSML｜ calls>",
    )
    if choice == "auto":
        suffix = TriggeredTagsFormat(triggers=["<｜DSML｜ calls>"], tags=[calls], excludes=["<think>", "</think>"])
    else:
        suffix = SequenceFormat(elements=[ConstStringFormat(value="\n\n"), calls])
    if reasoning:
        suffix = SequenceFormat(elements=[TagFormat(begin="", content=AnyTextFormat(), end="</think>"), suffix])
    return StructuralTag(format=suffix)
