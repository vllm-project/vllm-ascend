# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Structural-tag grammar for DeepSeek V4.1 DSML tool calls."""

from vllm.tool_parsers.structural_tag_registry import (
    register_vllm_structural_tag,
)
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

_V41_CALLS_START = "<｜DSML｜ calls>"
_V41_CALLS_END = "</｜DSML｜ calls>"
_V41_INVOKE_END = "</｜DSML｜ invoke>"
_V41_PARAMETER_END = "</｜DSML｜ parameter>"


@register_vllm_structural_tag("deepseek_v41")
def get_deepseek_v41_structural_tag(
    tools,
    builtin_tools,
    tool_choice,
    reasoning,
    token_suffix="",
):
    del builtin_tools, reasoning, token_suffix
    parameter = TagFormat(
        begin='<｜DSML｜ parameter name="',
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^\"]+"),
                ConstStringFormat(value='" string="'),
                OrFormat(
                    elements=[
                        SequenceFormat(
                            elements=[
                                ConstStringFormat(value='true">'),
                                AnyTextFormat(
                                    excludes=[
                                        _V41_PARAMETER_END,
                                        _V41_INVOKE_END,
                                        _V41_CALLS_END,
                                    ]
                                ),
                            ]
                        ),
                        SequenceFormat(
                            elements=[
                                ConstStringFormat(value='false">'),
                                JSONSchemaFormat(json_schema=True),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        end=f"{_V41_PARAMETER_END}\n",
    )
    calls = TagsWithSeparatorFormat(
        tags=[
            TagFormat(
                begin=f'<｜DSML｜ invoke name="{tool.function.name}">\n',
                content=StarFormat(content=parameter),
                end=f"{_V41_INVOKE_END}\n",
            )
            for tool in tools
        ],
        separator="",
        at_least_one=True,
        stop_after_first=tool_choice == "forced",
    )
    if tool_choice == "auto":
        if tools:
            format_ = TriggeredTagsFormat(
                triggers=[_V41_CALLS_START],
                tags=[
                    TagFormat(
                        begin=f"{_V41_CALLS_START}\n",
                        content=calls,
                        end=_V41_CALLS_END,
                    )
                ],
                excludes=["<think>", "</think>"],
            )
        else:
            format_ = AnyTextFormat(excludes=["<think>", "</think>"])
        return StructuralTag(format=format_)
    return StructuralTag(
        format=SequenceFormat(
            elements=[
                ConstStringFormat(value=f"\n\n{_V41_CALLS_START}\n"),
                calls,
                ConstStringFormat(value=_V41_CALLS_END),
            ]
        )
    )
