#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# GLM-4.7 tool-call streaming parser compatibility patch.
#

from __future__ import annotations

from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.utils import (
    extract_types_from_schema,
    find_tool_properties,
    partial_tag_overlap,
)

if not hasattr(Glm47MoeModelToolParser, "_ascend_original_extract_tool_call_regions"):
    Glm47MoeModelToolParser._ascend_original_extract_tool_call_regions = (
        Glm47MoeModelToolParser._extract_tool_call_regions
    )


def _patched_extract_tool_call_regions(
    self: Glm47MoeModelToolParser,
    text: str,
) -> list[tuple[str, bool]]:
    original_extract_tool_call_regions = self._ascend_original_extract_tool_call_regions
    regions = original_extract_tool_call_regions(text)
    normalized_regions: list[tuple[str, bool]] = []

    for inner_text, is_complete in regions:
        if is_complete and self.arg_key_start not in inner_text and "\n" not in inner_text:
            tool_name = inner_text.strip()
            inner_text = f"{tool_name}\n" if tool_name else inner_text
        normalized_regions.append((inner_text, is_complete))

    return normalized_regions


def _patched_is_string_type(
    self: Glm47MoeModelToolParser,
    tool_name: str,
    arg_name: str,
) -> bool:
    properties = find_tool_properties(self.tools, tool_name)
    if not isinstance(properties, dict):
        return True
    return "string" in extract_types_from_schema(properties.get(arg_name))


def _patched_build_args_json_so_far(
    self: Glm47MoeModelToolParser,
    tool_name: str,
    inner_text: str,
    is_complete: bool,
) -> str:
    args_so_far = self._ascend_original_build_args_json_so_far(tool_name, inner_text, is_complete)
    if is_complete:
        return args_so_far

    last_val_start = inner_text.rfind(self.arg_val_start)
    last_val_end = inner_text.rfind(self.arg_val_end)
    if last_val_start == -1 or last_val_end > last_val_start:
        return args_so_far

    last_key_match = None
    for match in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
        last_key_match = match
    if last_key_match is None:
        return args_so_far

    partial_key = last_key_match.group(1).strip()
    if self._is_string_type(tool_name, partial_key):
        return args_so_far

    partial_content = inner_text[last_val_start + len(self.arg_val_start) :]
    overlap = partial_tag_overlap(partial_content, self.arg_val_end)
    if overlap:
        partial_content = partial_content[:-overlap]
    if partial_content and args_so_far.endswith(partial_content):
        return args_so_far[: -len(partial_content)]
    return args_so_far


if not hasattr(Glm47MoeModelToolParser, "_ascend_original_build_args_json_so_far"):
    Glm47MoeModelToolParser._ascend_original_build_args_json_so_far = Glm47MoeModelToolParser._build_args_json_so_far


Glm47MoeModelToolParser._extract_tool_call_regions = _patched_extract_tool_call_regions
Glm47MoeModelToolParser._is_string_type = _patched_is_string_type
Glm47MoeModelToolParser._build_args_json_so_far = _patched_build_args_json_so_far
