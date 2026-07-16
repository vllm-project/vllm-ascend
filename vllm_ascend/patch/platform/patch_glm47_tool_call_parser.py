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


Glm47MoeModelToolParser._extract_tool_call_regions = _patched_extract_tool_call_regions
