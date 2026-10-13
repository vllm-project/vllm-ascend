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
# DeepSeek-V3-style reasoning usage: backport reasoning-token counting.
#

from collections.abc import Sequence

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser


def _count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
    parser = self._parser
    if not isinstance(parser, BaseThinkingReasoningParser):
        return parser.count_reasoning_tokens(token_ids)

    if parser.start_token_id in token_ids:
        return parser.count_reasoning_tokens(token_ids)

    # Some templates put the opening marker in the prompt. A missing closing
    # marker therefore means generation was truncated while still reasoning.
    try:
        return token_ids.index(parser.end_token_id)
    except ValueError:
        return len(token_ids)


DeepSeekV3ReasoningParser.count_reasoning_tokens = _count_reasoning_tokens
