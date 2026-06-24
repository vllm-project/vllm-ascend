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
# MiniMax-M2 usage accounting: backport reasoning-token counting.
#

from __future__ import annotations

from collections.abc import Sequence

from vllm.reasoning import minimax_m2_reasoning_parser as minimax_parser


def _count_minimax_reasoning_tokens(
    token_ids: Sequence[int],
    end_token_id: int | None,
) -> int:
    if end_token_id is None:
        return 0

    for idx, token_id in enumerate(token_ids):
        if token_id == end_token_id:
            return idx
    return len(token_ids)


def _patched_count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
    return _count_minimax_reasoning_tokens(token_ids, self.end_token_id)


minimax_parser.MiniMaxM2ReasoningParser.count_reasoning_tokens = _patched_count_reasoning_tokens
minimax_parser.MiniMaxM2AppendThinkReasoningParser.count_reasoning_tokens = _patched_count_reasoning_tokens
