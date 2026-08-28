#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from typing import Any

from vllm.tokenizers import deepseek_v4_encoding

_original_render_message = deepseek_v4_encoding.render_message


def _upstream_handles_trailing_system() -> bool:
    prompt = _original_render_message(
        0,
        [{"role": "system", "content": ""}],
        thinking_mode="chat",
    )
    return prompt.endswith(deepseek_v4_encoding.ASSISTANT_SP_TOKEN + deepseek_v4_encoding.thinking_end_token)


@wraps(_original_render_message)
def _patched_render_message(*args: Any, **kwargs: Any) -> str:
    prompt = _original_render_message(*args, **kwargs)
    index = kwargs["index"] if "index" in kwargs else args[0]
    messages = kwargs["messages"] if "messages" in kwargs else args[1]
    thinking_mode = kwargs["thinking_mode"] if "thinking_mode" in kwargs else args[2]
    drop_thinking = kwargs.get(
        "drop_thinking",
        args[3] if len(args) > 3 else True,
    )
    message = messages[index]
    if message.get("role") != "system" or message.get("task") is not None:
        return prompt
    if index + 1 < len(messages) and messages[index + 1].get("role") != "assistant":
        return prompt

    prompt += deepseek_v4_encoding.ASSISTANT_SP_TOKEN
    last_user_idx = kwargs.get(
        "last_user_idx",
        args[5] if len(args) > 5 else None,
    )
    if last_user_idx is None:
        last_user_idx = deepseek_v4_encoding.find_last_user_index(messages)
    if thinking_mode == "thinking" and (not drop_thinking or index >= last_user_idx):
        prompt += deepseek_v4_encoding.thinking_start_token
    else:
        prompt += deepseek_v4_encoding.thinking_end_token
    return prompt


if not _upstream_handles_trailing_system():
    deepseek_v4_encoding.render_message = _patched_render_message
