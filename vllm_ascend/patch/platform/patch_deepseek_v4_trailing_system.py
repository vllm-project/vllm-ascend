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

import inspect
from functools import wraps
from typing import Any

from vllm.tokenizers import deepseek_v4_encoding

_original_render_message = deepseek_v4_encoding.render_message
_render_message_signature = inspect.signature(_original_render_message)


def _needs_trailing_system_patch() -> bool:
    try:
        bound_args = _render_message_signature.bind(
            index=0,
            messages=[{"role": "system", "content": ""}],
            thinking_mode="chat",
        )
    except TypeError:
        return False
    bound_args.apply_defaults()
    prompt = _original_render_message(*bound_args.args, **bound_args.kwargs)
    return not prompt.endswith(deepseek_v4_encoding.ASSISTANT_SP_TOKEN + deepseek_v4_encoding.thinking_end_token)


@wraps(_original_render_message)
def _patched_render_message(*args: Any, **kwargs: Any) -> str:
    prompt = _original_render_message(*args, **kwargs)
    bound_args = _render_message_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    index = bound_args.arguments["index"]
    messages = bound_args.arguments["messages"]
    thinking_mode = bound_args.arguments["thinking_mode"]
    drop_thinking = bound_args.arguments.get("drop_thinking", True)
    message = messages[index]
    if message.get("role") != "system" or message.get("task") is not None:
        return prompt
    if index + 1 < len(messages) and messages[index + 1].get("role") != "assistant":
        return prompt

    prompt += deepseek_v4_encoding.ASSISTANT_SP_TOKEN
    last_user_idx = bound_args.arguments.get("last_user_idx")
    if last_user_idx is None:
        last_user_idx = deepseek_v4_encoding.find_last_user_index(messages)
    if thinking_mode == "thinking" and (not drop_thinking or index >= last_user_idx):
        prompt += deepseek_v4_encoding.thinking_start_token
    else:
        prompt += deepseek_v4_encoding.thinking_end_token
    return prompt


if _needs_trailing_system_patch():
    deepseek_v4_encoding.render_message = _patched_render_message
