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

import json
from functools import wraps
from typing import Any

from vllm.parser.deepseek_v4 import (
    _PARAM_RE,
    _PARTIAL_PARAM_RE,
    DeepSeekV4Parser,
    deepseek_v4_config,
)

_DIRECT_STRING_SLOTS_ATTR = "_ascend_direct_string_slots"
_PATCHED_ATTR = "_ascend_tool_streaming_patched"
_UPSTREAM_STRUCTURAL_CHARS = deepseek_v4_config().arg_structural_chars

_original_reset = DeepSeekV4Parser._reset
_original_compute_arg_delta = DeepSeekV4Parser._compute_arg_delta


@wraps(_original_reset)
def _patched_reset(
    self: DeepSeekV4Parser,
    *args: Any,
    **kwargs: Any,
) -> None:
    _original_reset(self, *args, **kwargs)
    setattr(self, _DIRECT_STRING_SLOTS_ATTR, set())


def _in_progress_string_key(raw_args: str) -> str | None:
    last_complete_end = 0
    for match in _PARAM_RE.finditer(raw_args):
        last_complete_end = match.end()

    match = _PARTIAL_PARAM_RE.search(raw_args, last_complete_end)
    if match is None or match.group(2) != "true":
        return None
    return match.group(1)


def _ends_in_open_json_string(json_prefix: str) -> bool:
    in_string = False
    escaped = False
    for char in json_prefix:
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
    return in_string


@wraps(_original_compute_arg_delta)
def _patched_compute_arg_delta(
    self: DeepSeekV4Parser,
    idx: int,
    raw_delta: str,
) -> str | None:
    direct_string_slots: set[int] = getattr(
        self,
        _DIRECT_STRING_SLOTS_ATTR,
        set(),
    )
    setattr(self, _DIRECT_STRING_SLOTS_ATTR, direct_string_slots)

    structural = self._arg_structural_chars
    is_plain_delta = structural is not None and structural.isdisjoint(raw_delta)
    # The lexer holds split closing tags, so a plain event here is committed
    # string body text and can be JSON-escaped without rescanning slot.args.
    if idx in direct_string_slots and is_plain_delta and self._arg_converter is not None and self._stream_arg_deltas:
        escaped_delta = json.dumps(raw_delta, ensure_ascii=False)[1:-1]
        if escaped_delta:
            self._tool_slots[idx].streamed_json += escaped_delta
            return escaped_delta
        return None

    direct_string_slots.discard(idx)
    arg_delta = _original_compute_arg_delta(self, idx, raw_delta)
    if not arg_delta:
        return arg_delta

    # Structural events always return to the converter. Re-arm direct streaming
    # only while the raw DSML still ends in a schema-stable string parameter.
    slot = self._tool_slots[idx]
    string_key = _in_progress_string_key(slot.args)
    if (
        string_key is not None
        and (slot.string_keys is None or string_key in slot.string_keys)
        and _ends_in_open_json_string(slot.streamed_json)
    ):
        direct_string_slots.add(idx)
    return arg_delta


if frozenset(">") == _UPSTREAM_STRUCTURAL_CHARS:
    if not getattr(DeepSeekV4Parser, _PATCHED_ATTR, False):
        DeepSeekV4Parser._reset = _patched_reset
        DeepSeekV4Parser._compute_arg_delta = _patched_compute_arg_delta
        setattr(DeepSeekV4Parser, _PATCHED_ATTR, True)
elif _UPSTREAM_STRUCTURAL_CHARS is not None:
    raise RuntimeError(
        "DeepSeek V4 tool streaming patch requires upstream "
        "arg_structural_chars to be frozenset('>') or None, got "
        f"{_UPSTREAM_STRUCTURAL_CHARS!r}"
    )
