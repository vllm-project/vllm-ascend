#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

"""Strict Kimi K3 XTML parsing shared by streaming and full responses."""

from __future__ import annotations

import codecs
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import regex as re

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

THINK_START = f"{OPEN_TOKEN}think{SEP_TOKEN}"
THINK_END = f"{CLOSE_TOKEN}think{SEP_TOKEN}"
RESPONSE_START = f"{OPEN_TOKEN}response{SEP_TOKEN}"
RESPONSE_END = f"{CLOSE_TOKEN}response{SEP_TOKEN}"
TOOLS_START = f"{OPEN_TOKEN}tools{SEP_TOKEN}"
TOOLS_END = f"{CLOSE_TOKEN}tools{SEP_TOKEN}"
CALL_END = f"{CLOSE_TOKEN}call{SEP_TOKEN}"
ARGUMENT_END = f"{CLOSE_TOKEN}argument{SEP_TOKEN}"
JSON_END = f"{CLOSE_TOKEN}json{SEP_TOKEN}"
MESSAGE_END = f"{CLOSE_TOKEN}message{SEP_TOKEN}"

# Unpaired surrogates cannot be produced by valid UTF-8 model text, so they
# preserve structural-token provenance without colliding with ordinary text.
_OPEN_TOKEN = "\ud800"
_CLOSE_TOKEN = "\ud801"
_SEP_TOKEN = "\ud802"
_END_OF_MSG_TOKEN = "\ud803"

_THINK_START = f"{_OPEN_TOKEN}think{_SEP_TOKEN}"
_THINK_END = f"{_CLOSE_TOKEN}think{_SEP_TOKEN}"
_RESPONSE_START = f"{_OPEN_TOKEN}response{_SEP_TOKEN}"
_RESPONSE_END = f"{_CLOSE_TOKEN}response{_SEP_TOKEN}"
_TOOLS_START = f"{_OPEN_TOKEN}tools{_SEP_TOKEN}"
_TOOLS_END = f"{_CLOSE_TOKEN}tools{_SEP_TOKEN}"
_CALL_END = f"{_CLOSE_TOKEN}call{_SEP_TOKEN}"
_ARGUMENT_END = f"{_CLOSE_TOKEN}argument{_SEP_TOKEN}"
_JSON_END = f"{_CLOSE_TOKEN}json{_SEP_TOKEN}"
_MESSAGE_END = f"{_CLOSE_TOKEN}message{_SEP_TOKEN}"

_CONTROL_TOKEN_TEXT = {
    OPEN_TOKEN: _OPEN_TOKEN,
    CLOSE_TOKEN: _CLOSE_TOKEN,
    SEP_TOKEN: _SEP_TOKEN,
    END_OF_MSG_TOKEN: _END_OF_MSG_TOKEN,
}
_PROTOCOL_TOKEN_TEXT = {value: key for key, value in _CONTROL_TOKEN_TEXT.items()}
_CONTROL_SENTINELS = tuple(_PROTOCOL_TOKEN_TEXT)

ToolMode = Literal["none", "auto", "required", "named"]

_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*)="([^"]*)"')
_UNKNOWN_ENTITY_RE = re.compile(r"&(?!amp;|quot;)")
_CALL_START_RE = re.compile(
    re.escape(f"{_OPEN_TOKEN}call") + r"(?P<attrs>.*?)" + re.escape(_SEP_TOKEN),
    re.DOTALL,
)
_ARGUMENT_START_RE = re.compile(
    re.escape(f"{_OPEN_TOKEN}argument") + r"(?P<attrs>.*?)" + re.escape(_SEP_TOKEN),
    re.DOTALL,
)
_JSON_START_RE = re.compile(
    re.escape(f"{_OPEN_TOKEN}json") + r"(?P<attrs>.*?)" + re.escape(_SEP_TOKEN),
    re.DOTALL,
)

_MAX_PENDING_TAG_CHARS = 8192


def _has_control_sentinel(text: str) -> bool:
    return any(sentinel in text for sentinel in _CONTROL_SENTINELS)


def _reject_unexpected_control_tokens(text: str) -> None:
    if _has_control_sentinel(text):
        raise KimiK3XTMLParseError("Unexpected K3 control token inside text content.")


class _KimiK3TokenDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._control_ids = {
            int(tokenizer.convert_tokens_to_ids(marker)): protocol_marker
            for marker, protocol_marker in _CONTROL_TOKEN_TEXT.items()
        }
        if len(self._control_ids) != len(_CONTROL_TOKEN_TEXT):
            raise RuntimeError("Kimi K3 control tokens must have distinct token IDs.")

        model = getattr(tokenizer, "model", None)
        self._decode_single_token_bytes = getattr(model, "decode_single_token_bytes", None)
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _reset_utf8_decoder(self) -> None:
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, token_ids: Sequence[int]) -> str:
        parts: list[str] = []
        for token_id in token_ids:
            protocol_marker = self._control_ids.get(int(token_id))
            if protocol_marker is not None:
                pending = self._utf8_decoder.decode(b"", final=True)
                if pending:
                    parts.append(pending)
                self._reset_utf8_decoder()
                parts.append(protocol_marker)
                continue

            if self._decode_single_token_bytes is not None:
                raw = self._decode_single_token_bytes(int(token_id))
                decoded = self._utf8_decoder.decode(raw, final=False)
            else:
                decoded = self._tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            if decoded:
                parts.append(decoded)
        return "".join(parts)

    def finish(self) -> str:
        pending = self._utf8_decoder.decode(b"", final=True)
        self._reset_utf8_decoder()
        return pending


class KimiK3XTMLParseError(RuntimeError):
    """Raised when a completed K3 response violates the XTML contract."""


@dataclass(frozen=True)
class KimiK3ParsedCall:
    name: str
    index: int
    arguments: str


@dataclass(frozen=True)
class KimiK3ParseSnapshot:
    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[KimiK3ParsedCall, ...] = ()


@dataclass(frozen=True)
class KimiK3ToolCallDelta:
    index: int
    name: str | None = None
    arguments: str | None = None


def partial_marker_overlap(text: str, marker: str) -> int:
    """Return the suffix length that can still grow into *marker*."""

    max_overlap = min(len(text), len(marker) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(marker[:overlap]):
            return overlap
    return 0


def _decode_attr_value(value: str) -> str:
    if _UNKNOWN_ENTITY_RE.search(value):
        raise KimiK3XTMLParseError(f"Unsupported XTML attribute entity in {value!r}.")
    # Decode in this order so ``&amp;quot;`` remains literal ``&quot;``.
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    cursor = 0
    for match in _ATTR_RE.finditer(raw_attrs):
        if raw_attrs[cursor : match.start()].strip():
            raise KimiK3XTMLParseError(f"Malformed XTML attributes: {raw_attrs!r}.")
        key = match.group(1)
        if key in attrs:
            raise KimiK3XTMLParseError(f"Duplicate XTML attribute {key!r}.")
        attrs[key] = _decode_attr_value(match.group(2))
        cursor = match.end()
    if raw_attrs[cursor:].strip():
        raise KimiK3XTMLParseError(f"Malformed XTML attributes: {raw_attrs!r}.")
    return attrs


def _require_attrs(
    raw_attrs: str,
    *,
    required: frozenset[str],
    tag: str,
) -> dict[str, str]:
    attrs = _parse_attrs(raw_attrs)
    if frozenset(attrs) != required:
        raise KimiK3XTMLParseError(f"K3 {tag} attributes must be exactly {sorted(required)}, got {sorted(attrs)}.")
    return attrs


def _reject_json_constant(value: str):
    raise KimiK3XTMLParseError(f"Non-finite JSON number {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise KimiK3XTMLParseError(f"Duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _load_json(raw_value: str):
    try:
        return json.loads(
            raw_value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except KimiK3XTMLParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise KimiK3XTMLParseError("Invalid JSON in K3 tool arguments.") from exc


def _json_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_typed_argument(raw_value: str, value_type: str):
    if value_type == "string":
        return raw_value

    value = _load_json(raw_value)
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise KimiK3XTMLParseError("K3 number argument is not numeric.")
        if isinstance(value, float) and not math.isfinite(value):
            raise KimiK3XTMLParseError("K3 number argument must be finite.")
    elif value_type == "boolean":
        if not isinstance(value, bool):
            raise KimiK3XTMLParseError("K3 boolean argument is not a boolean.")
    elif value_type == "null":
        if value is not None:
            raise KimiK3XTMLParseError("K3 null argument is not null.")
    elif value_type == "object":
        if not isinstance(value, dict):
            raise KimiK3XTMLParseError("K3 object argument is not an object.")
    elif value_type == "array":
        if not isinstance(value, list):
            raise KimiK3XTMLParseError("K3 array argument is not an array.")
    else:
        raise KimiK3XTMLParseError(f"Unsupported K3 argument type: {value_type!r}.")
    return value


@dataclass(frozen=True)
class KimiK3ParseDelta:
    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[KimiK3ToolCallDelta, ...] = ()

    def merged(self, other: KimiK3ParseDelta) -> KimiK3ParseDelta:
        calls: dict[int, KimiK3ToolCallDelta] = {}
        for call in (*self.tool_calls, *other.tool_calls):
            previous = calls.get(call.index)
            if previous is None:
                calls[call.index] = call
                continue
            calls[call.index] = KimiK3ToolCallDelta(
                index=call.index,
                name=previous.name if previous.name is not None else call.name,
                arguments=(previous.arguments or "") + (call.arguments or "")
                if previous.arguments is not None or call.arguments is not None
                else None,
            )
        return KimiK3ParseDelta(
            reasoning=self.reasoning + other.reasoning,
            content=self.content + other.content,
            tool_calls=tuple(calls.values()),
        )


class KimiK3XTMLParser:
    """Incrementally parse one K3 assistant XTML envelope."""

    def __init__(
        self,
        *,
        thinking_enabled: bool,
        tool_mode: ToolMode,
        allowed_tool_names: frozenset[str],
        named_tool: str | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.thinking_enabled = thinking_enabled
        self.tool_mode = tool_mode
        self.allowed_tool_names = allowed_tool_names
        self.named_tool = named_tool
        self._token_decoder = _KimiK3TokenDecoder(tokenizer) if tokenizer is not None else None
        self._buffer = ""
        self._phase = "start"
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._protocol_complete = False
        self._finished = False
        self._active_call_index: int | None = None
        self._active_call_name: str | None = None
        self._active_call_style: Literal["typed", "json"] | None = None
        self._active_argument_key: str | None = None
        self._active_argument_type: str | None = None
        self._active_argument_parts: list[str] = []
        self._seen_argument_keys: set[str] = set()
        self._streamed_arguments: list[list[str]] = []
        self._completed_calls: list[KimiK3ParsedCall] = []

    @property
    def snapshot(self) -> KimiK3ParseSnapshot:
        calls = () if self.tool_mode == "none" else tuple(self._completed_calls)
        return KimiK3ParseSnapshot(
            reasoning="".join(self._reasoning_parts),
            content="".join(self._content_parts),
            tool_calls=calls,
        )

    @staticmethod
    def _add_delta(
        deltas: dict[int, KimiK3ToolCallDelta],
        index: int,
        *,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        previous = deltas.get(index)
        if previous is None:
            deltas[index] = KimiK3ToolCallDelta(
                index=index,
                name=name,
                arguments=arguments,
            )
            return
        deltas[index] = KimiK3ToolCallDelta(
            index=index,
            name=previous.name if previous.name is not None else name,
            arguments=(previous.arguments or "") + (arguments or "")
            if previous.arguments is not None or arguments is not None
            else None,
        )

    def _emit_arguments(
        self,
        deltas: dict[int, KimiK3ToolCallDelta],
        arguments: str,
    ) -> None:
        index = self._active_call_index
        if index is None:
            raise KimiK3XTMLParseError("K3 streamed arguments have no active call.")
        self._streamed_arguments[index].append(arguments)
        if self.tool_mode != "none":
            self._add_delta(deltas, index, arguments=arguments)

    def _emit_reasoning(self, parts: list[str], value: str) -> None:
        if value:
            self._reasoning_parts.append(value)
            parts.append(value)

    def _emit_content(self, parts: list[str], value: str) -> None:
        if value:
            self._content_parts.append(value)
            parts.append(value)

    def _begin_call(
        self,
        raw_attrs: str,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        attrs = _require_attrs(
            raw_attrs,
            required=frozenset({"tool", "index"}),
            tag="call",
        )
        name = attrs["tool"]
        if not name:
            raise KimiK3XTMLParseError("K3 tool name must not be empty.")
        try:
            protocol_index = int(attrs["index"])
        except ValueError as exc:
            raise KimiK3XTMLParseError("K3 call index must be an integer.") from exc
        expected_index = len(self._streamed_arguments) + 1
        if protocol_index <= 0 or str(protocol_index) != attrs["index"]:
            raise KimiK3XTMLParseError("K3 call index must be a canonical positive integer.")
        if protocol_index != expected_index:
            raise KimiK3XTMLParseError(
                "K3 call indices must be unique and sequential from 1; "
                f"expected {expected_index}, got {protocol_index}."
            )
        if self.tool_mode != "none" and name not in self.allowed_tool_names:
            raise KimiK3XTMLParseError(f"Unknown K3 tool {name!r}.")
        if self.named_tool is not None and name != self.named_tool:
            raise KimiK3XTMLParseError(f"Named K3 tool choice requires {self.named_tool!r}, got {name!r}.")

        index = protocol_index - 1
        self._active_call_index = index
        self._active_call_name = name
        self._active_call_style = None
        self._active_argument_key = None
        self._seen_argument_keys.clear()
        self._streamed_arguments.append([])
        if self.tool_mode != "none":
            self._add_delta(deltas, index, name=name, arguments="")

    def _begin_typed_argument(
        self,
        raw_attrs: str,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        if self._active_call_style == "json":
            raise KimiK3XTMLParseError("K3 call mixes json arguments with typed arguments or stray text.")
        attrs = _require_attrs(
            raw_attrs,
            required=frozenset({"key", "type"}),
            tag="argument",
        )
        key = attrs["key"]
        value_type = attrs["type"]
        if not key:
            raise KimiK3XTMLParseError("K3 argument key must not be empty.")
        if key in self._seen_argument_keys:
            raise KimiK3XTMLParseError(f"Duplicate K3 argument key {key!r}.")
        if value_type not in {"string", "number", "boolean", "null", "object", "array"}:
            raise KimiK3XTMLParseError(f"Unsupported K3 argument type: {value_type!r}.")

        self._active_call_style = "typed"
        self._active_argument_key = key
        self._active_argument_type = value_type
        self._active_argument_parts.clear()
        self._seen_argument_keys.add(key)
        prefix = "{" if len(self._seen_argument_keys) == 1 else ","
        prefix += _json_compact(key) + ":"
        if value_type == "string":
            prefix += '"'
        self._emit_arguments(deltas, prefix)

    def _begin_json_argument(self, raw_attrs: str) -> None:
        if self._active_call_style is not None:
            raise KimiK3XTMLParseError("K3 call mixes json arguments with typed arguments or stray text.")
        attrs = _require_attrs(
            raw_attrs,
            required=frozenset({"type"}),
            tag="json",
        )
        if attrs["type"] != "object":
            raise KimiK3XTMLParseError("K3 json arguments must use type='object'.")
        self._active_call_style = "json"
        self._active_argument_key = None
        self._active_argument_type = "json"
        self._active_argument_parts.clear()

    def _stream_open_value(
        self,
        marker: str,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> bool:
        if not _has_control_sentinel(self._buffer):
            raw_value = self._buffer
            self._buffer = ""
            self._append_value_fragment(raw_value, deltas)
            return False

        marker_index = self._buffer.find(marker)
        if marker_index >= 0:
            raw_value = self._buffer[:marker_index]
            _reject_unexpected_control_tokens(raw_value)
            self._buffer = self._buffer[marker_index + len(marker) :]
            self._finish_value(raw_value, deltas)
            return True

        overlap = partial_marker_overlap(self._buffer, marker)
        safe_end = len(self._buffer) - overlap
        if safe_end <= 0:
            return False
        raw_value = self._buffer[:safe_end]
        _reject_unexpected_control_tokens(raw_value)
        self._buffer = self._buffer[safe_end:]
        self._append_value_fragment(raw_value, deltas)
        return False

    def _append_value_fragment(
        self,
        raw_value: str,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        value_type = self._active_argument_type
        if value_type is None:
            raise KimiK3XTMLParseError("K3 streamed value has no active argument.")
        self._active_argument_parts.append(raw_value)
        if not raw_value:
            return
        if value_type == "string":
            self._emit_arguments(
                deltas,
                json.dumps(raw_value, ensure_ascii=False)[1:-1],
            )
        elif value_type in ("object", "array", "json"):
            self._emit_arguments(deltas, raw_value)

    def _finish_value(
        self,
        raw_value: str,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        self._append_value_fragment(raw_value, deltas)
        value_type = self._active_argument_type
        full_value = "".join(self._active_argument_parts)
        if value_type == "json":
            value = _load_json(full_value)
            if not isinstance(value, dict):
                raise KimiK3XTMLParseError("K3 json tool arguments must be an object.")
        elif value_type is not None:
            value = _decode_typed_argument(full_value, value_type)
            if value_type == "string":
                self._emit_arguments(deltas, '"')
            elif value_type not in ("object", "array"):
                self._emit_arguments(deltas, _json_compact(value))
        self._active_argument_type = None
        self._active_argument_key = None
        self._active_argument_parts.clear()

    def _finish_call(
        self,
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        index = self._active_call_index
        name = self._active_call_name
        if index is None or name is None:
            raise KimiK3XTMLParseError("K3 call ended without an active call.")
        if self._active_call_style is None:
            self._emit_arguments(deltas, "{}")
        elif self._active_call_style == "typed":
            self._emit_arguments(deltas, "}")
        arguments = "".join(self._streamed_arguments[index])
        value = _load_json(arguments)
        if not isinstance(value, dict):
            raise KimiK3XTMLParseError("K3 streamed tool arguments must be an object.")
        self._completed_calls.append(
            KimiK3ParsedCall(
                name=name,
                index=index + 1,
                arguments=arguments,
            )
        )
        self._active_call_index = None
        self._active_call_name = None
        self._active_call_style = None
        self._active_argument_key = None
        self._seen_argument_keys.clear()

    @staticmethod
    def _partial_tag(buffer: str, prefix: str) -> bool:
        return prefix.startswith(buffer) or buffer.startswith(prefix)

    def _stream_channel(
        self,
        *,
        end_marker: str,
        forbidden_markers: tuple[str, ...],
        final: bool,
        emit,
    ) -> bool:
        if not _has_control_sentinel(self._buffer):
            safe_text = self._buffer
            self._buffer = ""
            emit(safe_text)
            if final:
                self._phase = "truncated"
            return False

        end_index = self._buffer.find(end_marker)
        if end_index >= 0:
            before_end = self._buffer[:end_index]
            if any(marker in before_end for marker in forbidden_markers):
                raise KimiK3XTMLParseError("Unexpected XTML block before the current K3 channel was closed.")
            _reject_unexpected_control_tokens(before_end)
            emit(before_end)
            self._buffer = self._buffer[end_index + len(end_marker) :]
            return True

        if any(marker in self._buffer for marker in forbidden_markers):
            raise KimiK3XTMLParseError("Unexpected XTML block before the current K3 channel was closed.")

        markers = (end_marker, *forbidden_markers)
        overlap = max((partial_marker_overlap(self._buffer, marker) for marker in markers), default=0)
        if final:
            safe_end = len(self._buffer) - overlap
            safe_text = self._buffer[:safe_end]
            _reject_unexpected_control_tokens(safe_text)
            emit(safe_text)
            self._buffer = ""
            self._phase = "truncated"
            return False

        safe_end = len(self._buffer) - overlap
        if safe_end > 0:
            safe_text = self._buffer[:safe_end]
            _reject_unexpected_control_tokens(safe_text)
            emit(safe_text)
            self._buffer = self._buffer[safe_end:]
        return False

    def _process_buffer(
        self,
        *,
        final: bool,
        reasoning_parts: list[str],
        content_parts: list[str],
        deltas: dict[int, KimiK3ToolCallDelta],
    ) -> None:
        while True:
            if self._phase == "start":
                marker = _THINK_START if self.thinking_enabled else _RESPONSE_START
                next_phase = "reasoning" if self.thinking_enabled else "response"
                if self._buffer.startswith(marker):
                    self._buffer = self._buffer[len(marker) :]
                    self._phase = next_phase
                    continue
                if self._buffer and marker.startswith(self._buffer):
                    if final:
                        self._buffer = ""
                        self._phase = "truncated"
                    return
                self._phase = next_phase
                continue

            if self._phase == "reasoning":
                if self._stream_channel(
                    end_marker=_THINK_END,
                    forbidden_markers=(_RESPONSE_START, _RESPONSE_END, _TOOLS_START, _TOOLS_END),
                    final=final,
                    emit=lambda value: self._emit_reasoning(reasoning_parts, value),
                ):
                    self._phase = "response_start"
                    continue
                return

            if self._phase == "response_start":
                if self._buffer.startswith(_RESPONSE_START):
                    self._buffer = self._buffer[len(_RESPONSE_START) :]
                    self._phase = "response"
                    continue
                if not self._buffer or _RESPONSE_START.startswith(self._buffer):
                    if final:
                        self._buffer = ""
                        self._phase = "truncated"
                    return
                raise KimiK3XTMLParseError("K3 think block must be followed by a response block.")

            if self._phase == "response":
                if self._stream_channel(
                    end_marker=_RESPONSE_END,
                    forbidden_markers=(_THINK_START, _THINK_END, _TOOLS_START, _TOOLS_END),
                    final=final,
                    emit=lambda value: self._emit_content(content_parts, value),
                ):
                    self._phase = "post_response"
                    continue
                return

            if self._phase == "post_response":
                if self._buffer.startswith(_TOOLS_START):
                    self._buffer = self._buffer[len(_TOOLS_START) :]
                    self._phase = "tools"
                    continue
                if self._buffer.startswith(_MESSAGE_END):
                    self._buffer = self._buffer[len(_MESSAGE_END) :]
                    self._protocol_complete = True
                    self._phase = "after_message"
                    continue
                if not self._buffer or any(marker.startswith(self._buffer) for marker in (_TOOLS_START, _MESSAGE_END)):
                    return
                raise KimiK3XTMLParseError("K3 response is missing the closing message marker.")

            if self._phase == "tools":
                stripped = self._buffer.lstrip()
                if stripped != self._buffer:
                    self._buffer = stripped
                    continue
                if self._buffer.startswith(_TOOLS_END):
                    self._buffer = self._buffer[len(_TOOLS_END) :]
                    self._phase = "after_tools"
                    continue
                if not self._buffer or _TOOLS_END.startswith(self._buffer):
                    return
                call_match = _CALL_START_RE.match(self._buffer)
                if call_match is not None:
                    self._buffer = self._buffer[call_match.end() :]
                    self._begin_call(call_match.group("attrs"), deltas)
                    self._phase = "call"
                    continue
                if self._partial_tag(self._buffer, f"{_OPEN_TOKEN}call"):
                    if len(self._buffer) > _MAX_PENDING_TAG_CHARS:
                        raise KimiK3XTMLParseError("K3 call tag exceeds the maximum supported length.")
                    return
                raise KimiK3XTMLParseError("Unexpected text or tag in K3 tools block.")

            if self._phase == "after_tools":
                if self._buffer.startswith(_MESSAGE_END):
                    self._buffer = self._buffer[len(_MESSAGE_END) :]
                    self._protocol_complete = True
                    self._phase = "after_message"
                    continue
                if not self._buffer or _MESSAGE_END.startswith(self._buffer):
                    return
                raise KimiK3XTMLParseError("K3 response is missing the closing message marker.")

            if self._phase == "after_message":
                if self._buffer.startswith(_END_OF_MSG_TOKEN):
                    self._buffer = self._buffer[len(_END_OF_MSG_TOKEN) :]
                    self._phase = "done"
                    continue
                if not self._buffer:
                    if final:
                        self._phase = "done"
                    return
                if _END_OF_MSG_TOKEN.startswith(self._buffer):
                    if final:
                        self._buffer = ""
                        self._phase = "truncated"
                    return
                if not self._buffer.strip():
                    self._buffer = ""
                    self._phase = "done"
                    continue
                raise KimiK3XTMLParseError("Unexpected text or additional XTML blocks after the K3 response.")

            if self._phase == "done":
                if self._buffer.strip():
                    raise KimiK3XTMLParseError("Unexpected text or additional XTML blocks after the K3 response.")
                self._buffer = ""
                return

            if self._phase == "truncated":
                if self._buffer:
                    raise KimiK3XTMLParseError("Unexpected text after a truncated K3 response.")
                return

            if self._phase in ("typed_value", "json_value"):
                marker = _JSON_END if self._phase == "json_value" else _ARGUMENT_END
                if not self._stream_open_value(marker, deltas):
                    return
                self._phase = "call"
                continue

            if self._phase == "call":
                stripped = self._buffer.lstrip()
                if stripped != self._buffer:
                    self._buffer = stripped
                    continue
                if self._buffer.startswith(_CALL_END):
                    self._buffer = self._buffer[len(_CALL_END) :]
                    self._finish_call(deltas)
                    self._phase = "tools"
                    continue
                if not self._buffer or _CALL_END.startswith(self._buffer):
                    return
                argument_match = _ARGUMENT_START_RE.match(self._buffer)
                if argument_match is not None:
                    self._buffer = self._buffer[argument_match.end() :]
                    self._begin_typed_argument(argument_match.group("attrs"), deltas)
                    self._phase = "typed_value"
                    continue
                json_match = _JSON_START_RE.match(self._buffer)
                if json_match is not None:
                    self._buffer = self._buffer[json_match.end() :]
                    self._begin_json_argument(json_match.group("attrs"))
                    self._phase = "json_value"
                    continue
                if any(
                    self._partial_tag(self._buffer, marker)
                    for marker in (f"{_OPEN_TOKEN}argument", f"{_OPEN_TOKEN}json")
                ):
                    if len(self._buffer) > _MAX_PENDING_TAG_CHARS:
                        raise KimiK3XTMLParseError("K3 argument tag exceeds the maximum supported length.")
                    return
                raise KimiK3XTMLParseError("Unexpected text or tag in K3 typed arguments.")

            raise AssertionError(f"Unknown K3 parser phase: {self._phase}")

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int] = (),
    ) -> KimiK3ParseDelta:
        if self._finished:
            raise KimiK3XTMLParseError("K3 parser cannot accept data after finish().")
        if delta_token_ids and self._token_decoder is not None:
            protocol_text = self._token_decoder.decode(delta_token_ids)
        elif delta_text:
            raise KimiK3XTMLParseError("K3 parsing requires original model output token IDs.")
        else:
            return KimiK3ParseDelta()

        if not protocol_text:
            return KimiK3ParseDelta()

        self._buffer += protocol_text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        deltas: dict[int, KimiK3ToolCallDelta] = {}
        self._process_buffer(
            final=False,
            reasoning_parts=reasoning_parts,
            content_parts=content_parts,
            deltas=deltas,
        )
        return KimiK3ParseDelta(
            reasoning="".join(reasoning_parts),
            content="".join(content_parts),
            tool_calls=tuple(deltas.values()),
        )

    def finish(self) -> KimiK3ParseDelta:
        if self._finished:
            return KimiK3ParseDelta()

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        deltas: dict[int, KimiK3ToolCallDelta] = {}
        pending = self._token_decoder.finish() if self._token_decoder is not None else ""
        if pending:
            self._buffer += pending
        self._process_buffer(
            final=True,
            reasoning_parts=reasoning_parts,
            content_parts=content_parts,
            deltas=deltas,
        )

        if self._protocol_complete and self.tool_mode in ("required", "named") and not self._completed_calls:
            raise KimiK3XTMLParseError(f"K3 tool_choice={self.tool_mode!r} completed without a valid tool call.")

        self._finished = True
        return KimiK3ParseDelta(
            reasoning="".join(reasoning_parts),
            content="".join(content_parts),
            tool_calls=tuple(deltas.values()),
        )
