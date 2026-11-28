#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Reassemble decoder responses before inspecting or forwarding their bytes.

SSE frames retain their original bytes, including the delimiter. Non-streaming
JSON is decoded at EOF: HTTP chunks and pretty-printed JSON blank lines are not
response boundaries. Each decoder attempt must use a fresh parser.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import NamedTuple

import regex as re

SSE_FRAME_SEPARATOR = re.compile(rb"\r?\n\r?\n")


class SSEFrame(NamedTuple):
    payload: dict | None
    is_sse: bool
    raw: bytes


class SSEFrameParser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._is_sse: bool | None = None
        self._done = False

    @property
    def pending(self) -> bytes:
        return bytes(self._buffer)

    def feed(self, chunk: bytes) -> list[SSEFrame]:
        if self._done:
            return []
        self._buffer.extend(chunk)
        if self._is_sse is None:
            prefix = self._buffer.lstrip()
            if not prefix:
                return []
            # JSON responses start with an object/array. All other prefixes
            # may be partial SSE field names (data, event, id, or a comment).
            self._is_sse = prefix[:1] not in (b"{", b"[")
        if not self._is_sse:
            return []

        frames = []
        start = 0
        while match := SSE_FRAME_SEPARATOR.search(self._buffer, start):
            raw = bytes(self._buffer[start : match.end()])
            start = match.end()
            frame = self._decode_frame(raw)
            if frame is not None:
                frames.append(frame)
            if self._done:
                self._buffer.clear()
                return frames
        # Compact once per network chunk, rather than copying the remaining
        # buffer for every frame in a coalesced read.
        if start:
            del self._buffer[:start]
        return frames

    def flush(self) -> SSEFrame | None:
        raw = bytes(self._buffer)
        self._buffer.clear()
        if not raw.strip():
            return None
        if not self._is_sse:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Decoder response must be a JSON object")
            return SSEFrame(payload, False, raw)
        # Preserve the existing tolerance for a valid final event without a
        # blank line, but never expose a partial JSON event to the client.
        return self._decode_frame(raw.rstrip(b"\r\n") + b"\n\n")

    def _decode_frame(self, raw: bytes) -> SSEFrame | None:
        data = []
        for line in raw.splitlines():
            if line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
        if not data:
            return SSEFrame(None, True, raw)
        value = b"\n".join(data).strip()
        if value == b"[DONE]":
            self._done = True
            return SSEFrame(None, True, raw)
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return SSEFrame(payload, True, raw)


async def iter_response_frames(source: AsyncGenerator[bytes, None]) -> AsyncGenerator[SSEFrame, None]:
    """Yield complete events, and flush non-streaming JSON at clean EOF only."""
    parser = SSEFrameParser()
    async with aclosing(source):
        async for chunk in source:
            for frame in parser.feed(chunk):
                yield frame
        frame = parser.flush()
        if frame is not None:
            yield frame
