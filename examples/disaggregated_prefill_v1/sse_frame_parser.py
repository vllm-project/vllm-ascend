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
"""Incremental SSE frame parsing for the PD load-balance proxies.

The decode backend always answers with SSE: one ``data: {json}`` frame per
output chunk, terminated by a ``data: [DONE]`` sentinel. Network boundaries
do not respect frame boundaries, so a proxy that parses each network chunk
independently loses frames:

* A frame's JSON split across two TCP segments fails to parse in both halves,
  so the frame -- and the token it carries -- is silently dropped.
* Two frames coalesced into a single read (``{"choices":[...]}`` followed by
  ``data: [DONE]``) make ``json.loads`` fail, because it requires the whole
  input to be a single JSON value. A complete response is dropped and the
  client receives an empty body.

``SSEFrameParser`` reassembles the byte stream into frames before decoding
them, so both cases are handled.
"""

import json
from typing import Any, NamedTuple

# The SSE frame terminator: a blank line.
SSE_FRAME_SEPARATOR = b"\n\n"

# Terminal sentinel emitted by the OpenAI-compatible streaming API.
SSE_DONE_SENTINEL = "[DONE]"


class SSEFrame(NamedTuple):
    """One decoded SSE frame.

    Attributes:
        payload: the decoded JSON object of the frame.
        is_sse: whether the frame carried the ``data:`` SSE prefix.
    """

    payload: dict[str, Any]
    is_sse: bool


class _Sentinel:
    """Marker object, compared by identity."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


#: The buffered bytes do not yet form a complete JSON value.
INCOMPLETE = _Sentinel("<incomplete>")

#: The frame is the ``[DONE]`` terminal sentinel.
DONE = _Sentinel("<done>")

#: The frame is terminated but its body is not a valid JSON object.
MALFORMED = _Sentinel("<malformed>")


class SSEFrameParser:
    """Reassemble an SSE byte stream into decoded JSON frames.

    Feed the raw network chunks to :meth:`feed`; it returns the frames that
    became complete. Frames that are still arriving stay buffered until the
    rest of their bytes show up.

    The parser never mutates the caller's data: it only reads what it is given,
    so the proxy can forward the original bytes to the client unchanged.
    """

    def __init__(self) -> None:
        self._buffer = b""

    @property
    def pending(self) -> bytes:
        """Bytes held back because they do not form a complete frame yet."""
        return self._buffer

    def feed(self, chunk: bytes) -> list[SSEFrame]:
        """Add *chunk* to the buffer and return the frames it completes.

        The ``[DONE]`` sentinel is not returned: it terminates the stream and
        carries no payload.
        """
        self._buffer += chunk
        frames: list[SSEFrame] = []
        while True:
            frame, self._buffer = self._split_next_frame(self._buffer)
            if frame is INCOMPLETE:
                break
            if frame is DONE:
                # Terminal sentinel: drop any trailing bytes and stop.
                self._buffer = b""
                break
            if frame is MALFORMED:
                # Unparsable frame: skip it and keep reading the stream.
                continue
            frames.append(frame)
        return frames

    def flush(self) -> SSEFrame | None:
        """Decode a trailing frame that was never terminated by a blank line.

        A backend may close the connection right after the last frame without
        sending the final blank line. Returns the decoded frame, or ``None``
        when the buffer holds nothing parseable.
        """
        remainder = self._buffer.strip()
        self._buffer = b""
        if not remainder:
            return None
        payload = self._decode_frame(remainder)
        if payload is INCOMPLETE or payload is DONE or payload is MALFORMED:
            return None
        return payload

    @staticmethod
    def _split_next_frame(buffer: bytes) -> tuple[Any, bytes]:
        """Split the first frame off *buffer*.

        Returns ``(INCOMPLETE, buffer)`` when the frame is not complete yet.
        """
        separator = buffer.find(SSE_FRAME_SEPARATOR)
        if separator < 0:
            return INCOMPLETE, buffer
        raw = buffer[:separator]
        rest = buffer[separator + len(SSE_FRAME_SEPARATOR) :]
        payload = SSEFrameParser._decode_frame(raw)
        if payload is INCOMPLETE:
            # A terminated frame whose body is not valid JSON is malformed.
            # Drop it instead of stalling the whole stream.
            return MALFORMED, rest
        return payload, rest

    @staticmethod
    def _decode_frame(raw: bytes) -> Any:
        """Decode one raw SSE frame body.

        Returns an :class:`SSEFrame`, ``INCOMPLETE`` when the body is not a
        complete JSON value, ``DONE`` for the terminal sentinel, or
        ``MALFORMED`` when the body is not a JSON object.
        """
        text = raw.strip()
        if not text:
            return INCOMPLETE
        is_sse = text.startswith(b"data:")
        if is_sse:
            text = text[len(b"data:") :].strip()
        if not text:
            return INCOMPLETE
        if text == SSE_DONE_SENTINEL.encode():
            return DONE
        try:
            payload, _ = json.JSONDecoder().raw_decode(text.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return MALFORMED
        if not isinstance(payload, dict):
            return MALFORMED
        return SSEFrame(payload, is_sse)
