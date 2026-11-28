# SPDX-License-Identifier: Apache-2.0
"""Messages and Responses text continuation for the layerwise PD example."""

import copy
import json


async def response_frames(response, streaming):
    if not streaming:
        yield await response.aread()
        return

    pending = bytearray()
    trailing_cr = False
    async for chunk in response.aiter_bytes():
        # SSE only recognizes CR and LF. Unicode line separators are legal
        # inside a JSON string and must survive even a split UTF-8 codepoint.
        if trailing_cr:
            chunk = chunk.removeprefix(b"\n")
        trailing_cr = chunk.endswith(b"\r")
        search_from = max(0, len(pending) - 1)
        pending.extend(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        start = 0
        while (end := pending.find(b"\n\n", search_from)) != -1:
            yield bytes(pending[start : end + 2])
            start = search_from = end + 2
        del pending[:start]


class TextContinuation:
    def __init__(self, api, request):
        self.api = api
        self.original = request
        self.request = request.copy()
        self.streaming = bool(request.get("stream"))
        self.previous_items = []
        self.previous_text = ""
        self.continued = False
        self.response_id = request.get("request_id")
        self.next_sequence = 0

    def begin_attempt(self):
        self.recompute = False
        self.text = []
        self.items = []
        self.structured = False

    def _resume(self, text, items, usage):
        if self.structured:
            raise ValueError("Recompute continuation requires text-only output")
        self.previous_items.extend(items)
        self.previous_text += text
        messages_api = self.api == "/messages"
        key = "messages" if messages_api else "input"
        original = self.original.get(key)
        if not messages_api and (isinstance(original, str) or not original):
            self.request[key] = (original or "") + self.previous_text
        elif isinstance(original, list) and original:
            first = original[0]
            if not isinstance(first, dict) or "content" not in first or first.get("type", "message") != "message":
                raise ValueError("Recompute requires a first message with content")
            content = copy.deepcopy(first["content"])
            if isinstance(content, list):
                if self.previous_text:
                    content.append({"type": "text" if messages_api else "input_text", "text": self.previous_text})
            else:
                content += self.previous_text
            self.request[key] = [{**first, "content": content}, *original[1:]]
        else:
            raise ValueError("Recompute requires string input or a first message with content")

        limit = "max_tokens" if messages_api else "max_output_tokens"
        if self.request.get(limit) is not None:
            self.request[limit] -= (usage or {}).get("output_tokens", 0)
        self.continued = self.recompute = True

    def _message(self, payload):
        event = payload.get("type")
        if event == "message_start" and self.continued:
            return False
        if event == "content_block_start":
            block = payload["content_block"]
            self.items.append(block)
            self.structured |= block["type"] != "text"
            self.text.append(block.get("text", ""))
        elif event == "content_block_delta":
            delta = payload["delta"]
            self.structured |= delta["type"] != "text_delta"
            self.text.append(delta.get("text", ""))
        if self.streaming and "index" in payload and self.continued:
            payload["index"] += len(self.previous_items)
        if self.streaming and event != "message_delta":
            return True

        result = payload.get("delta", {}) if self.streaming else payload
        reason = result.get("stop_reason")
        mapped_stop = (
            reason == "stop_sequence"
            and result.get("stop_sequence") == "recomputed"
            and "recomputed" not in (self.original.get("stop_sequences") or [])
        )
        if reason == "recomputed" or mapped_stop:
            if not self.streaming:
                self.items = payload.get("content", [])
                self.structured = any(block.get("type") != "text" for block in self.items)
                self.text = [block.get("text", "") for block in self.items]
            self._resume("".join(self.text), self.items, payload.get("usage"))
            return False
        if self.previous_items and not self.streaming and "content" in payload:
            payload["content"] = self.previous_items + payload["content"]
        return True

    def _response(self, payload):
        event = payload.get("type")
        result = payload.get("response", payload)
        if self.continued and event in ("response.created", "response.in_progress"):
            return False
        if "id" in result and ("response" in payload or not self.streaming):
            self.response_id = self.response_id or result["id"]
            result["id"] = self.response_id
        if "output_index" in payload and self.continued:
            payload["output_index"] += len(self.previous_items)
        if event == "response.output_text.delta":
            self.text.append(payload["delta"])
        elif event in ("response.output_item.added", "response.output_item.done"):
            self.structured |= payload["item"].get("type") != "message"
            if event == "response.output_item.done":
                self.items.append(payload["item"])
        elif event == "response.content_part.added":
            self.structured |= payload["part"].get("type") != "output_text"
        if not self.streaming or event in ("response.completed", "response.incomplete"):
            items = result.get("output") or self.items
            if result.get("stop_reason") == "recomputed":
                self.structured |= any(
                    item.get("type") != "message"
                    or any(part.get("type") != "output_text" for part in item.get("content", []))
                    for item in items
                )
                text = "".join(part.get("text", "") for item in items for part in item.get("content", []))
                self._resume(text or "".join(self.text), items, result.get("usage"))
                return False
            if self.previous_items and "output" in result:
                result["output"] = self.previous_items + items
        if "sequence_number" in payload:
            if self.continued:
                payload["sequence_number"] = self.next_sequence
            self.next_sequence = payload["sequence_number"] + 1
        return True

    def forward(self, frame):
        """Inspect a complete JSON body or SSE event before exposing it to the client."""
        if self.streaming:
            lines = frame.split(b"\n")
            data = b"\n".join(line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:"))
            if not data or data == b"[DONE]":
                return frame
        else:
            data = frame
        payload = json.loads(data)
        keep = self._message(payload) if self.api == "/messages" else self._response(payload)
        if not keep:
            return None
        if not self.continued and not (self.api == "/responses" and self.original.get("request_id")):
            return frame
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        if not self.streaming:
            return encoded
        fields = [line for line in lines if line and not line.startswith(b"data:")]
        return b"\n".join([*fields, b"data: " + encoded]) + b"\n\n"
