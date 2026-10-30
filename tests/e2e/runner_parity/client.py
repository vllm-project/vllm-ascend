# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Small HTTP collector for paired runners; no model or accelerator imports."""

import json
import time
import urllib.request
import uuid
from pathlib import Path


def post_json(base_url, route, body, timeout=1800):
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_completion(base_url, body, timeout=1800, error_dir=None):
    """Measure client-visible times, preserving the raw SSE events and usage."""
    payload = dict(body, stream=True, stream_options={"include_usage": True}, return_token_ids=True)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    first_visible = None
    token_ids = []
    events = []
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            content = line[5:].strip()
            if content == b"[DONE]":
                break
            event = json.loads(content)
            now = time.perf_counter()
            events.append({"elapsed_s": now - start, "data": event})
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices", [])
            delta_ids = [token for choice in choices for token in (choice.get("token_ids") or [])]
            if delta_ids:
                token_ids.extend(delta_ids)
                first = now if first is None else first
                last = now
            if first_visible is None and any(choice.get("text") for choice in choices):
                first_visible = now
    end = time.perf_counter()
    if (
        not usage
        or not usage.get("completion_tokens")
        or first is None
        or last is None
        or len(token_ids) != usage["completion_tokens"]
    ):
        details = {
            "request": payload,
            "usage": usage,
            "generated_token_ids": token_ids,
            "events": events,
            "e2e_s": end - start,
            "error": "Token-ID stream and final usage must be complete and agree",
        }
        if error_dir is not None:
            (Path(error_dir) / f"stream-error-{uuid.uuid4().hex}.json").write_text(
                json.dumps(details), encoding="utf-8"
            )
        raise AssertionError("Missing or inconsistent token-ID stream/usage; no latency result is valid")
    count = usage["completion_tokens"]
    return {
        "usage": usage,
        "ttft_s": first - start,
        "tpot_s": (last - first) / (count - 1) if count > 1 else None,
        "e2e_s": end - start,
        "generated_token_ids": token_ids,
        "first_visible_text_s": first_visible - start if first_visible is not None else None,
        "timing_source": "Generated token-ID chunks, including special tokens; not detokenized text visibility",
        "events": events,
    }
