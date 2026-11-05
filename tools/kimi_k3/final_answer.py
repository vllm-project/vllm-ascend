# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict final-channel and numeric-answer handling for GSM8K."""

import regex as re


def final_message(message, finish_reason):
    # A truncated response is not evidence of a completed answer.
    if finish_reason != "stop":
        return ""
    content = message.get("content") or ""
    if not content.strip():
        raw = message.get("reasoning") or message.get("reasoning_content") or ""
        match = re.fullmatch(
            r"\s*<\|open\|>response<\|sep\|>(.*?)<\|close\|>response<\|sep\|>"
            r"(?:\s*<\|close\|>message<\|sep\|>)?\s*",
            raw,
            re.S,
        )
        content = match[1] if match else ""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    if "<|open|>response<|sep|>" in content:
        match = re.search(r"<\|open\|>response<\|sep\|>(.*?)<\|close\|>response<\|sep\|>", content, re.S)
        return match[1].strip() if match else ""
    if "<think>" in content or "<|open|>think<|sep|>" in content:
        return ""
    return content.strip()


def gsm8k_final_postprocess(text):
    """Require an explicit answer or a numeric-only final response."""
    text = text.strip()
    number = r"([-+]?\d[\d,]*(?:\.\d+)?)"
    patterns = [
        r"(?:the\s+(?:final\s+)?answer\s+is|final\s+answer\s*[:=]|answer\s*[:=]|####)\s*\$?\s*" + number,
        r"\\boxed\{\s*" + number + r"\s*\}",
    ]
    matches = []
    for pattern in patterns:
        matches.extend((m.start(), m[1]) for m in re.finditer(pattern, text, re.I))
    if matches:
        return max(matches)[1].replace(",", "")
    match = re.fullmatch(r"\s*\$?\s*" + number + r"\s*[.!]?\s*", text)
    return match[1].replace(",", "") if match else "NULL"
