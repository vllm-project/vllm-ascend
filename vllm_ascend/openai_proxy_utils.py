from __future__ import annotations

from typing import Any


def append_text_to_prompt(prompt: Any, text: str) -> Any:
    """Append generated text to an OpenAI-compatible `prompt` field.

    The OpenAI `/v1/completions` API allows `prompt` to be a string or a list.
    This helper performs a best-effort append while preserving the input type.
    """
    if not text:
        return prompt
    if prompt is None:
        return text
    if isinstance(prompt, str):
        return prompt + text
    if isinstance(prompt, list):
        if not prompt:
            return [text]
        new_prompt = list(prompt)
        last = new_prompt[-1]
        if isinstance(last, str):
            new_prompt[-1] = last + text
            return new_prompt
        new_prompt[-1] = str(last) + text
        return new_prompt
    return str(prompt) + text


def append_text_to_chat_content(content: Any, text: str) -> Any:
    """Append generated text to an OpenAI-compatible chat message `content`.

    The OpenAI `/v1/chat/completions` API supports `content` as either a string
    or a list of content parts (e.g. `[{\"type\": \"text\", \"text\": \"...\"}]`).
    """
    if not text:
        return content
    if content is None:
        return text
    if isinstance(content, str):
        return content + text
    if isinstance(content, list):
        if not content:
            return [{"type": "text", "text": text}]
        new_content = list(content)
        last = new_content[-1]
        if isinstance(last, str):
            new_content[-1] = last + text
            return new_content
        if isinstance(last, dict):
            if last.get("type") == "text" and isinstance(last.get("text"), str):
                updated_last = dict(last)
                updated_last["text"] = updated_last["text"] + text
                new_content[-1] = updated_last
                return new_content
        new_content.append({"type": "text", "text": text})
        return new_content
    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            updated = dict(content)
            updated["text"] = updated["text"] + text
            return updated
    return str(content) + text


def streaming_response_kwargs(stream: bool) -> dict[str, Any]:
    """Return kwargs for FastAPI/Starlette StreamingResponse.

    For streaming responses, set an explicit `Content-Type: text/event-stream`
    header (without charset) to satisfy strict SSE checkers and to avoid
    Starlette's automatic `; charset=...` injection.
    """
    if stream:
        return {"headers": {"content-type": "text/event-stream"}}
    return {"media_type": "application/json"}

