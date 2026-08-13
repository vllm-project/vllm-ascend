# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

DSV4_SERVED_MODEL_NAME = "dsv4"
DSV4_DEFAULT_REASONING_EFFORT = "high"
DSV4_DEFAULT_THINKING_TYPE = "enabled"


_request_normalizer = ChatCompletionRequest.__pydantic_decorators__.model_validators[
    "_normalize_messages_before"
]
_original_request_normalizer = _request_normalizer.func


def _normalize_dsv4_request_before(data: Any) -> Any:
    if isinstance(data, dict) and data.get("model") == DSV4_SERVED_MODEL_NAME:
        data.setdefault("thinking", {"type": DSV4_DEFAULT_THINKING_TYPE})
        data.setdefault("reasoning_effort", DSV4_DEFAULT_REASONING_EFFORT)
    return _original_request_normalizer(data)


_request_normalizer.func = _normalize_dsv4_request_before
ChatCompletionRequest.model_rebuild(force=True)
