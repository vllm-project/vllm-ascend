# SPDX-License-Identifier: Apache-2.0

from typing import Any


def append_generated_text(origin_prompt: Any, generated_text: str) -> Any:
    """Append partial output without dropping multimodal content parts."""
    if isinstance(origin_prompt, list):
        text_part = [{"type": "text", "text": generated_text}]
        return origin_prompt + (text_part if generated_text else [])
    return (origin_prompt or "") + generated_text
