# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from typing import Any


def _load_proxy_utils():
    path = Path(__file__).parents[2] / "examples" / "disaggregated_prefill_v1" / "proxy_utils.py"
    spec = importlib.util.spec_from_file_location("disaggregated_prefill_proxy_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_append_generated_text_preserves_multimodal_parts():
    append_generated_text = _load_proxy_utils().append_generated_text
    prompt: list[dict[str, Any]] = [
        {"type": "text", "text": "Describe the image."},
        {"type": "image_url", "image_url": {"url": "file:///tmp/image.png"}},
    ]

    result = append_generated_text(prompt, "Partial answer")

    assert result == prompt + [{"type": "text", "text": "Partial answer"}]
    assert prompt[-1]["type"] == "image_url"


def test_append_generated_text_keeps_text_prompt_behavior():
    append_generated_text = _load_proxy_utils().append_generated_text

    assert append_generated_text("Prompt: ", "answer") == "Prompt: answer"
    assert append_generated_text(None, "answer") == "answer"
