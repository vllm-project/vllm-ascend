# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.renderers.online_renderer import OnlineRenderer

from vllm_ascend.patch.platform import patch_dots3_note


def test_renderer_preserves_video_bytes_and_completion_budget(monkeypatch):
    original = AsyncMock(return_value="rendered")
    monkeypatch.setattr(OnlineRenderer, "preprocess_chat", original)
    monkeypatch.setattr(patch_dots3_note, "_REGISTERED", False)
    monkeypatch.setitem(MODELS_CONFIG_MAP, "Dots3NoteForCausalLM", None)
    patch_dots3_note.register_dots3_note_video()
    request = ChatCompletionRequest(
        model="dots",
        messages=[],
        max_tokens=32,
        max_completion_tokens=64,
        media_io_kwargs={"video": {"video_backend": "opencv", "num_frames": 2}},
        mm_processor_kwargs={"seq": 1024},
    )
    renderer = SimpleNamespace(model_config=SimpleNamespace(architectures=["Dots3NoteForCausalLM"]))
    assert asyncio.run(OnlineRenderer.preprocess_chat(renderer, request, [])) == "rendered"
    forwarded = original.call_args.args[1]
    assert forwarded.mm_processor_kwargs == {"seq": 1024, "max_new_tokens": 64}
    assert forwarded.media_io_kwargs["video"] == {"video_backend": "nemotron_vl", "num_frames": 2}
    assert request.media_io_kwargs["video"]["video_backend"] == "opencv"

    original.reset_mock()
    renderer.model_config.architectures = ["OtherModel"]
    asyncio.run(OnlineRenderer.preprocess_chat(renderer, request, []))
    assert original.call_args.args[1] is request
