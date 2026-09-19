# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep raw video/audio and generation budgets in the Dots3 Note renderer."""

_REGISTERED = False


def register_dots3_note_video():
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig
    from vllm.renderers.online_renderer import OnlineRenderer

    class Dots3NoteConfig(VerifyAndUpdateConfig):
        @staticmethod
        def verify_and_update_model_config(model_config):
            mm_config = model_config.multimodal_config
            if mm_config is not None:
                video_kwargs = mm_config.media_io_kwargs.setdefault("video", {})
                video_kwargs.setdefault("num_frames", 1)
                video_kwargs.setdefault("video_backend", "nemotron_vl")

    MODELS_CONFIG_MAP["Dots3NoteForCausalLM"] = Dots3NoteConfig
    original = OnlineRenderer.preprocess_chat

    async def preprocess_chat(self, request, messages, *args, **kwargs):
        if "Dots3NoteForCausalLM" in self.model_config.architectures:
            mm_kwargs = dict(request.mm_processor_kwargs or {})
            max_tokens = getattr(request, "max_completion_tokens", None)
            if max_tokens is None:
                max_tokens = getattr(request, "max_tokens", None)
            if max_tokens is not None:
                mm_kwargs["max_new_tokens"] = max_tokens
            media_kwargs = dict(request.media_io_kwargs or {})
            media_kwargs["video"] = dict(media_kwargs.get("video") or {}, video_backend="nemotron_vl")
            request = request.model_copy(update={"mm_processor_kwargs": mm_kwargs, "media_io_kwargs": media_kwargs})
        return await original(self, request, messages, *args, **kwargs)

    OnlineRenderer.preprocess_chat = preprocess_chat
    _REGISTERED = True
