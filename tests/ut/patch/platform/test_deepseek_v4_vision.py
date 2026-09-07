# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.entrypoints import chat_utils
from vllm.entrypoints.chat_utils import MODALITY_PLACEHOLDERS_MAP

from vllm_ascend.patch.platform.patch_deepseek_v4_vision import (
    _make_multimodal_parser_patch,
)


def _tracker(*, vision: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4" if vision else "qwen3",
                vision_n_layers=32 if vision else 0,
            )
        )
    )


class _ImageContentParser:
    def __init__(self, model_config):
        self.model_config = model_config
        self._images = []

    def parse_image(self, image_url, uuid=None):
        del image_url, uuid
        self._images.append("<｜deepseek_image｜>")

    def mm_placeholder_storage(self):
        return {MODALITY_PLACEHOLDERS_MAP["image"]: self._images}


class _ImageTracker:
    def __init__(self):
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                vision_n_layers=32,
            ),
            enable_prompt_embeds=False,
        )

    def create_parser(self, mm_processor_kwargs=None):
        del mm_processor_kwargs
        return _ImageContentParser(self.model_config)


def test_vision_parser_preserves_content_order_and_separator():
    captured: dict[str, object] = {}

    def original(
        role,
        parts,
        mm_tracker,
        *,
        wrap_dicts,
        interleave_strings,
        mm_processor_kwargs=None,
        multimodal_content_part_separator="\n",
    ):
        captured.update(
            role=role,
            parts=parts,
            mm_tracker=mm_tracker,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
            mm_processor_kwargs=mm_processor_kwargs,
            multimodal_content_part_separator=multimodal_content_part_separator,
        )
        return ["parsed"]

    patched = _make_multimodal_parser_patch(original)
    result = patched(
        "user",
        ["before", {"type": "image_url"}, "after"],
        _tracker(vision=True),
        wrap_dicts=False,
        interleave_strings=False,
    )

    assert result == ["parsed"]
    assert captured["interleave_strings"] is True
    assert captured["multimodal_content_part_separator"] == "\n\n"

    parsed = chat_utils._parse_chat_message_content_parts(
        "user",
        [
            {"type": "text", "text": "before"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {"type": "text", "text": "after"},
        ],
        _ImageTracker(),
        wrap_dicts=False,
        interleave_strings=False,
    )
    assert parsed == [
        {
            "role": "user",
            "content": "before\n\n<｜deepseek_image｜>\n\nafter",
        }
    ]

    parsed = chat_utils._parse_chat_message_content_parts(
        "user",
        [
            {"type": "text", "text": "before\ninside"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {"type": "text", "text": "after"},
        ],
        _ImageTracker(),
        wrap_dicts=False,
        interleave_strings=False,
    )
    assert parsed[0]["content"] == "before\ninside\n\n<｜deepseek_image｜>\n\nafter"


def test_non_vision_parser_arguments_are_unchanged():
    captured: dict[str, object] = {}

    def original(role, parts, mm_tracker, *, wrap_dicts, interleave_strings):
        captured.update(
            role=role,
            parts=parts,
            mm_tracker=mm_tracker,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
        )
        return []

    patched = _make_multimodal_parser_patch(original)
    patched(
        "user",
        ["text"],
        _tracker(vision=False),
        wrap_dicts=True,
        interleave_strings=False,
    )

    assert captured["interleave_strings"] is False
    assert captured["wrap_dicts"] is True

    prompt = chat_utils._get_full_multimodal_text_prompt(
        {"<image>": ["<image>"]},
        ["before", "<image>", "after"],
        interleave_strings=True,
    )
    assert prompt == "before\n<image>\nafter"
