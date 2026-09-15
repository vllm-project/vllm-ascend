# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.inputs import ProcessorInputs

from vllm_ascend.patch.dots3_note_processor import (
    VIDEO_PLACEHOLDER,
    Dots3NoteDataParser,
    Dots3NoteMultiModalProcessor,
    _common,
)


def test_video_prompt_keeps_official_order_and_question_cache_key(monkeypatch):
    vocab = {"<|user|>": 1, "<|endofuser|>": 2, VIDEO_PLACEHOLDER: 3, "<no_think>": 5}
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.return_value = "question\n"
    processor = object.__new__(Dots3NoteMultiModalProcessor)
    processor.info = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    captured = {}

    def apply(self, inputs, timing_ctx):
        captured["inputs"] = inputs
        # Simulate upstream expansion of the video placeholder.
        return {"prompt_token_ids": [1, 9, 10, 100, 101, 5, 2, 4], "mm_placeholders": {"video": ["unchanged"]}}

    monkeypatch.setattr(_common.Dots3NoteMultiModalProcessor, "apply", apply)
    # Token 5 represents the no_think suffix inserted by the chat template.
    inputs = ProcessorInputs(prompt=[1, 9, 10, 3, 5, 2, 4], mm_data_items={"video": []})
    result = processor.apply(inputs, None)

    assert inputs.prompt == [1, 9, 10, 3, 5, 2, 4]
    assert captured["inputs"].prompt == inputs.prompt
    assert captured["inputs"].hf_processor_mm_kwargs["video_question"] == "question"
    assert result["prompt_token_ids"] == [1, 9, 10, 100, 101, 5, 2, 4]
    assert result["mm_placeholders"] == {"video": ["unchanged"]}


def test_nonvideo_processing_is_unchanged(monkeypatch):
    original = Mock(return_value={"prompt_token_ids": [1, 2]})
    monkeypatch.setattr(_common.Dots3NoteMultiModalProcessor, "apply", original)
    processor = object.__new__(Dots3NoteMultiModalProcessor)
    inputs = ProcessorInputs(prompt=[1, 2], mm_data_items={"image": []})
    assert processor.apply(inputs, None) == {"prompt_token_ids": [1, 2]}
    original.assert_called_once_with(inputs, None)


def test_raw_video_survives_cache_item_reparsing():
    parser = MultiModalDataParser(video_needs_metadata=True)
    frames = np.zeros((1, 28, 28, 3), dtype=np.uint8)
    metadata = {"original_video_bytes": b"original-video", "fps": 1.0}
    initial = parser.parse_mm_data({"video": [(frames, metadata)]})
    # The vLLM cache reparses unwrapped missing items through this path.
    missing = parser.parse_mm_data({"video": [initial["video"][0]]})
    processor = object.__new__(Dots3NoteMultiModalProcessor)
    data, _ = processor._get_hf_mm_data(missing)
    assert data["videos"] == [b"original-video"]


def test_decoded_video_without_metadata_remains_supported():
    parser = Dots3NoteDataParser(video_needs_metadata=True)
    frames = np.zeros((4, 28, 28, 3), dtype=np.uint8)
    items = parser.parse_mm_data({"video": [frames]})
    result_frames, metadata = items["video"][0]
    assert result_frames is frames
    assert metadata == {"fps": 1.0}
