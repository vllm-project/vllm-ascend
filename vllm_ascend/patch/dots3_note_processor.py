# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve raw NOTE video and question metadata through vLLM processing."""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import vllm
from vllm.multimodal.parse import MultiModalDataParser
from vllm.utils.import_utils import import_from_path

from vllm_ascend.patch.dots3_note_video import decode_audio, decode_frames


def _load_vllm_dots3_note_common_module():
    if vllm.__file__ is None:
        raise ImportError("Unable to locate the installed vLLM package")
    common_path = Path(vllm.__file__).resolve().parent / "models" / "dots3_note" / "common"
    if not common_path.is_dir():
        raise ImportError(f"The vLLM Dots3 Note common source was not found at {common_path}")
    package_name = "vllm_ascend.models.dots3_note._vllm_dots3_note_common"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(common_path)]
        sys.modules[package_name] = package
    processor_name = f"{package_name}.processor"
    if processor_name not in sys.modules:
        import_from_path(processor_name, common_path / "processor.py")
    return sys.modules[processor_name]


_common = _load_vllm_dots3_note_common_module()
_video: Any = sys.modules[f"{_common.__package__}.video"]
_video._decode_audio = decode_audio
_video._decode_frames = partial(decode_frames, _video)
AUDIO_END = _common.AUDIO_END
AUDIO_PAD = _common.AUDIO_PAD
AUDIO_START = _common.AUDIO_START
IMAGE_END = _common.IMAGE_END
IMAGE_PAD = _common.IMAGE_PAD
IMAGE_START = _common.IMAGE_START
VIDEO_PLACEHOLDER = _common.VIDEO_PLACEHOLDER
Dots3NoteDummyInputsBuilder = _common.Dots3NoteDummyInputsBuilder
load_note_config_section = _common.load_note_config_section


class Dots3NoteDataParser(MultiModalDataParser):
    def _get_video_with_metadata(self, video):
        frames, metadata = super()._get_video_with_metadata(video)
        return frames, metadata if metadata is not None else {"fps": 1.0}


# The upstream classes are loaded from a synthetic package to avoid CUDA imports.
class Dots3NoteProcessingInfo(_common.Dots3NoteProcessingInfo):  # type: ignore[name-defined]
    def get_data_parser(self):
        sample_rate = int((self.audio_config or {}).get("sampling_rate", 16000))
        return Dots3NoteDataParser(target_sr=float(sample_rate), target_channels=1, video_needs_metadata=True)


class Dots3NoteMultiModalProcessor(_common.Dots3NoteMultiModalProcessor):  # type: ignore[name-defined]
    def _get_hf_mm_data(self, mm_items):
        data, passthrough = super()._get_hf_mm_data(mm_items)
        if "video" in mm_items:
            videos = mm_items["video"]
            raw = []
            for index in range(videos.get_count()):
                item = videos.get(index)
                if isinstance(item, tuple) and "original_video_bytes" in item[1]:
                    item = item[1]["original_video_bytes"]
                raw.append(item)
            data = dict(data, videos=raw)
        return data, passthrough

    def apply(self, inputs, timing_ctx):
        if "video" not in inputs.mm_data_items:
            return super().apply(inputs, timing_ctx)

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        prompt = inputs.prompt
        marker = vocab[VIDEO_PLACEHOLDER]
        user_start = vocab["<|user|>"]
        marker_pos = prompt.index(marker)
        before_video = prompt[:marker_pos]
        if user_start not in before_video:
            return super().apply(inputs, timing_ctx)

        start = len(before_video) - before_video[::-1].index(user_start)
        end = prompt.index(vocab["<|endofuser|>"], marker_pos)
        template_markers = {marker, vocab.get("<no_think>"), vocab.get("<think>")}
        question_ids = [token for token in prompt[start:end] if token not in template_markers]
        question = tokenizer.decode(question_ids).strip()
        kwargs = dict(inputs.hf_processor_mm_kwargs)
        kwargs["video_question"] = question
        return super().apply(replace(inputs, hf_processor_mm_kwargs=kwargs), timing_ctx)
