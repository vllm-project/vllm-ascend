# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import importlib.util
import inspect
import unittest
from dataclasses import dataclass
from pathlib import Path

import vllm
from transformers import BatchFeature

from vllm_ascend.models.minimax_m3 import (
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxVLVisionModel,
    _get_minimax_m3_num_video_tokens,
)


@dataclass
class _FakeVideoProcessor:
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2


class _FakeProcessorContext:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object], dict[str, object]]] = []

    def call_hf_processor(
        self,
        processor: object,
        data: dict[str, object],
        kwargs: dict[str, object],
    ) -> BatchFeature:
        self.calls.append((processor, data, kwargs))
        return BatchFeature(data={"input_ids": []})


class _FakeProcessingInfo:
    def __init__(self) -> None:
        self.ctx = _FakeProcessorContext()
        self.processor_kwargs: dict[str, object] | None = None

    def get_hf_processor(self, **kwargs: object) -> str:
        self.processor_kwargs = dict(kwargs)
        return "fake_processor"


def _make_processor() -> MiniMaxM3VLMultiModalProcessor:
    processor = object.__new__(MiniMaxM3VLMultiModalProcessor)
    processor.info = _FakeProcessingInfo()
    return processor


class TestMiniMaxM3VitProcessor(unittest.TestCase):
    def test_vision_tower_uses_vllm_common_implementation(self) -> None:
        source_file = inspect.getsourcefile(MiniMaxVLVisionModel)
        self.assertIsNotNone(source_file)
        self.assertIsNotNone(vllm.__file__)
        expected_source = Path(vllm.__file__).resolve().parent / "models" / "minimax_m3" / "common" / "vision_tower.py"
        self.assertTrue(Path(source_file).samefile(expected_source))

    def test_standalone_vllm_vision_bridge_is_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("vllm_ascend.models.minimax_m3_vllm_vision"))

    def test_video_token_count_pads_temporal_frames(self) -> None:
        video_processor = _FakeVideoProcessor()

        self.assertEqual(
            _get_minimax_m3_num_video_tokens(
                video_processor,
                video_width=1260,
                video_height=700,
                num_frames=32,
            ),
            18000,
        )
        self.assertEqual(
            _get_minimax_m3_num_video_tokens(
                video_processor,
                video_width=1260,
                video_height=700,
                num_frames=31,
            ),
            18000,
        )
        self.assertEqual(
            _get_minimax_m3_num_video_tokens(
                video_processor,
                video_width=28,
                video_height=28,
                num_frames=3,
            ),
            2,
        )

    def test_call_hf_processor_resizes_raw_video_by_default(self) -> None:
        processor = _make_processor()

        result = processor._call_hf_processor(
            "hello",
            {"videos": ["video"]},
            {},
            {"return_tensors": "pt"},
        )

        self.assertIsInstance(result, BatchFeature)
        self.assertEqual(processor.info.processor_kwargs, {})
        hf_processor, data, kwargs = processor.info.ctx.calls[-1]
        self.assertEqual(hf_processor, "fake_processor")
        self.assertEqual(data, {"text": "hello", "videos": ["video"]})
        self.assertIs(kwargs["do_resize"], True)
        self.assertEqual(kwargs["return_tensors"], "pt")

    def test_call_hf_processor_keeps_explicit_resize_override(self) -> None:
        processor = _make_processor()

        processor._call_hf_processor(
            "hello",
            {"videos": ["video"]},
            {"do_resize": False},
            {},
        )

        _, _, kwargs = processor.info.ctx.calls[-1]
        self.assertIs(kwargs["do_resize"], False)
