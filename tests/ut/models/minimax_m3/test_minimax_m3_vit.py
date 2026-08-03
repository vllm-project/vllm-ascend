# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path

import vllm

from vllm_ascend.models.minimax_m3.minimax_m3_vl import (
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
    MiniMaxVLVisionModel,
)


class TestMiniMaxM3VitProcessor(unittest.TestCase):
    def test_vision_tower_uses_vllm_common_implementation(self) -> None:
        source_file = inspect.getsourcefile(MiniMaxVLVisionModel)
        assert source_file is not None
        vllm_file = vllm.__file__
        assert vllm_file is not None
        expected_source = Path(vllm_file).resolve().parent / "models" / "minimax_m3" / "common" / "vision_tower.py"
        self.assertTrue(Path(source_file).samefile(expected_source))

    def test_standalone_vllm_vision_bridge_is_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("vllm_ascend.models.minimax_m3.minimax_m3_vllm_vision"))

    def test_multimodal_processor_uses_vllm_common_implementation(self) -> None:
        self.assertEqual(
            inspect.getsourcefile(MiniMaxM3VLProcessingInfo),
            inspect.getsourcefile(MiniMaxM3VLDummyInputsBuilder),
        )
        self.assertEqual(
            inspect.getsourcefile(MiniMaxM3VLProcessingInfo),
            inspect.getsourcefile(MiniMaxM3VLMultiModalProcessor),
        )
