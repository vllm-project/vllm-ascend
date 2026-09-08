# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import vllm
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer

from vllm_ascend.models.minimax_m3 import minimax_m3_vl as m3_vl
from vllm_ascend.models.minimax_m3.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
    MiniMaxVLVisionModel,
)


class _DummyMiniMaxM3MultimodalConfig:
    mm_encoder_only = False
    mm_encoder_tp_mode = "weights"

    def __init__(self, limit_mm_per_prompt: dict[str, int]) -> None:
        self.limit_mm_per_prompt = limit_mm_per_prompt

    def get_limit_per_prompt(self, modality: str) -> int:
        return self.limit_mm_per_prompt.get(modality, 999)


class _DummyVisionTower(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()


class _DummyLanguageModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.lm_head = nn.Identity()
        self.expert_weights: list[list[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[nn.Module] = []
        self.num_moe_layers = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_shared_experts = 0
        self.num_redundant_experts = 0

    def make_empty_intermediate_tensors(self, *args, **kwargs):
        return None


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

    def test_shared_vision_tower_pruning_follows_image_video_limits(self) -> None:
        def make_vllm_config(limit_mm_per_prompt: dict[str, int]):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(vision_config=SimpleNamespace()),
                    hf_text_config=SimpleNamespace(hidden_size=1),
                    multimodal_config=_DummyMiniMaxM3MultimodalConfig(limit_mm_per_prompt),
                ),
                quant_config=None,
            )

        test_cases = [
            ({"image": 1, "video": 1}, _DummyVisionTower),
            ({"image": 1, "video": 0}, _DummyVisionTower),
            ({"image": 0, "video": 1}, _DummyVisionTower),
            ({"image": 0, "video": 0}, StageMissingLayer),
        ]

        with (
            patch.object(m3_vl, "MiniMaxVLVisionModel", _DummyVisionTower),
            patch.object(m3_vl, "MiniMaxM3SparseForCausalLM", _DummyLanguageModel),
        ):
            for limit_mm_per_prompt, expected_tower_type in test_cases:
                with self.subTest(limit_mm_per_prompt=limit_mm_per_prompt):
                    model = MiniMaxM3SparseForConditionalGeneration(vllm_config=make_vllm_config(limit_mm_per_prompt))

                    self.assertIsInstance(model.model.vision_tower, expected_tower_type)
                    self.assertIsInstance(model.language_model, _DummyLanguageModel)
