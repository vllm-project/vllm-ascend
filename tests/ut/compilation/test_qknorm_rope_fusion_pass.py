#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from unittest.mock import MagicMock, patch

import torch
from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

PASS_MODULE = "vllm_ascend.compilation.passes.qknorm_rope_fusion_pass"


def _vllm_config(dtype: torch.dtype = torch.bfloat16) -> VllmConfig:
    vllm_config = VllmConfig()
    vllm_config.model_config = MagicMock(dtype=dtype)
    return vllm_config


class TestQKNormRopeFusionPassGating(TestBase):
    """The pass must not build patterns it cannot trace.

    Registering a pattern allocates NPU example inputs and traces the rope
    kernel, so these tests assert on whether the pattern classes get
    instantiated rather than on the registered patterns themselves.
    """

    def setUp(self):
        layer = MagicMock(head_size=128, num_heads=32, num_kv_heads=8)
        patcher = patch(f"{PASS_MODULE}.get_layers_from_vllm_config", return_value={"layer": layer})
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch(f"{PASS_MODULE}.QKNormRopeFusionPatternWithBias")
    @patch(f"{PASS_MODULE}.QKNormRopeFusionPattern")
    @patch(f"{PASS_MODULE}.get_rope_dim", return_value=0)
    def test_model_without_rope_registers_nothing(self, _rope_dim, pattern, pattern_with_bias):
        # GLM-5.3-Flash reports qk_rope_head_dim == 0, so tracing the pattern
        # would ask Triton for a zero-width arange.
        QKNormRopeFusionPass(_vllm_config())

        pattern.assert_not_called()
        pattern_with_bias.assert_not_called()

    @patch(f"{PASS_MODULE}.QKNormRopeFusionPatternWithBias")
    @patch(f"{PASS_MODULE}.QKNormRopeFusionPattern")
    @patch(f"{PASS_MODULE}.get_rope_dim", return_value=128)
    def test_model_with_rope_registers_both_patterns(self, _rope_dim, pattern, pattern_with_bias):
        QKNormRopeFusionPass(_vllm_config())

        # One registration per supported epsilon.
        self.assertEqual(pattern.call_count, 2)
        self.assertEqual(pattern_with_bias.call_count, 2)

    @patch(f"{PASS_MODULE}.QKNormRopeFusionPatternWithBias")
    @patch(f"{PASS_MODULE}.QKNormRopeFusionPattern")
    @patch(f"{PASS_MODULE}.get_rope_dim", return_value=128)
    def test_unsupported_dtype_registers_nothing(self, _rope_dim, pattern, pattern_with_bias):
        QKNormRopeFusionPass(_vllm_config(dtype=torch.float16))

        pattern.assert_not_called()
        pattern_with_bias.assert_not_called()
