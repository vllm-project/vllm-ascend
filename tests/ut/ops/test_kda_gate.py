# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import unittest

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.triton.kda.gate import DEFAULT_KDA_LOWER_BOUND, apply_kda_gate


class TestApplyKdaGate(TestBase):
    def test_safe_gate_matches_glm53_formula(self):
        raw_g = torch.randn(1, 4, 2, 8)
        a_log = torch.randn(1, 1, 2, 1)
        g_bias = torch.randn(2 * 8)
        out = apply_kda_gate(raw_g, a_log, g_bias, safe_gate=True, lower_bound=DEFAULT_KDA_LOWER_BOUND)
        bias = g_bias.reshape(2, 8)
        expected = DEFAULT_KDA_LOWER_BOUND * torch.sigmoid(torch.exp(a_log.float()) * (raw_g.float() + bias))
        torch.testing.assert_close(out, expected.to(torch.float32), rtol=1e-5, atol=1e-5)
        self.assertEqual(out.shape, raw_g.shape)

    def test_softplus_gate_matches_kimi_formula(self):
        raw_g = torch.randn(1, 3, 2, 4)
        a_log = torch.tensor([0.2, -0.1]).view(1, 1, 2, 1)
        out = apply_kda_gate(raw_g, a_log, None, safe_gate=False)
        expected = (-torch.exp(a_log.float())) * torch.nn.functional.softplus(raw_g.float())
        torch.testing.assert_close(out, expected.to(torch.float32), rtol=1e-5, atol=1e-5)

    def test_head_mismatch_raises(self):
        raw_g = torch.randn(1, 2, 4, 8)
        a_log = torch.randn(2)
        with self.assertRaisesRegex(ValueError, "head dim mismatch"):
            apply_kda_gate(raw_g, a_log, None, safe_gate=True)


if __name__ == "__main__":
    unittest.main()
