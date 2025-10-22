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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_minicpm import forward


class TestPatchMiniCPM(TestBase):

    def setUp(self):
        self.mock_self = MagicMock()

        self.mock_self.q_size = 128
        self.mock_self.kv_size = 128

        self.mock_self.qkv_proj = MagicMock()
        self.mock_self.rotary_emb = MagicMock()
        self.mock_self.attn = MagicMock()
        self.mock_self.o_proj = MagicMock()

        self.positions = torch.tensor([1, 2, 3])
        self.hidden_states = torch.randn(3, 256)

        self.mock_qkv = torch.randn(3, 384)
        self.mock_q = self.mock_qkv[:, :128]
        self.mock_k = self.mock_qkv[:, 128:256]
        self.mock_v = self.mock_qkv[:, 256:]

        self.mock_self.qkv_proj.return_value = (self.mock_qkv, None)
        self.mock_self.rotary_emb.return_value = (self.mock_q, self.mock_k)
        self.mock_self.attn.return_value = torch.randn(3, 256)
        self.mock_self.o_proj.return_value = (torch.randn(3, 256), None)

    def test_forward_patched(self):
        from vllm.model_executor.models.minicpm import MiniCPMAttention

        self.assertIs(MiniCPMAttention.forward, forward)

    def test_forward_function(self):
        result = forward(self.mock_self, self.positions, self.hidden_states)

        self.mock_self.qkv_proj.assert_called_once_with(self.hidden_states)

        args, _ = self.mock_self.rotary_emb.call_args
        self.assertEqual(len(args), 3)
        self.assertTrue(torch.equal(args[0], self.positions))
        self.assertTrue(torch.equal(args[1], self.mock_q))
        self.assertTrue(torch.equal(args[2], self.mock_k))

        args, _ = self.mock_self.attn.call_args
        self.assertEqual(len(args), 3)
        self.assertTrue(torch.equal(args[0], self.mock_q))
        self.assertTrue(torch.equal(args[1], self.mock_k))
        self.assertTrue(torch.equal(args[2], self.mock_v))

        self.mock_self.o_proj.assert_called_once_with(
            self.mock_self.attn.return_value)

        self.assertEqual(result.shape, (3, 256))
        self.assertTrue(
            torch.equal(result, self.mock_self.o_proj.return_value[0]))
