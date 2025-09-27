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
import math
import unittest

import torch

from vllm_ascend.torchair.models.qwen3_moe import CustomQwen3MoeAttention


class DummyRMSNorm:

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        denom = (mean_sq + self.eps).sqrt()
        return x / denom


class TestCustomQwen3MoeAttention(unittest.TestCase):

    def setUp(self):
        self.batch = 2
        self.seq_len = 3
        self.q_size = 8
        self.kv_size = 8
        self.head_dim = 4
        self.rms_eps = 1e-6

        total_dim = self.q_size + 2 * self.kv_size

        self.qkv = torch.arange(self.batch * self.seq_len * total_dim,
                                dtype=torch.float32).reshape(
                                    self.batch, self.seq_len, total_dim)

    def test_constant_input_normalization(self):
        ones_qkv = torch.ones((1, 1, self.q_size + 2 * self.kv_size),
                              dtype=torch.float32)

        q_norm = DummyRMSNorm(self.head_dim, self.rms_eps)
        k_norm = DummyRMSNorm(self.head_dim, self.rms_eps)
        q, k, v = CustomQwen3MoeAttention.normalize_qkv(
            ones_qkv, self.q_size, self.kv_size, self.head_dim, q_norm, k_norm)

        norm_val = 1.0 / math.sqrt(1.0 + self.rms_eps)

        expected_q = torch.full((1, 1, self.q_size), norm_val)
        expected_k = torch.full((1, 1, self.kv_size), norm_val)
        expected_v = torch.ones((1, 1, self.kv_size), dtype=torch.float32)

        self.assertTrue(torch.allclose(q, expected_q, atol=1e-6))
        self.assertTrue(torch.allclose(k, expected_k, atol=1e-6))
        self.assertTrue(torch.equal(v, expected_v))
