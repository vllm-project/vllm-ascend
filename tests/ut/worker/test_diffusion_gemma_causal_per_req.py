#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import unittest

import torch

from tests.ut.base import TestBase


def select_allow_mask(
    causal_per_req: torch.Tensor,
    causal_allow: torch.Tensor,
    bidir_allow: torch.Tensor,
) -> torch.Tensor:
    cflag = causal_per_req.reshape(-1, 1, 1)
    return torch.where(cflag, causal_allow, bidir_allow)


class TestDiffusionGemmaCausalPerReq(TestBase):
    """Regression tests for mixed DiffusionGemma canvas batches."""

    def test_metadata_carries_per_request_causal_tensor(self):
        from vllm_ascend.attention.attention_v1 import AscendMetadata

        causal_per_req = torch.tensor([True, False])
        metadata = AscendMetadata(causal_per_req=causal_per_req)

        self.assertIs(metadata.causal_per_req, causal_per_req)
        self.assertEqual(metadata.causal_per_req.dtype, torch.bool)
        self.assertEqual(list(metadata.causal_per_req.shape), [2])
        self.assertNotEqual(
            bool(metadata.causal_per_req[0]),
            bool(metadata.causal_per_req[1]),
        )

    def test_branchless_select_preserves_mixed_phase(self):
        batch, query_len, max_seq_len = 2, 3, 3
        causal_allow = torch.tril(torch.ones(query_len, max_seq_len, dtype=torch.bool))
        causal_allow = causal_allow.unsqueeze(0).expand(batch, -1, -1).contiguous()
        bidir_allow = torch.ones(batch, query_len, max_seq_len, dtype=torch.bool)

        allow = select_allow_mask(
            torch.tensor([True, False]),
            causal_allow,
            bidir_allow,
        )

        self.assertTrue(torch.equal(allow[0], causal_allow[0]))
        self.assertTrue(torch.equal(allow[1], bidir_allow[1]))
        self.assertFalse(bool(allow[0, 0, -1]))
        self.assertTrue(bool(allow[1, 0, -1]))

    def test_branchless_select_all_causal_matches_scalar_case(self):
        batch, query_len, max_seq_len = 3, 4, 4
        causal_allow = torch.tril(torch.ones(query_len, max_seq_len, dtype=torch.bool))
        causal_allow = causal_allow.unsqueeze(0).expand(batch, -1, -1).contiguous()
        bidir_allow = torch.ones(batch, query_len, max_seq_len, dtype=torch.bool)

        allow = select_allow_mask(
            torch.ones(batch, dtype=torch.bool),
            causal_allow,
            bidir_allow,
        )

        self.assertTrue(torch.equal(allow, causal_allow))


if __name__ == "__main__":
    unittest.main()
