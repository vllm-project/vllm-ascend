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

import numpy as np
import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310


class MockCpuGpuBuffer:
    def __init__(self, max_size, dtype, device="cpu"):
        self.max_size = max_size
        self.dtype = dtype
        self.device = device
        self.cpu = torch.zeros(max_size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(max_size, dtype=dtype, device=device)

    def copy_to_gpu(self, size=None):
        if size is None:
            size = self.max_size
        self.gpu[:size].copy_(self.cpu[:size])


class MockCachedRequestState:
    def __init__(self, req_id, token_ids):
        self.req_id = req_id
        self.token_ids = token_ids

    def get_token_id(self, position):
        if position < len(self.token_ids):
            return self.token_ids[position]
        return 0


class MockInputBatch:
    def __init__(self, num_reqs, req_ids, vocab_size, num_tokens_no_spec=None):
        self.num_reqs = num_reqs
        self.req_ids = req_ids
        self.vocab_size = vocab_size
        if num_tokens_no_spec is None:
            self.num_tokens_no_spec = np.array([i + 11 for i in range(num_reqs)], dtype=np.int64)
        else:
            self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


class TestPrepareNextTokenIdsPadded310(TestBase):
    """Regression tests for 310P prepare_next_token_ids_padded override."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.proposer = AscendSpecDecodeBaseProposer310.__new__(AscendSpecDecodeBaseProposer310)
        self.proposer.backup_next_token_ids = MockCpuGpuBuffer(
            max_size=32,
            dtype=torch.int64,
            device=self.device,
        )

    def test_empty_discard_indices_skips_index_fill(self):
        """310P must not call index_fill_ when there are no discarded requests."""
        num_reqs = 3
        sampled_token_ids = torch.tensor(
            [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )
        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(10))),
            "req_1": MockCachedRequestState("req_1", list(range(15))),
            "req_2": MockCachedRequestState("req_2", list(range(20))),
        }
        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=1000,
            num_tokens_no_spec=[11, 16, 21],
        )

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=torch.tensor([], dtype=torch.int64),
            num_discarded_requests=0,
        )

        self.assertTrue(torch.equal(valid_sampled_tokens_count, torch.tensor([5, 5, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(next_token_ids, torch.tensor([104, 204, 304], dtype=torch.int64)))

    def test_discarded_requests_use_backup_tokens(self):
        """Discarded requests should mask sampled tokens and fall back to backup."""
        num_reqs = 3
        sampled_token_ids = torch.tensor(
            [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )
        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
            "req_2": MockCachedRequestState("req_2", list(range(25))),
        }
        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=1000,
            num_tokens_no_spec=[11, 21, 21],
        )

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=torch.tensor([0, 2], dtype=torch.int64),
            num_discarded_requests=2,
        )

        expected_backup_token_0 = requests["req_0"].get_token_id(10)
        expected_backup_token_2 = requests["req_2"].get_token_id(20)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, torch.tensor([0, 5, 0], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(
                next_token_ids,
                torch.tensor([expected_backup_token_0, 204, expected_backup_token_2], dtype=torch.int64),
            )
        )

    def test_rejected_tokens_without_discard(self):
        """Rejected (-1) tokens should still pick the last valid sampled token."""
        num_reqs = 3
        sampled_token_ids = torch.tensor(
            [
                [100, 101, -1, -1, -1],
                [200, 201, 202, 203, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=torch.int64,
        )
        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
            "req_2": MockCachedRequestState("req_2", list(range(25))),
        }
        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=1000,
            num_tokens_no_spec=[11, 16, 26],
        )

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=torch.tensor([], dtype=torch.int64),
            num_discarded_requests=0,
        )

        expected_backup_token_2 = requests["req_2"].get_token_id(25)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, torch.tensor([2, 4, 0], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(
                next_token_ids,
                torch.tensor([101, 203, expected_backup_token_2], dtype=torch.int64),
            )
        )
