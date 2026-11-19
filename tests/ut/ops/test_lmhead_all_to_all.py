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

import unittest
from unittest import mock

import torch

from vllm_ascend.ops.vocab_parallel_embedding import lmhead_all_to_all


def _make_comm_group(world_size):
    group = mock.MagicMock()
    group.world_size = world_size
    group.device_group = mock.MagicMock()
    return group


class _FakeAllToAllSingle:
    """Fake dist.all_to_all_single with real equal-split send/recv semantics.

    all_to_all_single splits dim 0 into ``world_size`` equal chunks, routes
    chunk ``j`` to rank ``j``, and rank ``r`` concatenates the received
    chunk ``r`` from every rank along dim 0 in rank order. The fake assumes
    invocations happen in rank order 0..world_size-1 (call order = rank
    order) and fills the output buffer with what that rank would actually
    receive. An identity copy (out.copy_(inp)) would not exercise the
    helper's reordering at all, so it must not be used here.
    """

    def __init__(self, full: torch.Tensor, world_size: int):
        self.full = full
        self.world_size = world_size
        self.call_count = 0

    def __call__(self, output, input_, group=None, **kwargs):
        rank = self.call_count
        self.call_count += 1
        n, v = self.full.shape
        np_, vp = n // self.world_size, v // self.world_size
        received = torch.cat(
            [self.full[:, i * vp : (i + 1) * vp].view(self.world_size, np_, vp)[rank] for i in range(self.world_size)],
            dim=0,
        )
        # `received` is [P*N/P, V/P]; the output buffer is [P, N/P, V/P],
        # so reshape before copy_.
        output.copy_(received.view(self.world_size, np_, vp))


class TestLmheadAllToAll(unittest.TestCase):
    def test_redistributes_to_expected_np_v(self):
        cases = [
            {"world_size": 2, "n": 4, "v": 4},
            {"world_size": 4, "n": 16, "v": 8},
        ]
        for case in cases:
            with self.subTest(**case):
                self._assert_redistribution(**case)

    def _assert_redistribution(self, world_size, n, v):
        # Unique values so every element can be asserted individually.
        full = torch.arange(n * v).reshape(n, v)
        vp = v // world_size
        np_ = n // world_size
        # Per-rank local input: all tokens x this rank's vocab shard.
        inputs = [full[:, r * vp : (r + 1) * vp].clone() for r in range(world_size)]
        group = _make_comm_group(world_size)
        fake = _FakeAllToAllSingle(full, world_size)
        with mock.patch("vllm_ascend.ops.vocab_parallel_embedding.dist.all_to_all_single", fake):
            for r in range(world_size):
                output = lmhead_all_to_all(inputs[r], group)
                expected = full[r * np_ : (r + 1) * np_, :]
                self.assertEqual(output.shape, expected.shape)
                self.assertTrue(torch.equal(output, expected), f"rank {r} mismatch")
        # Every rank's call must have reached the fake; otherwise the test
        # could degenerate into not exercising the helper internals.
        self.assertEqual(fake.call_count, world_size)

    def test_world_size_one_returns_input(self):
        logits = torch.arange(12).reshape(3, 4)
        group = _make_comm_group(world_size=1)
        with mock.patch("vllm_ascend.ops.vocab_parallel_embedding.dist.all_to_all_single") as fake:
            output = lmhead_all_to_all(logits, group)
        self.assertIs(output, logits)
        fake.assert_not_called()

    def test_raises_on_uneven_tokens(self):
        # N=3 is not divisible by world_size=2.
        logits = torch.arange(12).reshape(3, 4)
        group = _make_comm_group(world_size=2)
        with (
            mock.patch("vllm_ascend.ops.vocab_parallel_embedding.dist.all_to_all_single"),
            self.assertRaises(ValueError) as ctx,
        ):
            lmhead_all_to_all(logits, group)
        self.assertIn("divisible", str(ctx.exception))
