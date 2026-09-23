# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer


class FakeCommunicator:
    def __init__(self):
        self.sends = []
        self.recvs = []
        self.executed = False

    def set_transfer_context(self, old, layer_idx):
        self.context = old, layer_idx

    def add_send(self, tensors, rank, expert):
        self.sends.append((tensors, rank, expert))

    def add_recv(self, tensors, rank, expert):
        self.recvs.append((tensors, rank, expert))

    def execute(self):
        self.executed = True


class TestExplicitTransfer(unittest.TestCase):
    def test_transfer_accepts_independent_expert_tensors(self):
        communicator = FakeCommunicator()
        weights = [[torch.tensor([10.0]), torch.tensor([11.0])]]
        buffers = [[torch.zeros(1), torch.zeros(1)]]

        stage_explicit_layer_transfer(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 2, 1, 3]),
            np.array([[0, 1], [0, 1]]),
            np.array([[0, 0], [1, 1]]),
            weights,
            buffers,
            SimpleNamespace(size=lambda: 2, rank=lambda: 0),
            communicator,
        )

        self.assertIs(communicator.sends[0][0][0], weights[0][1])
        self.assertIs(communicator.recvs[0][0][0], buffers[0][1])

    def test_transfer_rejects_mismatched_independent_expert_tensors(self):
        communicator = FakeCommunicator()
        with self.assertRaisesRegex(ValueError, "slot-aligned schema"):
            stage_explicit_layer_transfer(
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([0, 2, 1, 3]),
                np.array([[0, 1], [0, 1]]),
                np.array([[0, 0], [1, 1]]),
                [[torch.zeros(1), torch.zeros(2)]],
                [[torch.zeros(1), torch.zeros(3)]],
                SimpleNamespace(size=lambda: 2, rank=lambda: 0),
                communicator,
            )

        self.assertFalse(hasattr(communicator, "context"))
        self.assertFalse(communicator.sends)
        self.assertFalse(communicator.recvs)

    def test_transfer_uses_exact_planned_sources(self):
        communicator = FakeCommunicator()
        old = torch.tensor([0, 1, 2, 3])
        new = torch.tensor([0, 2, 1, 3])
        weights = [torch.tensor([[10.0], [11.0]]), torch.tensor([[20.0, 21.0], [22.0, 23.0]])]
        buffers = [torch.zeros_like(weight) for weight in weights]
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

        metadata = stage_explicit_layer_transfer(
            old,
            new,
            np.array([[0, 1], [0, 1]]),
            np.array([[0, 0], [1, 1]]),
            weights,
            buffers,
            group,
            communicator,
        )

        self.assertEqual([(rank, expert) for _, rank, expert in communicator.sends], [(1, 1)])
        self.assertEqual([(rank, expert) for _, rank, expert in communicator.recvs], [(1, 2)])
        torch.testing.assert_close(communicator.sends[0][0][0], weights[0][1])
        torch.testing.assert_close(communicator.sends[0][0][1], weights[1][1])
        np.testing.assert_array_equal(metadata.is_unchanged, [True, False])
        np.testing.assert_array_equal(metadata.is_received_locally, [True, False])
        np.testing.assert_array_equal(metadata.recv_primary_mask, [False, True])
        self.assertEqual(metadata.recv_count, 1)
        np.testing.assert_array_equal(metadata.recv_expert_ids, [2, -1])
        np.testing.assert_array_equal(metadata.recv_dst_rows, [1, -1])
        self.assertTrue(communicator.executed)

    def test_transfer_rejects_false_source_before_communication(self):
        communicator = FakeCommunicator()
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

        with self.assertRaisesRegex(RuntimeError, "does not own"):
            stage_explicit_layer_transfer(
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([0, 2, 1, 3]),
                np.array([[0, 1], [0, 1]]),
                np.array([[0, 1], [1, 1]]),
                [torch.ones((2, 1))],
                [torch.zeros((2, 1))],
                group,
                communicator,
            )

        self.assertFalse(communicator.sends)
        self.assertFalse(communicator.recvs)
        self.assertFalse(communicator.executed)

    def test_transfer_rejects_buffer_schema_before_side_effects(self):
        communicator = FakeCommunicator()
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

        with self.assertRaisesRegex(ValueError, "slot-aligned schema"):
            stage_explicit_layer_transfer(
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([0, 2, 1, 3]),
                np.array([[0, 1], [0, 1]]),
                np.array([[0, 0], [1, 1]]),
                [torch.ones((2, 1)), torch.ones((2, 2))],
                [torch.zeros((2, 1)), torch.zeros((2, 3))],
                group,
                communicator,
            )

        self.assertFalse(hasattr(communicator, "context"))
        self.assertFalse(communicator.sends)
        self.assertFalse(communicator.recvs)
        self.assertFalse(communicator.executed)


if __name__ == "__main__":
    unittest.main()
