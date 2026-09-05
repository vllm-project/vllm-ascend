import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.worker.utils import copy_snapshot_to_gpu


class TestQueryStartLocCopy(unittest.TestCase):
    def test_copy_uses_stable_cpu_snapshot(self):
        class DeferredCopy:
            def copy_(self, source, non_blocking=False):
                self.source = source
                self.non_blocking = non_blocking
                return self

        cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
        gpu = DeferredCopy()
        query_start_loc = SimpleNamespace(cpu=cpu, gpu=gpu)

        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            copy_snapshot_to_gpu(query_start_loc)
        cpu.fill_(99)

        self.assertEqual(gpu.source.tolist(), [0, 2, 5])
        self.assertNotEqual(gpu.source.data_ptr(), cpu.data_ptr())
        self.assertTrue(gpu.non_blocking)

    def test_copy_pins_snapshot(self):
        cpu = MagicMock()
        snapshot = MagicMock()
        pinned_snapshot = MagicMock()
        cpu.clone.return_value = snapshot
        snapshot.pin_memory.return_value = pinned_snapshot
        gpu = MagicMock()

        copy_snapshot_to_gpu(SimpleNamespace(cpu=cpu, gpu=gpu))

        snapshot.pin_memory.assert_called_once_with()
        gpu.copy_.assert_called_once_with(pinned_snapshot, non_blocking=True)
