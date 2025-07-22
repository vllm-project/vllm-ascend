import zlib
from unittest.mock import MagicMock

import torch

from tests.ut.base import TestBase
from vllm_ascend.distributed.kv_transfer.simple_buffer import (SimpleBuffer,
                                                               int32_hash)


class MockSimplePipe:

    def __init__(self):
        self.cluster_id = 0
        self.send_tensor = MagicMock()
        self.recv_tensor = MagicMock()
        self.deallocate_buffer = MagicMock()


class TestSimpleBuffer(TestBase):

    def setUp(self):
        self.pipe = MockSimplePipe()
        self.buffer = SimpleBuffer(self.pipe)

    def test_int32_hash(self):
        self.assertEqual(int32_hash("test"), zlib.adler32(b"test"))

    def test_insert(self):
        input_tokens = torch.tensor([1, 2, 3])
        roi = torch.tensor([1, 0, 1])
        key = torch.randn(2, 3, 4, 5)
        value = torch.randn(2, 3, 4, 5)
        hidden = torch.randn(3, 6)

        self.buffer.num_layers = 2
        self.buffer.num_heads = 4
        self.buffer.head_size = 5
        self.buffer.hidden_size = 6
        self.buffer.dtype = torch.float32

        self.buffer.insert(input_tokens, roi, key, value, hidden, "req1")

        self.pipe.send_tensor.assert_called()

    def test_drop_select(self):
        input_tokens = torch.tensor([1, 2, 3])
        roi = None

        self.buffer.num_layers = 2
        self.buffer.num_heads = 4
        self.buffer.head_size = 5
        self.buffer.hidden_size = 6
        self.buffer.dtype = torch.float32

        self.pipe.recv_tensor.side_effect = [
            (MagicMock(), torch.randn(1, 2, 3 * 4 * 5)),
            (MagicMock(), torch.randn(1, 2, 3 * 4 * 5)),
            (MagicMock(), torch.randn(1, 3, 6))
        ]

        result = self.buffer.drop_select(input_tokens, roi, "req1")
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertIsInstance(result[2], torch.Tensor)
        self.assertIsNone(result[3])
        self.assertEqual(result[0].shape, (2, 3, 4, 5))

    def test_close(self):
        self.buffer.close()
