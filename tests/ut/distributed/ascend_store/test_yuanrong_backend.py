import os
import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongConfig,
    YuanrongHelper,
)


class TestYuanrongConfig(unittest.TestCase):
    @patch.dict(os.environ, {"DS_WORKER_ADDR": "localhost:8080"}, clear=False)
    def test_load_from_env_basic(self):
        config = YuanrongConfig.load_from_env()
        self.assertEqual(config.worker_addr, "localhost:8080")
        self.assertFalse(config.enable_exclusive_connection)
        self.assertFalse(config.enable_remote_h2d)

    @patch.dict(
        os.environ,
        {
            "DS_WORKER_ADDR": "host:9000",
            "DS_ENABLE_EXCLUSIVE_CONNECTION": "1",
            "DS_ENABLE_REMOTE_H2D": "1",
        },
        clear=False,
    )
    def test_load_from_env_all_flags(self):
        config = YuanrongConfig.load_from_env()
        self.assertEqual(config.worker_addr, "host:9000")
        self.assertTrue(config.enable_exclusive_connection)
        self.assertTrue(config.enable_remote_h2d)

    @patch.dict(os.environ, {}, clear=True)
    def test_load_from_env_missing_addr(self):
        # Remove DS_WORKER_ADDR if present
        os.environ.pop("DS_WORKER_ADDR", None)
        with self.assertRaises(ValueError):
            YuanrongConfig.load_from_env()


class TestYuanrongHelper(unittest.TestCase):
    def setUp(self):
        self.blob_cls = MagicMock()
        self.blob_list_cls = MagicMock()
        self.helper = YuanrongHelper(self.blob_cls, self.blob_list_cls)

    def test_normalize_keys_valid(self):
        keys = ["abc", "key-1", "key_2"]
        result = self.helper.normalize_keys(keys)
        self.assertEqual(result, keys)

    def test_normalize_keys_with_special_chars(self):
        # Key with @ is valid
        keys = ["model@pcp0@dcp0@head_or_tp_rank:0@hash"]
        result = self.helper.normalize_keys(keys)
        self.assertEqual(result, keys)

    def test_normalize_keys_invalid_chars(self):
        # Key with spaces/dots gets normalized
        keys = ["key with spaces.and.dots"]
        result = self.helper.normalize_keys(keys)
        self.assertEqual(len(result), 1)
        # Should not contain spaces or dots
        self.assertNotIn(" ", result[0])
        self.assertNotIn(".", result[0])
        # Should have hash suffix
        self.assertIn("__", result[0])

    def test_normalize_keys_too_long(self):
        # Key longer than 255 chars
        long_key = "a" * 300
        result = self.helper.normalize_keys([long_key])
        self.assertLessEqual(len(result[0]), 255)

    def test_normalize_keys_empty(self):
        result = self.helper.normalize_keys([])
        self.assertEqual(result, [])

    def test_make_blob_lists(self):
        self.helper._device_id = 0
        addrs = [[100, 200], [300, 400]]
        sizes = [[10, 20], [30, 40]]
        result = self.helper.make_blob_lists(addrs, sizes)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.blob_cls.call_count, 4)

    def test_make_blob_lists_length_mismatch(self):
        self.helper._device_id = 0
        with self.assertRaises(ValueError):
            self.helper.make_blob_lists([[1, 2]], [[10, 20], [30, 40]])

    def test_make_blob_lists_inner_length_mismatch(self):
        self.helper._device_id = 0
        with self.assertRaises(ValueError):
            self.helper.make_blob_lists([[1, 2]], [[10]])

    def test_make_blob_lists_no_device(self):
        self.helper._device_id = None
        with self.assertRaises(RuntimeError):
            self.helper.make_blob_lists([[1]], [[10]])


if __name__ == "__main__":
    unittest.main()
