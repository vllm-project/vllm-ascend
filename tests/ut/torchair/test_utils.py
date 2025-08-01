import os
from contextlib import nullcontext

import torchair

from tests.ut.base import TestBase
from vllm_ascend.torchair import utils


class TestTorchairUtils(TestBase):

    def test_get_torchair_current_work_dir(self):
        cache_dir = utils.TORCHAIR_CACHE_DIR
        work_dir = utils._get_torchair_current_work_dir()
        self.assertEqual(cache_dir, work_dir)
        work_dir = utils._get_torchair_current_work_dir("test")
        self.assertEqual(os.path.join(cache_dir, "test"), work_dir)

    def test_torchair_cache_dir(self):
        utils.write_kv_cache_bytes_to_file(0, 100)
        self.assertTrue(utils.check_torchair_cache_exist(),
                        "Create torchair cache dir failed")
        self.assertTrue(utils.check_kv_cache_bytes_cache_exist(),
                        "Create kv cache bytes cache dir failed")
        kv_cache_bytes = utils.read_kv_cache_bytes_from_file(0)
        self.assertEqual(100, kv_cache_bytes)
        utils.delete_torchair_cache_file()
        self.assertFalse(utils.check_torchair_cache_exist(),
                         "Delete torchair cache dir failed")
        self.assertFalse(utils.check_kv_cache_bytes_cache_exist(),
                         "Delete kv cache bytes cache dir failed")

    def test_super_kernel(self):
        super_kernel_unenable = utils.super_kernel("prefix", "stream-fusion=1",
                                                   False)
        self.assertTrue(super_kernel_unenable, nullcontext())
        super_kernel_enable = utils.super_kernel("prefix", "stream-fusion=1",
                                                 True)
        self.assertIsInstance(super_kernel_enable, torchair.scope._Scope)
