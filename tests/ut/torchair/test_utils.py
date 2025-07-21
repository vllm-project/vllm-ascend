import os
from unittest import mock

import torch

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

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    @mock.patch('vllm_ascend.utils.is_310p')
    def test_maybe_converting_weight_acl_format(self, mock_310p, mock_npu_cast,
                                                mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_310p.return_value = True
        mock_get_format.return_value = 1
        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        utils.converting_weight_acl_format_310p(model, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(fused_moe.w13_weight.data, 1)

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    @mock.patch('vllm_ascend.utils.is_310p')
    def test_maybe_converting_weight_acl_format_format_true(
            self, mock_310p, mock_npu_cast, mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_310p.return_value = True
        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ
        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ

        utils.converting_weight_acl_format_310p(model, ACL_FORMAT_FRACTAL_NZ)
