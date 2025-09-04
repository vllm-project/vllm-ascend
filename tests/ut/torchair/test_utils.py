import os
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from unittest.mock import MagicMock, patch

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

    def test_torchair_cache_dir_multiple_ranks(self):
        ranks = [0, 1, 2, 3]
        values = [100, 200, 300, 400]

        with ThreadPoolExecutor() as executor:
            executor.map(utils.write_kv_cache_bytes_to_file, ranks, values)
        for rank, expected in zip(ranks, values):
            self.assertEqual(expected,
                             utils.read_kv_cache_bytes_from_file(rank))
        utils.delete_torchair_cache_file()

        self.assertFalse(utils.check_torchair_cache_exist(),
                         "Delete torchair cache dir failed")
        self.assertFalse(utils.check_kv_cache_bytes_cache_exist(),
                         "Delete kv cache bytes cache dir failed")

    def test_delete_torchair_cache_file_multiple_times(self):
        utils.write_kv_cache_bytes_to_file(0, 100)
        utils.delete_torchair_cache_file()
        for i in range(5):
            try:
                utils.delete_torchair_cache_file()
            except FileNotFoundError:
                self.fail(
                    f"Unexpected FileNotFoundError on delete call #{i+2}")

    @patch('vllm.ModelRegistry')
    def test_register_torchair_model(self, mock_model_registry):
        mock_registry = MagicMock()
        mock_model_registry.return_value = mock_registry
        utils.register_torchair_model()

        self.assertEqual(mock_model_registry.register_model.call_count, 6)
        call_args_list = mock_model_registry.register_model.call_args_list

        expected_registrations = [
            ("DeepSeekMTPModel",
             "vllm_ascend.torchair.models.torchair_deepseek_mtp:TorchairDeepSeekMTP"
             ),
            ("DeepseekV2ForCausalLM",
             "vllm_ascend.torchair.models.torchair_deepseek_v2:TorchairDeepseekV2ForCausalLM"
             ),
            ("DeepseekV3ForCausalLM",
             "vllm_ascend.torchair.models.torchair_deepseek_v3:TorchairDeepseekV3ForCausalLM"
             ),
            ("Qwen2ForCausalLM",
             "vllm_ascend.torchair.models.qwen2:CustomQwen2ForCausalLM"),
            ("Qwen3MoeForCausalLM",
             "vllm_ascend.torchair.models.qwen3_moe:CustomQwen3MoeForCausalLM"
             ),
            ("PanguProMoEForCausalLM",
             "vllm_ascend.torchair.models.torchair_pangu_moe:PanguProMoEForCausalLM"
             )
        ]

        for i, (expected_name,
                expected_path) in enumerate(expected_registrations):
            args, kwargs = call_args_list[i]
            self.assertEqual(args[0], expected_name)
            self.assertEqual(args[1], expected_path)

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    def test_converting_weight_acl_format(self, mock_npu_cast,
                                          mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_get_format.return_value = 1
        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        utils.converting_weight_acl_format(model, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(fused_moe.w13_weight.data, 1)

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    def test_converting_weight_acl_format_format_true(self, mock_npu_cast,
                                                      mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ
        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        utils.converting_weight_acl_format(model, ACL_FORMAT_FRACTAL_NZ)
        mock_npu_cast.assert_not_called()
