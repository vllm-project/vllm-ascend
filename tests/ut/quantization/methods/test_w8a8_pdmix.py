from unittest.mock import MagicMock, patch
import torch
from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_vllm_config
from vllm_ascend.quantization.methods import AscendW8A8LinearMethod, AscendW8A8DynamicLinearMethod, \
    AscendW8A8PDMixLinearMethod, AscendW8A8PDMixFusedMoeMethod


class TestAscendW8A8PDMixLinearScheme(TestBase):
    def setUp(self):
        self.method = AscendW8A8LinearMethod()

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_init_creates_static_and_dynamic_methods(self, mock_vllm_config):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        scheme = AscendW8A8PDMixLinearMethod()
        self.assertTrue(isinstance(scheme._static_method, AscendW8A8LinearMethod))
        self.assertTrue(isinstance(scheme._dynamic_method, AscendW8A8DynamicLinearMethod))
        self.assertFalse(scheme._is_kv_consumer)

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_get_weight_delegates_to_static(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance
        mock_static_instance = MagicMock()
        mock_static_instance.get_weight.return_value = {"weight": torch.empty(128, 256, dtype=torch.int8)}
        mock_static_cls.return_value = mock_static_instance
        scheme = AscendW8A8PDMixLinearMethod()
        for input_size, output_size in [(64, 128), (256, 512), (1024, 2048)]:
            result = scheme.get_weight(input_size, output_size, torch.bfloat16)
            mock_static_instance.get_weight.assert_called_with(input_size, output_size, torch.bfloat16)
            mock_dynamic_instance.get_weight.assert_not_called()


    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_get_pertensor_param_delegates_to_static(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance
        mock_static_instance = MagicMock()
        mock_static_instance.get_pertensor_param.return_value = {}
        mock_static_cls.return_value = mock_static_instance
        scheme = AscendW8A8PDMixLinearMethod()
        result = scheme.get_pertensor_param(torch.bfloat16)
        mock_static_instance.get_pertensor_param.assert_called_once_with(torch.bfloat16)
        mock_dynamic_instance.get_pertensor_param.assert_not_called()

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_get_perchannel_param_delegates_to_static(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance
        mock_static_instance = MagicMock()
        mock_static_instance.get_perchannel_param.return_value = {}
        mock_static_cls.return_value = mock_static_instance
        scheme = AscendW8A8PDMixLinearMethod()
        result = scheme.get_perchannel_param(128, torch.bfloat16)
        mock_static_instance.get_perchannel_param.assert_called_once_with(128, torch.bfloat16)
        mock_dynamic_instance.get_perchannel_param.assert_not_called()

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_apply_uses_static_for_kv_consumer(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_static_instance = MagicMock()
        mock_static_instance.apply.return_value = torch.randn(4, 128)
        mock_static_cls.return_value = mock_static_instance
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance
        scheme = AscendW8A8PDMixLinearMethod()
        layer = MagicMock()
        layer.is_kv_consumer = True
        x = torch.randn(4, 256)
        scheme.apply(layer, x)
        mock_static_instance.apply.assert_called_once()
        mock_dynamic_instance.apply.assert_not_called()

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_apply_uses_dynamic_for_non_kv_consumer(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.apply.return_value = torch.randn(4, 128)
        mock_dynamic_cls.return_value = mock_dynamic_instance
        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance
        scheme = AscendW8A8PDMixLinearMethod()
        layer = MagicMock()
        layer.is_kv_consumer = False
        x = torch.randn(4, 256)
        scheme.apply(layer, x)
        mock_dynamic_instance.apply.assert_called_once()
        mock_static_instance.apply.assert_not_called()

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_process_weights_after_loading_sets_is_kv_consumer(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=None)
        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance
        scheme = AscendW8A8PDMixLinearMethod()
        layer = MagicMock()
        layer.weight_scale = MagicMock(data=torch.randn(128, 1, dtype=torch.bfloat16))
        scheme.process_weights_after_loading(layer)
        mock_static_instance.process_weights_after_loading.assert_called_once_with(layer)
        mock_dynamic_instance.process_weights_after_loading.assert_not_called()
        self.assertFalse(layer.is_kv_consumer)

    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8LinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.AscendW8A8DynamicLinearMethod")
    @patch("vllm_ascend.quantization.methods.w8a8_pdmix.get_current_vllm_config")
    def test_is_kv_consumer_when_kv_transfer_config_set(self, mock_vllm_config, mock_dynamic_cls, mock_static_cls):
        mock_kv_config = MagicMock()
        mock_kv_config.is_kv_consumer = True
        mock_vllm_config.return_value = create_mock_vllm_config(kv_transfer_config=mock_kv_config)
        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance
        scheme = AscendW8A8PDMixLinearMethod()
        layer = MagicMock()
        layer.weight_scale = MagicMock(data=torch.randn(128, 1, dtype=torch.bfloat16))
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.is_kv_consumer)


class TestAscendW8A8PDMixMoEScheme(TestBase):

    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def test_get_dynamic_quant_param(self, mock_ascend, mock_vllm, mock_ep, mock_mc2):
        mock_mc2.side_effect = AttributeError()
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = MagicMock(eplb_config=MagicMock(dynamic_eplb=False))
        scheme = AscendW8A8PDMixFusedMoeMethod()
        num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype = 8, 256, 128, torch.bfloat16
        result = scheme.get_dynamic_quant_param(num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype)
        # test adds extra params
        self.assertEqual(result["w2_deq_scale"].shape, (num_experts, hidden_sizes))
        self.assertEqual(result["w2_deq_scale"].dtype, torch.float32)
        self.assertEqual(result["w13_deq_scale"].shape, (num_experts, 2 * intermediate_size_per_partition))
        self.assertEqual(result["w13_deq_scale"].dtype, torch.float32)
        self.assertEqual(result["w2_input_offset"].shape, (num_experts, 1))
        self.assertEqual(result["w2_input_offset"].dtype, torch.int8)
        self.assertEqual(result["w13_input_offset"].shape, (num_experts, 1))
        self.assertEqual(result["w13_input_offset"].dtype, torch.int8)

        # test parent params
        self.assertIn("w13_weight_scale", result)
        self.assertIn("w13_weight_offset", result)
        self.assertIn("w2_weight_scale", result)
        self.assertIn("w2_weight_offset", result)