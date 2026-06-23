import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.methods.base import QuantType


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        VllmEplbAdaptor._registered_moe_layers = []

        n_routed_experts = 256
        self.mock_layer = MagicMock()
        self.mock_layer.local_num_experts = n_routed_experts
        self.mock_layer.ep_rank = 0
        self.mock_layer.quant_type = QuantType.W8A8
        self.mock_layer.w13_weight_list = [torch.randn(256, 128) for _ in range(n_routed_experts)]
        self.mock_layer.w2_weight_list = [torch.randn(128, 256) for _ in range(n_routed_experts)]
        self.mock_layer.w13_weight_scale_fp32_list = [torch.tensor([1.0]) for _ in range(n_routed_experts)]
        self.mock_layer.w2_weight_scale_list = [torch.tensor([1.0]) for _ in range(n_routed_experts)]
        self.mock_layer.w13_weight = torch.randn(n_routed_experts, 256, 128)
        self.mock_layer.w2_weight = torch.randn(n_routed_experts, 128, 256)
        self.mock_layer.moe_load = torch.randn(n_routed_experts)
        self.mock_layer.global_expert_map = torch.arange(n_routed_experts * 4).reshape(n_routed_experts, 4)
        self.mock_layer.get_log2phy_map.return_value = torch.arange(4)
        self.mock_layer.clear_moe_load = MagicMock()
        VllmEplbAdaptor.register_layer(self.mock_layer)

        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        mock_model.config = config
        del mock_model.language_model
        self.model = mock_model
        num_dense_layers = getattr(config, "first_k_dense_replace", 0)
        self.model.model.layers[num_dense_layers].mlp.experts.quant_type = QuantType.W8A8

        self.mock_rank = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0).start()
        self.mock_size = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=4).start()

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_fp16(self, mock_func):
        self.model.quant_config = None
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_language_model_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        model = MagicMock()
        model.language_model = self.model
        model.config.text_config = self.model.config
        VllmEplbAdaptor(model)

    def test_pp_eplb_adaptor_init_with_registered_layer(self):
        """PP+EPLB: adaptor picks up MoE layers registered via register_layer."""
        VllmEplbAdaptor._registered_moe_layers = []
        layer = MagicMock()
        layer.local_num_experts = 4
        layer.ep_rank = 0
        layer.quant_type = QuantType.W8A8
        layer.w13_weight_list = [torch.randn(256, 128) for _ in range(4)]
        layer.w2_weight_list = [torch.randn(128, 256) for _ in range(4)]
        layer.w13_weight_scale_fp32_list = [torch.tensor([1.0]) for _ in range(4)]
        layer.w2_weight_scale_list = [torch.tensor([1.0]) for _ in range(4)]
        layer.moe_load = torch.randn(4)
        layer.global_expert_map = torch.arange(16).reshape(4, 4)
        layer.get_log2phy_map.return_value = torch.arange(4)
        VllmEplbAdaptor.register_layer(layer)

        with patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.enable_fused_mc2 = 0
            mock_get_config.return_value = mock_config
            model = MagicMock()
            model.quant_config = MagicMock()
            model.config.first_k_dense_replace = 0
            del model.language_model
            adaptor = VllmEplbAdaptor(model)

        self.assertEqual(adaptor.num_moe_layers, 1)
        self.assertEqual(adaptor.num_local_experts, 4)
        self.assertEqual(adaptor.ep_rank, 0)

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()
        VllmEplbAdaptor._registered_moe_layers = []


if __name__ == "__main__":
    unittest.main()
