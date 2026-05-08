import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.methods.base import QuantType


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        n_routed_experts = 256
        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        mock_model.config = config
        mock_model.get_expert_map.return_value = [i for i in range(n_routed_experts)]
        mock_model.get_log2phy_map.return_value = [i for i in range(n_routed_experts)]
        del mock_model.language_model
        self.model = mock_model
        num_dense_layers = getattr(config, "first_k_dense_replace", 0)
        self.model.model.layers[num_dense_layers].mlp.experts.quant_type = QuantType.W8A8

        patcher = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_dynamic_eplb_group", return_value=MagicMock())
        self.mock_get_group = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_group = self.mock_get_group.return_value
        self.mock_group.rank_in_group = 0
        self.mock_group.world_size = 4

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_fp16(self, mock_func):
        self.model.quant_config = None
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_w8a8(self, mock_func):
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_language_model_w8a8(self, mock_func):
        model = MagicMock()
        model.language_model = self.model
        model.config.text_config = self.model.config
        VllmEplbAdaptor(model)


if __name__ == "__main__":
    unittest.main()
