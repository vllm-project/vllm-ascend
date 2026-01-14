import pytest
import unittest
from unittest.mock import MagicMock

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from transformers import DeepseekV2Config


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        n_routed_experts = 256
        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        mock_model.config = config
        mock_model.get_expert_map.return_value = [i for i in range(n_routed_experts)]
        mock_model.get_log2phy_map.return_value = [i for i in range(n_routed_experts)]

    def test_init_fp16(self):
        
    