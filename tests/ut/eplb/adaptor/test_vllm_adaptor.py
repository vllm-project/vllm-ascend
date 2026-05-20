import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.methods.base import QuantType


class TestVllmAdaptor(unittest.TestCase):

    def setUp(self):
        VllmEplbAdaptor._registered_moe_layers = []

        self.layer = MagicMock()
        self.layer.local_num_experts = 4
        self.layer.ep_rank = 0
        self.layer.quant_type = QuantType.W8A8
        self.layer.w13_weight_list = [torch.randn(256, 128) for _ in range(4)]
        self.layer.w2_weight_list = [torch.randn(128, 256) for _ in range(4)]
        self.layer.w13_weight_scale_fp32_list = [torch.tensor([1.0]) for _ in range(4)]
        self.layer.w2_weight_scale_list = [torch.tensor([1.0]) for _ in range(4)]
        self.layer.moe_load = torch.randn(4)
        self.layer.global_expert_map = torch.arange(16).reshape(4, 4)
        self.layer.get_log2phy_map.return_value = torch.arange(4)
        self.layer.clear_moe_load = lambda: self.layer.moe_load.zero_()
        VllmEplbAdaptor.register_layer(self.layer)

        self.model = MagicMock()
        self.model.quant_config = MagicMock()
        self.model.config.first_k_dense_replace = 0
        del self.model.language_model

        self.mock_rank = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0).start()
        self.mock_size = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=4).start()
        self.adaptor = VllmEplbAdaptor(self.model)

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()
        VllmEplbAdaptor._registered_moe_layers = []

    def test_register_layer(self):
        VllmEplbAdaptor._registered_moe_layers = []
        VllmEplbAdaptor.register_layer(MagicMock())
        VllmEplbAdaptor.register_layer(MagicMock())
        self.assertEqual(len(VllmEplbAdaptor._registered_moe_layers), 2)

    def test_init_fp16(self):
        VllmEplbAdaptor._registered_moe_layers = []
        layer = MagicMock()
        layer.local_num_experts = 4
        layer.ep_rank = 0
        layer.w13_weight = [torch.randn(256, 128) for _ in range(4)]
        layer.w2_weight = [torch.randn(128, 256) for _ in range(4)]
        layer.moe_load = torch.randn(4)
        layer.global_expert_map = torch.arange(16).reshape(4, 4)
        layer.get_log2phy_map.return_value = torch.arange(4)
        VllmEplbAdaptor.register_layer(layer)
        model = MagicMock()
        model.quant_config = None
        model.config.first_k_dense_replace = 0
        del model.language_model
        adaptor = VllmEplbAdaptor(model)
        self.assertEqual(adaptor.num_moe_layers, 1)
        self.assertEqual(adaptor.num_local_experts, 4)
        self.assertEqual(adaptor.expert_weight_names, ["w13_weight", "w2_weight"])

    def test_init_w8a8(self):
        self.assertEqual(self.adaptor.num_moe_layers, 1)
        self.assertEqual(self.adaptor.num_local_experts, 4)
        self.assertEqual(self.adaptor.ep_rank, 0)
        self.assertEqual(self.adaptor.expert_weight_names, [
            "w13_weight_list", "w2_weight_list",
            "w13_weight_scale_fp32_list", "w2_weight_scale_list",
        ])

    def test_init_unsupported_quant(self):
        VllmEplbAdaptor._registered_moe_layers = []
        layer = MagicMock()
        layer.quant_type = "FP8"
        layer.local_num_experts = 4
        layer.ep_rank = 0
        layer.moe_load = torch.randn(4)
        VllmEplbAdaptor.register_layer(layer)
        model = MagicMock()
        model.quant_config = MagicMock()
        model.config.first_k_dense_replace = 0
        del model.language_model
        with self.assertRaises(ValueError):
            VllmEplbAdaptor(model)

    def test_init_language_model(self):
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
        model = MagicMock()
        model.language_model = MagicMock()
        model.language_model.quant_config = MagicMock()
        model.config.text_config = MagicMock()
        model.config.text_config.first_k_dense_replace = 0
        VllmEplbAdaptor(model)

    def test_param_dict_keys(self):
        self.assertEqual(set(self.adaptor.param_dict.keys()), {
            "0.w13_weight_list", "0.w2_weight_list",
            "0.w13_weight_scale_fp32_list", "0.w2_weight_scale_list",
        })

    def test_buffer_tensor_contiguous(self):
        for buf_list in self.adaptor.buffer_tensor_list:
            for buf in buf_list:
                self.assertTrue(buf.is_contiguous())

    def test_buffer_tensor_shape(self):
        ref = self.adaptor.param_dict["0.w13_weight_list"][0]
        buf = self.adaptor.buffer_tensor_list[0][0]
        self.assertEqual(buf.shape, ref.shape)
        self.assertEqual(buf.dtype, ref.dtype)

    def test_get_rank_expert_workload(self):
        loads = self.adaptor.get_rank_expert_workload()
        self.assertEqual(loads.shape, (1, 4))

    def test_clear_all_moe_loads(self):
        self.adaptor.clear_all_moe_loads()
        self.assertTrue(torch.all(self.layer.moe_load == 0))

    def test_do_update_expert_map(self):
        new_map = torch.tensor([3, 2, 1, 0], dtype=torch.long)
        self.adaptor.expert_map_per_layer_cpu[0] = torch.zeros(4, dtype=torch.long)
        self.adaptor.do_update_expert_map(0, new_map)
        self.assertTrue(torch.equal(self.adaptor.expert_map_per_layer_cpu[0], new_map))

    def test_do_update_expert_weight(self):
        known = torch.full_like(self.adaptor.param_dict["0.w13_weight_list"][0], 42.0)
        self.adaptor.buffer_tensor_list[0][0].copy_(known)
        self.adaptor.do_update_expert_weight(0, 0, 0)
        self.assertTrue(torch.all(self.adaptor.param_dict["0.w13_weight_list"][0] == 42.0))

    def test_do_update_log2phy_map(self):
        new_map = torch.tensor([1, 1, 2, 2])
        self.adaptor.do_update_log2phy_map(0, new_map)
        self.assertTrue(torch.equal(self.adaptor.log2phy_map_per_layer[0], new_map))

    def test_do_update_log2phy_map_none(self):
        self.adaptor.log2phy_map_per_layer[0] = None
        self.adaptor.do_update_log2phy_map(0, torch.tensor([1, 1, 2, 2]))

    def test_get_global_expert_map(self):
        result = self.adaptor.get_global_expert_map()
        self.assertEqual(result.shape, (1, 4))
        self.assertIn(0, self.adaptor.expert_map_per_layer_cpu)


if __name__ == "__main__":
    unittest.main()
