import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import EPLB_EXPERT_WEIGHT_NAMES, VllmEplbAdaptor
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.quant_type import QuantType


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
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_fp16(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 1
        mock_get_config.return_value = mock_config
        self.model.quant_config = None
        adaptor = VllmEplbAdaptor(self.model)
        self.assertEqual(adaptor.expert_weight_key_per_layer[0], (QuantType.NONE, True))
        self.assertIs(adaptor.expert_param_per_layer[0][0][0], self.mock_layer.w13_weight_list[0])
        self.assertIs(adaptor.expert_param_per_layer[0][0][1], self.mock_layer.w2_weight_list[0])

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_fp16_with_parameter_accessor(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 1
        mock_get_config.return_value = mock_config
        self.model.quant_config = None
        self.mock_layer.get_eplb_parameter = MagicMock(side_effect=lambda name: getattr(self.mock_layer, name))

        adaptor = VllmEplbAdaptor(self.model)

        self.assertIs(adaptor.expert_param_per_layer[0][0][0], self.mock_layer.w13_weight_list[0])
        self.assertIs(adaptor.expert_param_per_layer[0][0][1], self.mock_layer.w2_weight_list[0])
        self.mock_layer.get_eplb_parameter.assert_has_calls([call("w13_weight_list"), call("w2_weight_list")])

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

    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_mixed_quant_type_per_layer(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 1
        mock_get_config.return_value = mock_config

        VllmEplbAdaptor._registered_moe_layers = []
        num_local_experts = 2
        w8a8_layer = MagicMock()
        w8a8_layer.local_num_experts = num_local_experts
        w8a8_layer.ep_rank = 0
        w8a8_layer.quant_type = QuantType.W8A8
        w8a8_layer.w13_weight_list = [torch.randn(2, 2) for _ in range(num_local_experts)]
        w8a8_layer.w2_weight_list = [torch.randn(2, 2) for _ in range(num_local_experts)]
        w8a8_layer.w13_weight_scale_fp32_list = [torch.randn(1) for _ in range(num_local_experts)]
        w8a8_layer.w2_weight_scale_list = [torch.randn(1) for _ in range(num_local_experts)]
        w8a8_layer.fused_w1_scale_list = [torch.randn(1) for _ in range(num_local_experts)]
        w8a8_layer.fused_w2_scale_list = [torch.randn(1) for _ in range(num_local_experts)]
        w8a8_layer.moe_load = torch.zeros(num_local_experts)
        w8a8_layer.global_expert_map = torch.arange(num_local_experts * 4).reshape(num_local_experts, 4)
        w8a8_layer.get_log2phy_map.return_value = torch.arange(4)

        mxfp8_layer = MagicMock()
        mxfp8_layer.local_num_experts = num_local_experts
        mxfp8_layer.ep_rank = 0
        mxfp8_layer.quant_type = QuantType.W8A8MXFP
        mxfp8_layer.w13_weight = torch.randn(num_local_experts, 2, 2)
        mxfp8_layer.w2_weight = torch.randn(num_local_experts, 2, 2)
        mxfp8_layer.w13_weight_scale = torch.randn(num_local_experts, 1)
        mxfp8_layer.w2_weight_scale = torch.randn(num_local_experts, 1)
        mxfp8_layer.moe_load = torch.zeros(num_local_experts)
        mxfp8_layer.global_expert_map = torch.arange(num_local_experts * 4).reshape(num_local_experts, 4)
        mxfp8_layer.get_log2phy_map.return_value = torch.arange(4)

        VllmEplbAdaptor.register_layer(w8a8_layer)
        VllmEplbAdaptor.register_layer(mxfp8_layer)

        model = MagicMock()
        model.quant_config = MagicMock()
        model.config.first_k_dense_replace = 0
        del model.language_model
        adaptor = VllmEplbAdaptor(model)

        w8a8_key = (QuantType.W8A8, True)
        mxfp8_key = (QuantType.W8A8MXFP, True)
        self.assertEqual(adaptor.expert_weight_key_per_layer[0], w8a8_key)
        self.assertEqual(adaptor.expert_weight_key_per_layer[1], mxfp8_key)
        self.assertEqual(len(adaptor.buffer_tensor_list[w8a8_key][0]), len(EPLB_EXPERT_WEIGHT_NAMES[w8a8_key]))
        self.assertEqual(len(adaptor.buffer_tensor_list[mxfp8_key][0]), len(EPLB_EXPERT_WEIGHT_NAMES[mxfp8_key]))
        self.assertEqual(len(adaptor.expert_param_per_layer[0][0]), len(EPLB_EXPERT_WEIGHT_NAMES[w8a8_key]))
        self.assertEqual(len(adaptor.expert_param_per_layer[1][0]), len(EPLB_EXPERT_WEIGHT_NAMES[mxfp8_key]))

    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_reused_buffer_requires_same_expert_weight_shape(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config

        VllmEplbAdaptor._registered_moe_layers = []
        num_local_experts = 2
        for weight_shape in [(2, 2), (3, 2)]:
            layer = MagicMock()
            layer.local_num_experts = num_local_experts
            layer.ep_rank = 0
            layer.quant_type = QuantType.W8A8
            layer.w13_weight_list = [torch.randn(*weight_shape) for _ in range(num_local_experts)]
            layer.w2_weight_list = [torch.randn(2, 2) for _ in range(num_local_experts)]
            layer.w13_weight_scale_fp32_list = [torch.randn(1) for _ in range(num_local_experts)]
            layer.w2_weight_scale_list = [torch.randn(1) for _ in range(num_local_experts)]
            layer.moe_load = torch.zeros(num_local_experts)
            layer.global_expert_map = torch.arange(num_local_experts * 4).reshape(num_local_experts, 4)
            layer.get_log2phy_map.return_value = torch.arange(4)
            VllmEplbAdaptor.register_layer(layer)

        model = MagicMock()
        model.quant_config = MagicMock()
        model.config.first_k_dense_replace = 0
        del model.language_model

        with self.assertRaisesRegex(AssertionError, "EPLB expert weight shapes mismatch"):
            VllmEplbAdaptor(model)

    def test_do_update_expert_map_refreshes_layer_runtime_maps(self):
        # Regression test for issue #14080: the layer-facing maps must track
        # the worker's authoritative map, not just the CPU bookkeeping copy.
        layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
        old_map = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, -1, -1], dtype=torch.int32)
        # Worker payload (int64): rank 1 owns physical experts 5..9 in slots 0..4.
        new_map = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4], dtype=torch.int64)
        layer.moe_config = SimpleNamespace(ep_rank=1)
        global_expert_map = torch.stack([torch.full((10,), -1, dtype=torch.int32), old_map])
        local_phys_ids = torch.zeros(5, dtype=torch.int64)
        layer.ascend_expert_map = old_map
        layer.global_expert_map = global_expert_map
        layer.local_phys_expert_ids = local_phys_ids

        adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
        adaptor.moe_layers = [layer]
        adaptor.expert_map_per_layer_cpu = [old_map.clone()]

        VllmEplbAdaptor.do_update_expert_map(adaptor, 0, new_map)

        self.assertTrue(torch.equal(adaptor.expert_map_per_layer_cpu[0], new_map.to(torch.int32)))
        self.assertTrue(torch.equal(layer.ascend_expert_map, new_map.to(torch.int32)))
        self.assertTrue(torch.equal(layer.global_expert_map[1], new_map.to(torch.int32)))
        self.assertTrue(torch.equal(layer.local_phys_expert_ids, torch.tensor([5, 6, 7, 8, 9], dtype=torch.int64)))
        # The refresh must update the maps in place (copy_): a captured ACL
        # graph holds references to the original tensor objects, and
        # rebinding the attributes would leave it reading the stale map.
        self.assertIs(layer.ascend_expert_map, old_map)
        self.assertIs(layer.global_expert_map, global_expert_map)
        self.assertIs(layer.local_phys_expert_ids, local_phys_ids)

    def test_export_tensor_to_file_writes_logical_expert_ids(self):
        # expert_maps is [num_layers, ep, physical]; entries index physical
        # expert IDs. The record file format stores logical IDs per device.
        expert_maps = torch.tensor(
            [
                [
                    [0, 1, 2, 3, 4, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 0, 1, 2, 3, 4],
                ]
            ],
            dtype=torch.int32,
        )
        adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
        adaptor.rank_id = 0
        adaptor.phys_to_logical = torch.tensor([7, 2, 0, 3, 5, 6, 1, 4, 7, 2], dtype=torch.int32)

        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name
        VllmEplbAdaptor._export_tensor_to_file(adaptor, expert_maps, path)
        with open(path) as f:
            record = json.load(f)
        os.unlink(path)

        device_experts = [device["device_expert"] for device in record["layer_list"][0]["device_list"]]
        self.assertEqual(device_experts, [[7, 2, 0, 3, 5], [6, 1, 4, 7, 2]])

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()
        VllmEplbAdaptor._registered_moe_layers = []


if __name__ == "__main__":
    unittest.main()
