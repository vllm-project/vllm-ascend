import unittest
from unittest.mock import MagicMock, call, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import EPLB_EXPERT_WEIGHT_NAMES, VllmEplbAdaptor
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
        mock_model.modules.return_value = [self.mock_layer]
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
            model.modules.return_value = [layer]
            adaptor = VllmEplbAdaptor(model)

        self.assertEqual(adaptor.num_moe_layers, 1)
        self.assertEqual(adaptor.num_local_experts, 4)
        self.assertEqual(adaptor.ep_rank, 0)

    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_excludes_other_models_registered_layers(self, mock_get_config):
        mock_get_config.return_value.enable_fused_mc2 = 0
        draft_layer = MagicMock()
        VllmEplbAdaptor._registered_moe_layers.insert(0, draft_layer)
        adaptor = VllmEplbAdaptor(self.model)

        self.assertEqual(adaptor.moe_layers, [self.mock_layer])
        self.assertEqual(adaptor.num_moe_layers, 1)
        self.assertEqual(adaptor.expert_weight_key_per_layer[0], (QuantType.W8A8, False))
        self.assertEqual(VllmEplbAdaptor._registered_moe_layers, [draft_layer, self.mock_layer])

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
        model.modules.return_value = [w8a8_layer, mxfp8_layer]
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
        model.modules.return_value = list(VllmEplbAdaptor._registered_moe_layers)

        with self.assertRaisesRegex(AssertionError, "EPLB expert weight shapes mismatch"):
            VllmEplbAdaptor(model)

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()
        VllmEplbAdaptor._registered_moe_layers = []

    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.torch_npu.copy_memory_")
    def test_update_mxfp_expert_preserves_storage_and_weight_views(self, mock_copy):
        mock_copy.side_effect = lambda dst, src: dst.untyped_storage().copy_(src.untyped_storage())
        adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
        key = (QuantType.W4A8MXFP, False)
        weights = [torch.zeros(16, dtype=torch.uint8) for _ in range(4)]
        buffers = [torch.arange(16, dtype=torch.uint8) + index for index in range(4)]
        weights[0] = weights[0].view(torch.float4_e2m1fn_x2)
        buffers[0] = buffers[0].view(torch.float4_e2m1fn_x2)
        aliases = [tensor.view(torch.uint8) for tensor in weights]
        pointers = [tensor.data_ptr() for tensor in weights]
        adaptor.expert_weight_key_per_layer = {0: key}
        adaptor.expert_param_per_layer = {0: [weights]}
        adaptor.buffer_tensor_list = {key: [buffers]}

        adaptor.do_update_expert_weight(0, 0, 0)

        self.assertEqual(mock_copy.call_count, 2)
        for index, (tensor, alias, buffer) in enumerate(zip(weights, aliases, buffers)):
            self.assertEqual(tensor.data_ptr(), pointers[index])
            self.assertEqual(alias.data_ptr(), pointers[index])
            torch.testing.assert_close(alias, buffer.view(torch.uint8))


if __name__ == "__main__":
    unittest.main()
