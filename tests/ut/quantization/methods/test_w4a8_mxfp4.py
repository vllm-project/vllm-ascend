from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.fp8 import AscendW4A8MXFPDSDynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod


class TestAscendW4A8MXFPFusedMoEMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep.return_value = Mock()
        self.method = AscendW4A8MXFPDynamicFusedMoEMethod()

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch.npu.empty_cache")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu.npu_format_cast")
    def test_process_weights_builds_per_expert_lists_for_dynamic_eplb(self, mock_format_cast, _mock_empty_cache):
        mock_format_cast.side_effect = lambda weight, *args, **kwargs: weight
        self.method.dynamic_eplb = True

        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (2, 8, 4), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (2, 6, 4), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(torch.randint(0, 255, (2, 8, 4), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (2, 6, 4), dtype=torch.uint8), requires_grad=False)
        w13_storage_ptr = layer.w13_weight.data.untyped_storage().data_ptr()
        w13_scale_storage_ptr = layer.w13_weight_scale.data.untyped_storage().data_ptr()

        self.method.process_weights_after_loading(layer)

        self.assertEqual(len(layer.w13_weight_list), 2)
        self.assertEqual(layer.w13_weight_list[0].shape, (4, 8))
        self.assertEqual(layer.w2_weight_list[0].shape, (4, 6))
        self.assertEqual(layer.w13_weight_scale_list[0].shape, (2, 8, 2))
        self.assertEqual(layer.w2_weight_scale_list[0].shape, (2, 6, 2))
        self.assertEqual(layer.w13_weight_list[0].untyped_storage().data_ptr(), w13_storage_ptr)
        self.assertNotEqual(layer.w13_weight_scale_list[0].untyped_storage().data_ptr(), w13_scale_storage_ptr)
        self.assertFalse(hasattr(layer, "w13_weight"))
        self.assertFalse(hasattr(layer, "w2_weight"))
        self.assertFalse(hasattr(layer, "w13_weight_scale"))
        self.assertFalse(hasattr(layer, "w2_weight_scale"))

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_forward_context")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.select_experts")
    def test_apply_passes_per_expert_lists_for_dynamic_eplb(self, mock_select_experts, mock_forward_context):
        self.method.dynamic_eplb = True
        topk_weights = torch.rand(3, 1)
        topk_ids = torch.zeros(3, 1, dtype=torch.int64)
        mock_select_experts.return_value = topk_weights, topk_ids
        moe_comm_method = Mock()
        moe_comm_method.fused_experts.return_value = torch.rand(3, 4)
        mock_forward_context.return_value = SimpleNamespace(moe_comm_method=moe_comm_method)

        layer = nn.Module()
        layer.w13_weight_list = [torch.rand(4, 8), torch.rand(4, 8)]
        layer.w2_weight_list = [torch.rand(4, 6), torch.rand(4, 6)]
        layer.w13_weight_scale_list = [torch.rand(2, 8, 2), torch.rand(2, 8, 2)]
        layer.w2_weight_scale_list = [torch.rand(2, 6, 2), torch.rand(2, 6, 2)]
        layer.swiglu_limit = 7.0

        self.method.apply(
            layer=layer,
            x=torch.rand(3, 4),
            router_logits=torch.rand(3, 2),
            top_k=1,
            renormalize=True,
            num_experts=2,
            enable_force_load_balance=False,
        )

        fused_experts_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_experts_input.weights.w1, layer.w13_weight_list)
        self.assertIs(fused_experts_input.weights.w2, layer.w2_weight_list)
        self.assertIs(fused_experts_input.weights.w1_scale, layer.w13_weight_scale_list)
        self.assertIs(fused_experts_input.weights.w2_scale, layer.w2_weight_scale_list)
        self.assertTrue(fused_experts_input.dynamic_eplb)


class TestAscendW4A8MXFPDSFusedMoEMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm, mock_ep):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_ep.return_value = Mock()
        self.method = AscendW4A8MXFPDSDynamicFusedMoEMethod({})

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.torch.npu.empty_cache")
    @patch("vllm_ascend.quantization.methods.fp8.torch_npu.npu_format_cast")
    def test_process_weights_builds_per_expert_lists_for_dynamic_eplb(self, mock_format_cast, _mock_empty_cache):
        mock_format_cast.side_effect = lambda weight, *args, **kwargs: weight
        self.method.dynamic_eplb = True

        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randint(0, 255, (2, 8, 4), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = nn.Parameter(torch.randint(0, 255, (2, 6, 4), dtype=torch.uint8), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(torch.randint(0, 255, (2, 8, 4), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight_scale = nn.Parameter(torch.randint(0, 255, (2, 6, 4), dtype=torch.uint8), requires_grad=False)
        w13_storage_ptr = layer.w13_weight.data.untyped_storage().data_ptr()
        w13_scale_storage_ptr = layer.w13_weight_scale.data.untyped_storage().data_ptr()

        self.method.process_weights_after_loading(layer)

        self.assertEqual(len(layer.w13_weight_list), 2)
        self.assertEqual(layer.w13_weight_list[0].shape, (4, 8))
        self.assertEqual(layer.w2_weight_list[0].shape, (4, 6))
        self.assertEqual(layer.w13_weight_scale_list[0].shape, (2, 8, 2))
        self.assertEqual(layer.w2_weight_scale_list[0].shape, (2, 6, 2))
        self.assertEqual(layer.w13_weight_list[0].untyped_storage().data_ptr(), w13_storage_ptr)
        self.assertNotEqual(layer.w13_weight_scale_list[0].untyped_storage().data_ptr(), w13_scale_storage_ptr)
        self.assertFalse(hasattr(layer, "w13_weight"))
        self.assertFalse(hasattr(layer, "w2_weight"))
        self.assertFalse(hasattr(layer, "w13_weight_scale"))
        self.assertFalse(hasattr(layer, "w2_weight_scale"))
