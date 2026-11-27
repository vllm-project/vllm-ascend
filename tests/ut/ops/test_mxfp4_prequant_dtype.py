# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

import vllm_ascend.ops.fused_moe.routed_experts as routed_module
import vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4 as quant_module
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4 import AscendW4A4MXFP4DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


class TestMXFP4PrequantGuards(TestCase):
    def test_unsafe_routers_and_layer_configuration_keep_float(self):
        class CustomRouter(AscendFusedTopKRouter):
            pass

        comm = object.__new__(AllGatherCommImpl)
        for router_type in (AscendFusedTopKRouter, AscendGroupedTopKRouter, CustomRouter):
            for allow in (False, True):
                for custom in (None, lambda x: x):
                    router = object.__new__(router_type)
                    router.custom_routing_function = custom
                    layer = SimpleNamespace(
                        quant_type=QuantType.W4A4MXFP,
                        moe_config=SimpleNamespace(is_sequence_parallel=False),
                        _allow_mxfp4_prequant=allow,
                        router=router,
                    )
                    expected = (
                        QuantType.W4A4MXFP
                        if allow and router_type is not CustomRouter and custom is None
                        else QuantType.NONE
                    )
                    with self.subTest(router=router_type, allow=allow, custom=custom):
                        self.assertEqual(AscendRoutedExperts._get_prepare_quant_type(layer, comm), expected)

    def test_other_backends_and_sp_keep_existing_prepare_policy(self):
        layer = SimpleNamespace(
            quant_type=QuantType.W4A4MXFP,
            moe_config=SimpleNamespace(is_sequence_parallel=False),
            _allow_mxfp4_prequant=False,
        )
        self.assertEqual(AscendRoutedExperts._get_prepare_quant_type(layer, object()), QuantType.W4A4MXFP)
        layer.moe_config.is_sequence_parallel = True
        comm = object.__new__(AllGatherCommImpl)
        self.assertEqual(AscendRoutedExperts._get_prepare_quant_type(layer, comm), QuantType.W4A4MXFP)

    def test_forward_preserves_dtype_and_clears_it_on_failure(self):
        for dtype in (torch.float16, torch.bfloat16):
            for fail in (False, True):
                with self.subTest(dtype=dtype, fail=fail):
                    packed = torch.zeros(3, 64, dtype=torch.uint8)
                    comm = Mock()
                    comm.prepare.return_value = SimpleNamespace(
                        hidden_states=packed,
                        router_logits=torch.zeros(3, 8),
                        mc2_mask=None,
                        padded_hidden_states_shape=None,
                        pertoken_scale=torch.ones(3, 2, 2, dtype=torch.uint8),
                    )
                    layer = SimpleNamespace(
                        enable_npugraph_ex_static_kernel=False,
                        moe_config=SimpleNamespace(is_sequence_parallel=False),
                        _get_prepare_quant_type=lambda _: QuantType.W4A4MXFP,
                        _select_experts=Mock(return_value=(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int32))),
                        router=SimpleNamespace(eplb_state=None),
                        quant_method=Mock(),
                        return_with_event=False,
                    )

                    def apply(layer=layer, dtype=dtype, fail=fail, **kwargs):
                        self.assertEqual(layer.ascend_activation_dtype, dtype)
                        if fail:
                            raise RuntimeError("expert failure")
                        return SimpleNamespace(routed_out=torch.ones(3, 128, dtype=dtype))

                    layer.quant_method.apply.side_effect = apply
                    context = SimpleNamespace(in_profile_run=False, moe_comm_type=None, moe_comm_method=None)
                    with (
                        patch.object(routed_module, "get_forward_context", return_value=SimpleNamespace()),
                        patch.object(routed_module, "_EXTRA_CTX", context),
                        patch.object(routed_module, "get_moe_comm_method", return_value=comm),
                    ):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "expert failure"):
                                AscendRoutedExperts.forward_impl(
                                    layer,
                                    hidden_states=torch.ones(3, 128, dtype=dtype),
                                    router_logits=torch.zeros(3, 8),
                                )
                        else:
                            AscendRoutedExperts.forward_impl(
                                layer,
                                hidden_states=torch.ones(3, 128, dtype=dtype),
                                router_logits=torch.zeros(3, 8),
                            )
                    self.assertIsNone(layer.ascend_activation_dtype)
                    self.assertIsNone(layer.ascend_pertoken_scale)
                    self.assertIsNone(layer.ascend_mc2_mask)


class TestMXFP4OutputDtype(TestCase):
    def test_apply_propagates_original_dtype_into_expert_parameters(self):
        scheme = object.__new__(AscendW4A4MXFP4DynamicFusedMoEMethod)
        comm = SimpleNamespace(fused_experts=Mock())
        for dtype in (torch.float16, torch.bfloat16):
            layer = SimpleNamespace(
                ascend_activation_dtype=dtype,
                ascend_expert_map=None,
                global_redundant_expert_num=0,
                ascend_mc2_mask=None,
                apply_router_weight_on_input=False,
                ascend_pertoken_scale=torch.ones(3, 2, 2),
                activation="silu",
            )
            with (
                patch.object(quant_module, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)),
                patch.object(quant_module, "build_fused_experts_input") as build,
            ):
                scheme.apply(layer, torch.zeros(3, 64, dtype=torch.uint8), None, None, None, None)
                self.assertEqual(build.call_args.kwargs["mxfp_use_bf16"], dtype == torch.bfloat16)

    def test_packed_apply_requires_original_dtype(self):
        scheme = object.__new__(AscendW4A4MXFP4DynamicFusedMoEMethod)
        with self.assertRaisesRegex(ValueError, "original activation dtype"):
            scheme.apply(SimpleNamespace(), torch.zeros(3, 64, dtype=torch.uint8), None, None, None, None)

    def test_gmm2_keeps_fp16_and_bf16_output(self):
        scheme = object.__new__(AscendW4A4MXFP4DynamicFusedMoEMethod)
        for dtype in (torch.float16, torch.bfloat16):
            mlp_input = SimpleNamespace(
                hidden_states=torch.zeros(3, 64, dtype=torch.uint8),
                layer=SimpleNamespace(w2_weight=None, w2_weight_scale=None),
                quant=SimpleNamespace(mxfp=SimpleNamespace(use_bf16=dtype == torch.bfloat16)),
                group_list_type=1,
                group_list=None,
            )
            with patch.object(quant_module.torch_npu, "npu_grouped_matmul", create=True) as gmm:
                gmm.side_effect = lambda **kwargs: [torch.ones(3, 128, dtype=kwargs["output_dtype"])]
                result = scheme.apply_gmm2(mlp_input, None, None)
                self.assertEqual(result.dtype, dtype)
