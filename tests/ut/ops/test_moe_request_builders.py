import unittest

import torch

import vllm_ascend.ops.fused_moe.moe_runtime_args as runtime_args
from vllm_ascend.ops.fused_moe.moe_request_builders import (
    build_fused_experts_input,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherRoutingMetadata,
    MoEMxfpParams,
    MoETokenDispatchOutput,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType


class TestMoERequestBuilders(unittest.TestCase):
    def test_runtime_args_facade_exports_public_types(self):
        expected_symbols = [
            "MoEAllGatherRoutingMetadata",
            "MoEAllToAllRoutingMetadata",
            "MoEFusedExpertsInput",
            "MoEMC2RoutingMetadata",
            "MoEMlpComputeInput",
            "MoEMlpKernelParams",
            "MoEMxfpParams",
            "MoEPrepareOutput",
            "MoEQuantParams",
            "MoEReservedQuantParams",
            "MoERoutingParams",
            "MoETokenCombineOutput",
            "MoETokenDispatchInput",
            "MoETokenDispatchOutput",
            "MoEWeights",
            "TMoERoutingMetadata",
        ]

        for symbol in expected_symbols:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(runtime_args, symbol))

    def test_build_fused_experts_input_preserves_runtime_semantics(self):
        for quant_type in (
            QuantType.NONE,
            QuantType.W4A16,
            QuantType.W4A8,
            QuantType.W8A8,
            QuantType.MXFP8,
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(4, 8)
                topk_weights = torch.randn(4, 2)
                topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32)
                request = build_fused_experts_input(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    w1=torch.randn(2, 8, 16),
                    w2=torch.randn(2, 16, 8),
                    quant_type=quant_type,
                    dynamic_eplb=True,
                    expert_map=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                    global_redundant_expert_num=2,
                    mc2_mask=torch.tensor([True, False, True, False]),
                    apply_router_weight_on_input=True,
                    log2phy=torch.tensor([3, 2, 1, 0], dtype=torch.int32),
                    pertoken_scale=torch.randn(4),
                    activation="gelu",
                )

                self.assertIs(request.hidden_states, hidden_states)
                self.assertIs(request.topk_weights, topk_weights)
                self.assertIs(request.topk_ids, topk_ids)
                self.assertTrue(request.dynamic_eplb)
                self.assertTrue(request.routing.apply_router_weight_on_input)
                self.assertEqual(request.routing.global_redundant_expert_num, 2)
                self.assertEqual(request.activation, "gelu")
                self.assertEqual(request.quant.quant_type, quant_type)

    def test_build_fused_experts_input_merges_dense_and_quant_weights(self):
        w1 = torch.randn(2, 8, 16)
        w2 = torch.randn(2, 16, 8)
        w1_scale = [torch.randn(1)]
        w2_scale = [torch.randn(1)]
        w1_scale_bias = torch.randn(1)
        w2_scale_bias = torch.randn(1)
        w1_offset = torch.randn(1)
        w2_offset = torch.randn(1)

        request = build_fused_experts_input(
            hidden_states=torch.randn(4, 8),
            topk_weights=torch.randn(4, 2),
            topk_ids=torch.randint(0, 4, (4, 2), dtype=torch.int32),
            w1=w1,
            w2=w2,
            quant_type=QuantType.W8A8,
            dynamic_eplb=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        )

        self.assertIsInstance(request.weights, MoEWeights)
        self.assertIs(request.weights.w1, w1)
        self.assertIs(request.weights.w2, w2)
        self.assertIs(request.weights.w1_scale, w1_scale)
        self.assertIs(request.weights.w2_scale, w2_scale)
        self.assertIs(request.weights.w1_scale_bias, w1_scale_bias)
        self.assertIs(request.weights.w2_scale_bias, w2_scale_bias)
        self.assertIs(request.weights.w1_offset, w1_offset)
        self.assertIs(request.weights.w2_offset, w2_offset)

    def test_build_token_dispatch_input_supports_remapped_topk_ids(self):
        request = build_fused_experts_input(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.randn(2, 1),
            topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
            w1=torch.randn(1, 4, 8),
            w2=torch.randn(1, 8, 4),
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
        routed_topk_ids = torch.tensor([[3], [2]], dtype=torch.int32)

        dispatch_request = build_token_dispatch_input(
            request=request,
            topk_ids=routed_topk_ids,
        )

        self.assertIs(dispatch_request.hidden_states, request.hidden_states)
        self.assertIs(dispatch_request.topk_weights, request.topk_weights)
        self.assertIs(dispatch_request.routing, request.routing)
        self.assertIs(dispatch_request.quant, request.quant)
        self.assertIs(dispatch_request.topk_ids, routed_topk_ids)

    def test_build_mlp_compute_input_derives_kernel_params(self):
        request = build_fused_experts_input(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            quant_type=QuantType.MXFP8,
            dynamic_eplb=False,
            mxfp=MoEMxfpParams(
                act_quant_type=torch.float8_e4m3fn,
                weight_quant_type=torch.float8_e4m3fn,
                scale_dtype=torch.float32,
                per_token_scale_dtype=torch.float16,
                use_bf16=False,
            ),
            w1_scale=[torch.randn(1)],
            w2_scale=[torch.randn(1)],
        )
        dispatch_result = MoETokenDispatchOutput(
            hidden_states=torch.randn(4, 8, dtype=torch.bfloat16),
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(4, 1),
            routing_metadata=MoEAllGatherRoutingMetadata(
                topk_weights=request.topk_weights,
                expanded_row_idx=torch.arange(4, dtype=torch.int32),
                restore_shape=torch.Size([2, 8]),
            ),
        )

        mlp_request = build_mlp_compute_input(
            request=request,
            dispatch_result=dispatch_result,
            use_fusion_ops=True,
        )

        self.assertIs(mlp_request.hidden_states, dispatch_result.hidden_states)
        self.assertIs(mlp_request.weights, request.weights)
        self.assertIs(mlp_request.weights.w1_scale, request.weights.w1_scale)
        self.assertIs(mlp_request.weights.w2_scale, request.weights.w2_scale)
        self.assertTrue(mlp_request.kernel.fusion)
        self.assertTrue(mlp_request.kernel.use_mxfp_quant)
        self.assertEqual(mlp_request.kernel.scale_type, torch.float32)
        self.assertEqual(mlp_request.kernel.per_token_scale_type, torch.float16)
        self.assertFalse(mlp_request.kernel.use_bf16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
