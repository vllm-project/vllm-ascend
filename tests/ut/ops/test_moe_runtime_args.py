#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import unittest

import torch

import vllm_ascend.ops.fused_moe.moe_runtime_args as runtime_args
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoETokenDispatchOutput,
    MoEWeights,
    build_fused_experts_input,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.quantization.quant_type import QuantType


class TestMoERuntimeArgs(unittest.TestCase):
    def test_runtime_args_facade_exports_public_contracts_and_builders(self):
        expected_symbols = [
            "MoEAllGatherCombineMetadata",
            "MoEAllToAllCombineMetadata",
            "MoEFusedExpertsInput",
            "MoEMC2CombineMetadata",
            "MoEMlpComputeInput",
            "MoEPrepareOutput",
            "MoEQuantParams",
            "MoERoutingParams",
            "MoETokenDispatchInput",
            "MoETokenDispatchOutput",
            "MoEWeights",
            "TMoECombineMetadata",
            "build_fused_experts_input",
            "build_mlp_compute_input",
            "build_token_dispatch_input",
        ]

        for symbol in expected_symbols:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(runtime_args, symbol))
        self.assertFalse(hasattr(runtime_args, "MoEMxfpParams"))

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
                fused_experts_input = build_fused_experts_input(
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
                    mxfp_act_quant_type=torch.float8_e4m3fn if quant_type == QuantType.MXFP8 else None,
                )

                self.assertIs(fused_experts_input.hidden_states, hidden_states)
                self.assertIs(fused_experts_input.topk_weights, topk_weights)
                self.assertIs(fused_experts_input.topk_ids, topk_ids)
                self.assertTrue(fused_experts_input.dynamic_eplb)
                self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
                self.assertEqual(fused_experts_input.routing.global_redundant_expert_num, 2)
                self.assertEqual(fused_experts_input.activation, "gelu")
                self.assertEqual(fused_experts_input.quant.quant_type, quant_type)

    def test_build_fused_experts_input_merges_dense_and_quant_weights(self):
        w1 = torch.randn(2, 8, 16)
        w2 = torch.randn(2, 16, 8)
        w1_scale = [torch.randn(1)]
        w2_scale = [torch.randn(1)]
        w1_scale_bias = torch.randn(1)
        w2_scale_bias = torch.randn(1)
        w1_offset = torch.randn(1)
        w2_offset = torch.randn(1)

        fused_experts_input = build_fused_experts_input(
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

        self.assertIsInstance(fused_experts_input.weights, MoEWeights)
        self.assertIs(fused_experts_input.weights.w1, w1)
        self.assertIs(fused_experts_input.weights.w2, w2)
        self.assertIs(fused_experts_input.weights.w1_scale, w1_scale)
        self.assertIs(fused_experts_input.weights.w2_scale, w2_scale)
        self.assertIs(fused_experts_input.weights.w1_scale_bias, w1_scale_bias)
        self.assertIs(fused_experts_input.weights.w2_scale_bias, w2_scale_bias)
        self.assertIs(fused_experts_input.weights.w1_offset, w1_offset)
        self.assertIs(fused_experts_input.weights.w2_offset, w2_offset)

    def test_build_token_dispatch_input_supports_remapped_topk_ids(self):
        fused_experts_input = build_fused_experts_input(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.randn(2, 1),
            topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
            w1=torch.randn(1, 4, 8),
            w2=torch.randn(1, 8, 4),
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
        routed_topk_ids = torch.tensor([[3], [2]], dtype=torch.int32)

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )

        self.assertIs(token_dispatch_input.hidden_states, fused_experts_input.hidden_states)
        self.assertIs(token_dispatch_input.topk_weights, fused_experts_input.topk_weights)
        self.assertIs(token_dispatch_input.routing, fused_experts_input.routing)
        self.assertIs(token_dispatch_input.quant, fused_experts_input.quant)
        self.assertIs(token_dispatch_input.topk_ids, routed_topk_ids)

    def test_build_fused_experts_input_requires_primitive_mxfp_params_for_mxfp_quant(self):
        with self.assertRaisesRegex(ValueError, "primitive MXFP params are required"):
            build_fused_experts_input(
                hidden_states=torch.randn(2, 8),
                topk_weights=torch.randn(2, 2),
                topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
                w1=torch.randn(2, 8, 16),
                w2=torch.randn(2, 16, 8),
                quant_type=QuantType.MXFP8,
                dynamic_eplb=False,
            )

    def test_build_mlp_compute_input_derives_fusion_and_preserves_mxfp_params(self):
        fused_experts_input = build_fused_experts_input(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            quant_type=QuantType.MXFP8,
            dynamic_eplb=False,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch.float8_e4m3fn,
            mxfp_scale_dtype=torch.float32,
            mxfp_per_token_scale_dtype=torch.float16,
            mxfp_use_bf16=False,
            w1_scale=[torch.randn(1)],
            w2_scale=[torch.randn(1)],
        )
        token_dispatch_output = MoETokenDispatchOutput(
            hidden_states=torch.randn(4, 8, dtype=torch.bfloat16),
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(4, 1),
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=fused_experts_input.topk_weights,
                expanded_row_idx=torch.arange(4, dtype=torch.int32),
                restore_shape=torch.Size([2, 8]),
            ),
        )

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=True,
        )

        self.assertIs(mlp_compute_input.hidden_states, token_dispatch_output.hidden_states)
        self.assertIs(mlp_compute_input.weights, fused_experts_input.weights)
        self.assertIs(mlp_compute_input.weights.w1_scale, fused_experts_input.weights.w1_scale)
        self.assertIs(mlp_compute_input.weights.w2_scale, fused_experts_input.weights.w2_scale)
        self.assertTrue(mlp_compute_input.fusion)
        self.assertTrue(mlp_compute_input.quant.is_mxfp)
        assert mlp_compute_input.quant.mxfp is not None
        self.assertEqual(mlp_compute_input.quant.mxfp.scale_dtype, torch.float32)
        self.assertEqual(mlp_compute_input.quant.mxfp.per_token_scale_dtype, torch.float16)
        self.assertFalse(mlp_compute_input.quant.mxfp.use_bf16)

    def test_build_fused_experts_input_constructs_internal_mxfp_leaf_from_primitives(self):
        fused_experts_input = build_fused_experts_input(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            quant_type=QuantType.MXFP8,
            dynamic_eplb=False,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch.float8_e4m3fn,
            mxfp_scale_dtype=torch.float32,
            mxfp_per_token_scale_dtype=torch.float16,
            mxfp_use_bf16=False,
        )

        self.assertTrue(fused_experts_input.quant.is_mxfp)
        assert fused_experts_input.quant.mxfp is not None
        self.assertEqual(fused_experts_input.quant.mxfp.act_quant_type, torch.float8_e4m3fn)
        self.assertEqual(fused_experts_input.quant.mxfp.weight_quant_type, torch.float8_e4m3fn)
        self.assertEqual(fused_experts_input.quant.mxfp.scale_dtype, torch.float32)
        self.assertEqual(fused_experts_input.quant.mxfp.per_token_scale_dtype, torch.float16)
        self.assertFalse(fused_experts_input.quant.mxfp.use_bf16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
