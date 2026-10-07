#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310
from vllm_ascend._310p.fused_moe.moe_comm_method import AllGatherCommImpl310
from vllm_ascend._310p.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
    MoEWeights,
    build_fused_experts_input,
)
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.quant_type import QuantType


def build_mlp_compute_input_fixture(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    with_quant: bool,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    group_list_type: int = 1,
    layer=None,
    activation: MoEActivation | str = MoEActivation.SILU,
    topk_scales: torch.Tensor | None = None,
    lora_context=None,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=None,
        topk_scales=topk_scales,
        weights=MoEWeights(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale),
        quant=MoEQuantParams(quant_type=QuantType.W8A8 if with_quant else QuantType.NONE),
        fusion=False,
        layer=layer,
        activation=activation,
        need_trans=False,
        dynamic_eplb=False,
        lora_context=lora_context,
    )


class _UnquantLayer(SimpleNamespace):
    """310P unquantized routed-expert layer (weights pre-transposed + NZ)."""

    def __init__(self, hidden: int, inter: int, experts: int = 2):
        super().__init__(
            w13_weight=torch.randn(experts, hidden, inter, dtype=torch.float16),
            w2_weight=torch.randn(experts, inter, hidden, dtype=torch.float16),
        )


class _W8A8Layer(SimpleNamespace):
    """310P W8A8 routed-expert layer (int8 weights + per-channel scales)."""

    def __init__(self, hidden: int, inter: int, experts: int = 2):
        super().__init__(
            w13_weight=torch.randint(-8, 8, (experts, 2 * inter, hidden), dtype=torch.int8),
            w2_weight=torch.randint(-8, 8, (experts, hidden, inter), dtype=torch.int8),
            w13_weight_scale=torch.rand(experts, 2 * inter, dtype=torch.float32),
            w2_weight_scale=torch.rand(experts, hidden, dtype=torch.float32),
        )


class TestApplyMoEMLP310(TestBase):
    def test_fused_activation_path_dispatches_to_fused_hook(self):
        quant_method = MagicMock()
        quant_method.supports_fused_activation.return_value = True
        quant_method.apply_gmm1_act_quant.return_value = (torch.randn(4, 8), None)
        quant_method.apply_gmm2.return_value = torch.randn(4, 8)
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            group_list=torch.tensor([4], dtype=torch.int64),
            with_quant=True,
        )

        output, before_gmm2_evt = apply_moe_mlp(mlp_compute_input, quant_method)

        quant_method.apply_gmm1_act_quant.assert_called_once_with(mlp_compute_input)
        quant_method.apply_gmm1.assert_not_called()
        quant_method.apply_act_quant.assert_not_called()
        quant_method.apply_gmm2.assert_called_once()
        self.assertIsNotNone(before_gmm2_evt)

    def test_only_fused_path_is_used_for_supported_activation(self):
        quant_method = MagicMock()
        quant_method.supports_fused_activation.return_value = True
        quant_method.apply_gmm1_act_quant.return_value = (torch.randn(4, 8), None)
        quant_method.apply_gmm2.return_value = torch.randn(4, 8)
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            group_list=torch.tensor([4], dtype=torch.int64),
            with_quant=True,
        )

        output, before_gmm2_evt = apply_moe_mlp(mlp_compute_input, quant_method)

        quant_method.apply_gmm1_act_quant.assert_called_once_with(mlp_compute_input)
        # The separate gmm1 / act_quant hooks are no longer reachable on 310P.
        quant_method.apply_gmm1.assert_not_called()
        quant_method.apply_act_quant.assert_not_called()
        quant_method.apply_gmm2.assert_called_once()
        self.assertIsNotNone(before_gmm2_evt)

    def test_unsupported_activation_is_rejected(self):
        quant_method = MagicMock()
        # 310P MoE only supports swiglu; the method reports no support for the
        # requested activation.
        quant_method.supports_fused_activation.return_value = False
        for activation in (
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            with self.subTest(activation=activation):
                mlp_compute_input = build_mlp_compute_input_fixture(
                    hidden_states=torch.randn(4, 8),
                    w1=torch.randn(2, 8, 16),
                    w2=torch.randn(2, 16, 8),
                    group_list=torch.tensor([4], dtype=torch.int64),
                    with_quant=False,
                    activation=activation,
                )
                with self.assertRaises(NotImplementedError):
                    apply_moe_mlp(mlp_compute_input, quant_method)

        quant_method.apply_gmm1_act_quant.assert_not_called()
        quant_method.apply_gmm2.assert_not_called()


class TestUnquantHooks310(TestBase):
    def _build_method_and_input(self):
        layer = _UnquantLayer(hidden=8, inter=16)
        method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
        method.moe = SimpleNamespace(has_bias=False)
        group_list = torch.tensor([2, 4], dtype=torch.int64)
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8, dtype=torch.float16),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            group_list=group_list,
            with_quant=False,
            group_list_type=0,
            layer=layer,
        )
        return method, mlp_compute_input, layer

    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_grouped_matmul", create=True)
    def test_apply_gmm1_act_quant_matches_legacy_gmm1_swiglu(self, mock_npu_grouped_matmul, mock_npu_swiglu):
        method, mlp_compute_input, layer = self._build_method_and_input()
        mock_gmm1_out = torch.randn(4, 32, dtype=torch.float16)
        mock_swiglu_out = torch.randn(4, 16, dtype=torch.float16)
        mock_npu_grouped_matmul.return_value = [mock_gmm1_out]
        mock_npu_swiglu.return_value = mock_swiglu_out

        output, act_out_scale = method.apply_gmm1_act_quant(mlp_compute_input)

        self.assertIs(output, mock_swiglu_out)
        self.assertIsNone(act_out_scale)
        mock_npu_grouped_matmul.assert_called_once_with(
            x=[mlp_compute_input.hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )
        mock_npu_swiglu.assert_called_once_with(mock_gmm1_out)

    @patch("torch_npu.npu_grouped_matmul", create=True)
    def test_apply_gmm2_matches_legacy_gmm2(self, mock_npu_grouped_matmul):
        method, mlp_compute_input, layer = self._build_method_and_input()
        act_out = torch.randn(4, 32, dtype=torch.float16)
        mock_gmm2_out = torch.randn(4, 8, dtype=torch.float16)
        mock_npu_grouped_matmul.return_value = [mock_gmm2_out]

        output = method.apply_gmm2(mlp_compute_input, act_out, None)

        self.assertIs(output, mock_gmm2_out)
        mock_npu_grouped_matmul.assert_called_once_with(
            x=[act_out],
            weight=[layer.w2_weight],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )

    def test_get_mlp_weights_returns_tuple_layout(self):
        method, _, layer = self._build_method_and_input()
        w1, w2 = method.get_mlp_weights(layer)
        self.assertIs(w1, layer.w13_weight)
        self.assertIs(w2, layer.w2_weight)

    def test_supports_fused_activation_silu_only(self):
        method, _, _ = self._build_method_and_input()
        self.assertTrue(method.supports_fused_activation("silu"))
        self.assertTrue(method.supports_fused_activation("swiglu"))
        self.assertTrue(method.supports_fused_activation(MoEActivation.SILU))
        self.assertFalse(method.supports_fused_activation(MoEActivation.GELU))


class TestW8A8Hooks310(TestBase):
    def _build_method_and_input(self):
        layer = _W8A8Layer(hidden=8, inter=16)
        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16
        group_list = torch.tensor([2, 4], dtype=torch.int64)
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8, dtype=torch.float16),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_list=group_list,
            with_quant=True,
            group_list_type=0,
            layer=layer,
        )
        return method, mlp_compute_input, layer

    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True)
    def test_apply_gmm1_act_quant_matches_legacy_quant_apply_mlp(self, mock_quant_gmm, mock_npu_swiglu):
        method, mlp_compute_input, layer = self._build_method_and_input()
        mock_gmm1_out = torch.randn(4, 32, dtype=torch.float16)
        mock_swiglu_out = torch.randn(4, 32, dtype=torch.float16)
        mock_quant_gmm.return_value = mock_gmm1_out
        mock_npu_swiglu.return_value = mock_swiglu_out

        output, act_out_scale = method.apply_gmm1_act_quant(mlp_compute_input)

        self.assertIs(output, mock_swiglu_out)
        self.assertIsNone(act_out_scale)
        # Legacy quant_apply_mlp called: quant gmm1 -> swiglu (no explicit
        # activation quant; gmm2 re-quantizes internally).
        mock_quant_gmm.assert_called_once_with(
            x=mlp_compute_input.hidden_states,
            quantized_weight=layer.w13_weight,
            weight_scale=layer.w13_weight_scale,
            group_list=mlp_compute_input.group_list,
            quant_mode="pertoken",
        )
        mock_npu_swiglu.assert_called_once_with(mock_gmm1_out)

    @patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True)
    def test_apply_gmm2_matches_legacy_gmm2(self, mock_quant_gmm):
        method, mlp_compute_input, layer = self._build_method_and_input()
        act_out = torch.randn(4, 32, dtype=torch.float16)
        mock_gmm2_out = torch.randn(4, 8, dtype=torch.float16)
        mock_quant_gmm.return_value = mock_gmm2_out

        output = method.apply_gmm2(mlp_compute_input, act_out, None)

        self.assertIs(output, mock_gmm2_out)
        mock_quant_gmm.assert_called_once_with(
            x=act_out,
            quantized_weight=layer.w2_weight,
            weight_scale=layer.w2_weight_scale,
            group_list=mlp_compute_input.group_list,
            quant_mode="pertoken",
        )

    @patch("torch.cumsum")
    @patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True)
    def test_group_list_type_1_converts_to_cumsum_once(self, mock_quant_gmm, mock_cumsum):
        layer = _W8A8Layer(hidden=8, inter=16)
        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16
        count_group_list = torch.tensor([2, 2], dtype=torch.int64)
        cumsum_group_list = torch.tensor([2, 4], dtype=torch.int64)
        mock_cumsum.return_value = cumsum_group_list
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8, dtype=torch.float16),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_list=count_group_list,
            with_quant=True,
            group_list_type=1,
            layer=layer,
        )

        method.apply_gmm1_act_quant(mlp_compute_input)
        method.apply_gmm2(mlp_compute_input, torch.randn(4, 32), None)

        # Both gmm1 and gmm2 must receive the SAME converted group_list, and
        # cumsum must be applied once per call (not accumulated twice).
        self.assertEqual(mock_cumsum.call_count, 2)
        for call_args in mock_quant_gmm.call_args_list:
            self.assertIs(call_args.kwargs["group_list"], cumsum_group_list)

    def test_supports_fused_activation_silu_only(self):
        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        self.assertTrue(method.supports_fused_activation("silu"))
        self.assertTrue(method.supports_fused_activation(MoEActivation.SILU))
        self.assertFalse(method.supports_fused_activation("gelu"))
        self.assertFalse(method.supports_fused_activation(MoEActivation.GELU))
        self.assertFalse(method.supports_fused_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE))

    def test_get_mlp_weights_returns_moeweights_payload(self):
        method, _, layer = self._build_method_and_input()
        weights = method.get_mlp_weights(layer)
        self.assertIsInstance(weights, MoEWeights)
        self.assertIs(weights.w1, layer.w13_weight)
        self.assertIs(weights.w1_scale, layer.w13_weight_scale)
        self.assertIs(weights.w2, layer.w2_weight)
        self.assertIs(weights.w2_scale, layer.w2_weight_scale)


class TestApplyMoeMLPFullChain310(TestBase):
    """End-to-end dispatch through real 310P methods with mocked NPU kernels.

    Asserts the kernel invocation sequence is identical to the pre-refactor
    ``unified_apply_mlp`` (quant/unquant) implementations.
    """

    def test_unquant_chain_matches_legacy_unified_apply_mlp(self):
        layer = _UnquantLayer(hidden=8, inter=16)
        method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
        method.moe = SimpleNamespace(has_bias=False)
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8, dtype=torch.float16),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            group_list=torch.tensor([2, 4], dtype=torch.int64),
            with_quant=False,
            group_list_type=0,
            layer=layer,
        )
        gmm1_out = torch.randn(4, 32, dtype=torch.float16)
        gmm2_out = torch.randn(4, 8, dtype=torch.float16)

        with (
            patch("torch_npu.npu_grouped_matmul", create=True) as mock_gmm,
            patch("torch_npu.npu_swiglu") as mock_swiglu,
        ):
            mock_gmm.side_effect = [[gmm1_out], [gmm2_out]]
            mock_swiglu.return_value = torch.randn(4, 32, dtype=torch.float16)

            output, _ = apply_moe_mlp(mlp_compute_input, method)

        self.assertEqual(mock_gmm.call_count, 2)
        self.assertEqual(mock_swiglu.call_count, 1)
        mock_gmm.assert_has_calls(
            [
                call(
                    x=[mlp_compute_input.hidden_states],
                    weight=[layer.w13_weight],
                    split_item=2,
                    group_list_type=0,
                    group_type=0,
                    group_list=mlp_compute_input.group_list,
                ),
                call(
                    x=[mock_swiglu.return_value],
                    weight=[layer.w2_weight],
                    split_item=2,
                    group_list_type=0,
                    group_type=0,
                    group_list=mlp_compute_input.group_list,
                ),
            ],
            any_order=True,
        )
        self.assertEqual(output.shape, mlp_compute_input.hidden_states.shape)

    def test_quant_chain_matches_legacy_unified_apply_mlp(self):
        layer = _W8A8Layer(hidden=8, inter=16)
        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16
        mlp_compute_input = build_mlp_compute_input_fixture(
            hidden_states=torch.randn(4, 8, dtype=torch.float16),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_list=torch.tensor([2, 4], dtype=torch.int64),
            with_quant=True,
            group_list_type=0,
            layer=layer,
        )
        gmm1_out = torch.randn(4, 32, dtype=torch.float16)
        gmm2_out = torch.randn(4, 8, dtype=torch.float16)

        with (
            patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True) as mock_quant_gmm,
            patch("torch_npu.npu_swiglu") as mock_swiglu,
        ):
            mock_quant_gmm.side_effect = [gmm1_out, gmm2_out]
            mock_swiglu.return_value = torch.randn(4, 32, dtype=torch.float16)

            output, _ = apply_moe_mlp(mlp_compute_input, method)

        self.assertEqual(mock_quant_gmm.call_count, 2)
        self.assertEqual(mock_swiglu.call_count, 1)
        mock_quant_gmm.assert_has_calls(
            [
                call(
                    x=mlp_compute_input.hidden_states,
                    quantized_weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale,
                    group_list=mlp_compute_input.group_list,
                    quant_mode="pertoken",
                ),
                call(
                    x=mock_swiglu.return_value,
                    quantized_weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale,
                    group_list=mlp_compute_input.group_list,
                    quant_mode="pertoken",
                ),
            ],
            any_order=True,
        )
        self.assertEqual(output.shape, mlp_compute_input.hidden_states.shape)


class TestAllGatherCommImpl310FusedExperts(TestBase):
    def _build_comm_and_input(self):
        comm = AllGatherCommImpl310.__new__(AllGatherCommImpl310)
        comm.moe_config = SimpleNamespace(
            activation=MoEActivation.SILU,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            # Upstream reads these from the moe config (no getattr default).
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
        )
        comm.token_dispatcher = MagicMock()
        hidden_states = torch.randn(4, 8, dtype=torch.float16)
        topk_weights = torch.rand(4, 1, dtype=torch.float16)
        topk_ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)
        fused_experts_input = build_fused_experts_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=SimpleNamespace(),
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=0,
            dynamic_scale=None,
            topk_scales=None,
            combine_metadata=SimpleNamespace(
                topk_weights=topk_weights,
                expanded_row_idx=None,
                restore_shape=hidden_states.shape,
            ),
        )
        comm.token_dispatcher.token_dispatch.return_value = dispatch_output
        comm.token_dispatcher.token_combine.return_value = hidden_states
        return comm, fused_experts_input, hidden_states

    def test_fused_experts_dispatches_apply_moe_mlp_with_quant_method(self):
        comm, fused_experts_input, hidden_states = self._build_comm_and_input()
        quant_method = MagicMock()
        mlp_output = torch.randn(4, 8, dtype=torch.float16)
        before_gmm2_evt = object()

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX") as mock_ctx,
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.apply_moe_mlp") as mock_apply_mlp,
            patch(
                "vllm_ascend.ops.fused_moe.dataclass.moe_mlp.enable_fusion_gmmswigluquant",
                return_value=False,
            ),
        ):
            mock_ctx.moe_comm_method = object()
            mock_apply_mlp.return_value = (mlp_output, before_gmm2_evt)
            result = comm.fused_experts(fused_experts_input=fused_experts_input, quant_method=quant_method)

        self.assertIs(result.routed_out, hidden_states)
        self.assertIs(result.before_gmm2_evt, before_gmm2_evt)
        mock_apply_mlp.assert_called_once()
        self.assertIs(mock_apply_mlp.call_args.args[1], quant_method)
        # The MLP compute input must carry the routed-expert layer so hooks can
        # read the weights from it.
        self.assertIs(mock_apply_mlp.call_args.args[0].layer, fused_experts_input.layer)
        comm.token_dispatcher.token_combine.assert_called_once_with(
            hidden_states=mlp_output,
            combine_metadata=comm.token_dispatcher.token_dispatch.return_value.combine_metadata,
        )

    def test_fused_experts_without_quant_method_raises_not_implemented(self):
        comm, fused_experts_input, _ = self._build_comm_and_input()
        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX") as mock_ctx,
            patch(
                "vllm_ascend.ops.fused_moe.dataclass.moe_mlp.enable_fusion_gmmswigluquant",
                return_value=False,
            ),
        ):
            mock_ctx.moe_comm_method = object()
            with self.assertRaises(NotImplementedError):
                comm.fused_experts(fused_experts_input=fused_experts_input)
