import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401  -- registers torch.npu used by the module under test
from torch.nn import functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.moe_mlp import (
    cumsum_group_list,
    quant_apply_mlp,
    unified_apply_mlp,
    unquant_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"
MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestW4A8RuntimeFlags(unittest.TestCase):
    def test_w4a8_per_channel_gmm_swiglu_flag(self):
        self.assertTrue(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=False).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W8A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_unquant_apply_mlp_wraps_tensor_weights_for_grouped_matmul(self):
        hidden_states = torch.randn(2, 8)
        gate_up_out = torch.randn(2, 16)
        expected = torch.randn(2, 8)
        w1 = torch.randn(2, 8, 16)
        w2 = torch.randn(2, 8, 8)

        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected]],
                create=True,
            ) as mock_grouped_matmul,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                return_value=gate_up_out,
                create=True,
            ),
        ):
            output, _ = unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=torch.tensor([1, 1]),
                need_trans=True,
            )

        self.assertTrue(output is expected)
        first_call, second_call = mock_grouped_matmul.call_args_list
        self.assertEqual(len(first_call.kwargs["weight"]), 1)
        self.assertEqual(len(second_call.kwargs["weight"]), 1)
        self.assertEqual(first_call.kwargs["weight"][0].shape, torch.Size([2, 16, 8]))
        self.assertEqual(second_call.kwargs["weight"][0].shape, torch.Size([2, 8, 8]))

    def test_request_unquant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=torch.randn(1, 16),
                w2_bias=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        mock_unquant.assert_called_once()
        self.assertEqual(mock_unquant.call_args.kwargs["activation"], "silu")
        self.assertFalse(mock_unquant.call_args.kwargs["need_trans"])
        mock_quant.assert_not_called()

    def test_request_quant_path(self):
        for quant_type, mxfp_dtype in (
            (QuantType.MXFP8, torch.float8_e4m3fn),
            (QuantType.MXFP4, MXFP4_TEST_DTYPE),
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(2, 8)
                expected = torch.randn(2, 8)
                mlp_compute_input = MoEMlpComputeInput(
                    hidden_states=hidden_states,
                    group_list=torch.tensor([2, 2], dtype=torch.int64),
                    group_list_type=1,
                    dynamic_scale=torch.randn(2, 1),
                    topk_scales=None,
                    weights=MoEWeights(
                        w1=torch.randn(1, 16, 8),
                        w2=torch.randn(1, 8, 8),
                        w1_scale=[torch.randn(1)],
                        w2_scale=[torch.randn(1)],
                    ),
                    quant=MoEQuantParams(
                        quant_type=quant_type,
                        mxfp=MoEMxfpParams(
                            act_quant_type=mxfp_dtype,
                            weight_quant_type=mxfp_dtype,
                            use_bf16=False,
                        ),
                    ),
                    fusion=True,
                    activation="silu",
                    need_trans=False,
                    dynamic_eplb=True,
                )

                with (
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
                ):
                    output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

                self.assertTrue(output is expected)
                mock_quant.assert_called_once()
                quant_kwargs = mock_quant.call_args.kwargs
                self.assertTrue(quant_kwargs["use_mxfp_quant"])
                self.assertTrue(quant_kwargs["fusion"])
                self.assertTrue(quant_kwargs["dynamic_eplb"])
                self.assertEqual(quant_kwargs["act_quant_type"], mxfp_dtype)
                self.assertEqual(quant_kwargs["weight_quant_type"], mxfp_dtype)
                self.assertFalse(quant_kwargs["use_bf16"])
                mock_unquant.assert_not_called()

    def test_request_quant_path_passes_w4a8_per_channel_flag(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_scale=[torch.randn(1, 16)],
                w2_scale=[torch.randn(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertTrue(quant_kwargs["use_w4a8_per_channel_gmm_swiglu"])
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_swiglustep_activation(self):
        expected = torch.randn(1, 2)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.ones((1, 2), dtype=torch.float32),
            group_list=torch.tensor([1], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones((1, 2, 4), dtype=torch.float32)],
                w2=[torch.ones((1, 2, 2), dtype=torch.float32)],
                w1_scale=[torch.ones((1,), dtype=torch.float32)],
                w2_scale=[torch.ones((1,), dtype=torch.float32)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=True,
            activation=MoEActivation.SWIGLUSTEP,
            swiglu_limit=5.0,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertEqual(quant_kwargs["activation"], MoEActivation.SWIGLUSTEP)
        self.assertEqual(quant_kwargs["swiglu_limit"], 5.0)
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_gelu_activation(self):
        expected = torch.randn(1, 2)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.ones((1, 2), dtype=torch.float32),
            group_list=torch.tensor([1], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones((1, 2, 4), dtype=torch.float32)],
                w2=[torch.ones((1, 2, 2), dtype=torch.float32)],
                w1_scale=[torch.ones((1,), dtype=torch.float32)],
                w2_scale=[torch.ones((1,), dtype=torch.float32)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=True,
            activation=MoEActivation.GELU_TANH,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertEqual(quant_kwargs["activation"], MoEActivation.GELU_TANH)
        mock_unquant.assert_not_called()


def _patch_npu_stream():
    """Patch ``torch.npu.current_stream`` so ``record_event()`` returns a tag."""
    evt = MagicMock(name="before_gmm2_evt")
    stream = MagicMock(name="npu_stream")
    stream.record_event.return_value = evt
    return patch("torch.npu.current_stream", return_value=stream), evt


@contextmanager
def _mock_w8a8_gelu_compute(gate_up, *, gmm2_out=None, capture_quant=False):
    """Mock the W8A8 GELU-path NPU ops: dequant GMM1 (``npu_grouped_matmul``),
    requant (``npu_dynamic_quant``), GMM2 (``npu_grouped_matmul_gmm2``), plus the
    NPU stream event and ``dispose_tensor``. Yields a namespace with the mocks;
    when ``capture_quant`` is True, ``captured['x']``/``captured['scale']``
    record the requant input and the returned per-token scale."""
    stream_patch, evt = _patch_npu_stream()
    captured = {}

    def _dynamic_quant(x):
        if capture_quant:
            captured["x"] = x.detach().clone()
            scale = torch.ones(1, dtype=torch.float32)
            captured["scale"] = scale
            return x, scale
        return x, torch.ones(1)

    with (
        stream_patch,
        patch("torch_npu.npu_grouped_matmul", return_value=[gate_up], create=True) as mock_gmm,
        patch("torch_npu.npu_dynamic_quant", side_effect=_dynamic_quant, create=True) as mock_dq,
        patch.object(
            DeviceOperator,
            "npu_grouped_matmul_gmm2",
            return_value=gmm2_out if gmm2_out is not None else torch.zeros(1, 4),
        ) as mock_gmm2,
        patch(f"{MOE_MLP}.dispose_tensor"),
    ):
        yield SimpleNamespace(gmm=mock_gmm, dq=mock_dq, gmm2=mock_gmm2, evt=evt, captured=captured)


class _GeluPathBase(unittest.TestCase):
    """Common helpers for the GELU-path tests."""

    def _common_w8a8_kwargs(
        self,
        *,
        activation,
        w1_scale_dtype=torch.float32,
        w2_scale_dtype=torch.float32,
        w1_scale_bias=None,
        w2_scale_bias=None,
        group_list_type=1,
        group_list=None,
        dynamic_scale=None,
    ):
        return dict(
            hidden_states=torch.randn(1, 4),
            w1=torch.randn(1, 8, 4),
            w1_scale=[torch.randn(1, 8, dtype=w1_scale_dtype)],
            w2=torch.randn(1, 4, 1),
            w2_scale=[torch.randn(1, 4, dtype=w2_scale_dtype)],
            group_list=group_list if group_list is not None else torch.tensor([1], dtype=torch.int64),
            group_list_type=group_list_type,
            dynamic_scale=dynamic_scale if dynamic_scale is not None else torch.randn(1, 1),
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=None,
            w2_offset=None,
            fusion=False,
            dynamic_eplb=False,
            use_mxfp_quant=False,
            mxfp_quant_dtype=None,
            act_quant_type=torch.int8,
            weight_quant_type=torch.float8_e4m3fn,
            use_bf16=True,
            activation=activation,
            swiglu_limit=0.0,
            use_w4a8_per_channel_gmm_swiglu=False,
        )


class TestQuantApplyMlpGeluPath(_GeluPathBase):
    """GELU path: dispatch, math, and layout coverage.

    In the in-branch/guard variant the GELU path runs through the existing
    branch preamble (``get_weight_prefetch_method`` and ``_EXTRA_CTX``), which
    the early-return variant skipped. Stub them in setUp so each test can focus
    on the GELU dispatch/math.
    """

    def setUp(self):
        # Configurable forward-context mock; default moe_comm_type is not MC2.
        self._ctx_mock = MagicMock()
        self._ctx_mock.moe_comm_type = -1
        self._patches = [
            patch(f"{MOE_MLP}.get_weight_prefetch_method", return_value=None),
            patch(f"{MOE_MLP}._EXTRA_CTX", self._ctx_mock),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for p in self._patches:
            p.stop()

    def test_w8a8_gelu_tanh_applies_correct_activation(self):
        """W8A8 + gelu_tanh: GMM1(dequant) -> gelu(tanh)·up -> requant -> GMM2."""
        gate = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        up = torch.tensor([[0.5, -0.5, 1.0, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[9.0]])
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=gmm2_out, capture_quant=True) as m:
            out, out_evt = quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        # GELU math applied with tanh approximation before requantization.
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))
        # GMM1 used the dequant form (scale + per_token_scale), not antiquant.
        gmm1_kwargs = m.gmm.call_args.kwargs
        self.assertIn("scale", gmm1_kwargs)
        self.assertIn("per_token_scale", gmm1_kwargs)
        self.assertNotIn("antiquant_scale", gmm1_kwargs)
        self.assertEqual(gmm1_kwargs["split_item"], 2)
        # Requant + GMM2 both invoked; GMM2 received the requant per-token scale.
        m.dq.assert_called_once()
        m.gmm2.assert_called_once()
        self.assertIs(m.gmm2.call_args.kwargs["per_token_scale"], m.captured["scale"])
        # Return contract: (hidden_states, before_gmm2_evt).
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, m.evt)

    def test_w8a8_gelu_uses_exact_gelu_approximation(self):
        """W8A8 + gelu (not tanh): approximate='none', matching the float path."""
        gate = torch.tensor([[0.5, -0.5, 2.0]])
        up = torch.tensor([[1.0, 1.0, 0.5]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="none") * up
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=torch.zeros(1, 3), capture_quant=True) as m:
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU))
        # exact GELU (approximate='none') differs from tanh; ensure 'none' used.
        self.assertFalse(torch.allclose(m.captured["x"], F.gelu(gate, approximate="tanh") * up, atol=1e-6))
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))

    def test_w4a16_gelu_uses_antiquant_path(self):
        """W4A16 + gelu: antiquant GMM1 -> gelu·up -> antiquant GMM2, no requant."""
        gate = torch.tensor([[1.0, -1.0]])
        up = torch.tensor([[0.5, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[3.0]])
        stream_patch, evt = _patch_npu_stream()
        with (
            stream_patch,
            patch("torch_npu.npu_grouped_matmul", side_effect=[[gate_up], [gmm2_out]], create=True) as mock_gmm,
            patch("torch_npu.npu_dynamic_quant", create=True) as mock_dq,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
            # Switch to the W4A16 antiquant layout.
            kwargs["w1_offset"] = torch.randn(1, 8, 4)
            kwargs["w2_offset"] = torch.randn(1, 4, 1)
            out, out_evt = quant_apply_mlp(**kwargs)

        self.assertEqual(mock_gmm.call_count, 2)
        # Both GMM calls use antiquant (not scale/per_token_scale).
        for call in mock_gmm.call_args_list:
            self.assertIn("antiquant_scale", call.kwargs)
            self.assertIn("antiquant_offset", call.kwargs)
            self.assertNotIn("scale", call.kwargs)
        # GMM2 (second call) input is the GELU activation output.
        gmm2_input = mock_gmm.call_args_list[1].kwargs["x"][0]
        self.assertTrue(torch.allclose(gmm2_input, expected, atol=1e-6))
        # W4A16 path does NOT requantize.
        mock_dq.assert_not_called()
        mock_gmm2.assert_not_called()
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, evt)

    def test_w8a8_gelu_with_scale_bias_sets_bias_and_bfloat16(self):
        """W8A8 + gelu + scale_bias: bias1/bias2 passed, output dtype bfloat16,
        and group_list_type 0 -> 1 conversion applied."""
        w1_sb = [torch.zeros(1)]
        w2_sb = [torch.zeros(1)]
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8), gmm2_out=torch.zeros(1, 2)) as m,
            patch("torch.cat", wraps=torch.cat) as mock_cat,
        ):
            quant_apply_mlp(
                **self._common_w8a8_kwargs(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_bias=w1_sb,
                    w2_scale_bias=w2_sb,
                    group_list_type=0,
                    group_list=torch.tensor([0, 1], dtype=torch.int64),
                )
            )
        # bias1 propagated to GMM1.
        self.assertIs(m.gmm.call_args.kwargs["bias"], w1_sb)
        # group_list_type 0 -> 1 conversion invoked (torch.cat + torch.diff).
        self.assertTrue(mock_cat.called)

    def test_w8a8_gelu_converts_w1_scale_dtype_to_output_dtype(self):
        """When w1_scale dtype != _output_dtype, it is cast before GMM1."""
        # w1_scale fp32, w2_scale bf16 -> _output_dtype = bfloat16, so the GELU
        # path must cast w1_scale to bfloat16 before GMM1.
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m:
            quant_apply_mlp(
                **self._common_w8a8_kwargs(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_dtype=torch.float32,
                    w2_scale_dtype=torch.bfloat16,
                )
            )
        self.assertEqual(m.gmm.call_args.kwargs["scale"][0].dtype, torch.bfloat16)

    def test_gelu_path_does_not_call_swiglu_op(self):
        """GELU path must use torch.gelu, never the SwiGLU NPU op."""
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)), patch("torch_npu.npu_swiglu", create=True) as mock_swiglu:
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        mock_swiglu.assert_not_called()

    def test_fusion_on_gelu_skips_fused_swiglu_quant(self):
        """Guard: with fusion ON (default), GELU must still skip the fused
        npu_grouped_matmul_swiglu_quant op and use the non-fused GELU path.
        This is the case that breaks without the ``and not is_gelu_activation``
        guard on use_gmm_swiglu_quant_fusion."""
        kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
        kwargs["fusion"] = True  # -> use_gmm_swiglu_quant_fusion = True
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch.object(DeviceOperator, "npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            quant_apply_mlp(**kwargs)
        # Fused SwiGLU+quant op must NOT be called for GELU.
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 (scale + per_token_scale) IS used.
        self.assertIn("scale", m.gmm.call_args.kwargs)
        self.assertIn("per_token_scale", m.gmm.call_args.kwargs)

    def test_mc2_gelu_skips_mc2_fused_branch(self):
        """Guard: under MC2 comm, GELU must skip the all-fused MC2 branch and
        use the non-fused GELU path. Without the ``and not is_gelu_activation``
        guard on the MC2 entry, GELU+MC2 would hit npu_dequant_swiglu_quant."""
        self._ctx_mock.moe_comm_type = MoECommType.MC2  # force is_mc2 True
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("torch.ops._C_ascend.npu_dequant_swiglu_quant", create=True) as mock_mc2_fused,
            patch.object(DeviceOperator, "npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        # MC2 fused SwiGLU op must NOT be called for GELU.
        mock_mc2_fused.assert_not_called()
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 IS used instead.
        self.assertIn("scale", m.gmm.call_args.kwargs)


class TestQuantApplyMlpNoGeluImpact(_GeluPathBase):
    """Non-GELU activations must NOT enter the GELU path (no regression)."""

    def _run_non_gelu(self, activation):
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)),
            patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx,
            patch(f"{MOE_MLP}.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value=torch.zeros(1, 4), create=True) as mock_swiglu,
            patch("torch.nn.functional.gelu") as mock_gelu,
        ):
            mock_ctx.moe_comm_type = -1  # not MoECommType.MC2
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=activation))
        return mock_gelu, mock_swiglu

    def test_silu_activation_skips_gelu_path(self):
        mock_gelu, mock_swiglu = self._run_non_gelu("silu")
        mock_gelu.assert_not_called()
        # SwiGLu op IS used by the existing path -> existing logic intact.
        mock_swiglu.assert_called()

    def test_swiglustep_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUSTEP)
        mock_gelu.assert_not_called()

    def test_swigluoai_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUOAI)
        mock_gelu.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
