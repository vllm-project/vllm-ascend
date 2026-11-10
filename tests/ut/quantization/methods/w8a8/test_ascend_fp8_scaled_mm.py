#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms.interface import PlatformEnum

from tests.ut.base import TestBase
from vllm_ascend.device.hardware_profile import QuantizationBackendFamily
from vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm import (
    _INT8_MAX,
    _INT8_MIN,
    _MIN_CHANNEL_SCALE,
    _MXFP8_GROUP_SIZE,
    AscendFP8ScaledMMInt8LinearKernel,
    AscendFP8ScaledMMLinearKernel,
    register_ascend_fp8_lm_head_kernel,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod,
)

N = 64
K = 128

# Patched in tests: resolving the real profile reads the device configuration.
_PROFILE_PATH = "vllm_ascend.device.hardware_profile.get_current_hardware_profile"


def make_config(out_dtype=torch.bfloat16, weight_shape=(N, K)):
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=out_dtype,
    )


def make_profile(mxfp8: bool, backend: QuantizationBackendFamily):
    return SimpleNamespace(
        supports=lambda capability: mxfp8,
        quantization_backend_family=backend,
    )


class TestAscendFP8ScaledMMKernel(TestBase):
    @patch(_PROFILE_PATH)
    def test_is_supported_only_on_out_of_tree_platform(self, mock_profile):
        mock_profile.return_value.supports.return_value = True
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
        ) as mock_platform:
            mock_platform.is_out_of_tree.return_value = True
            ok, _ = AscendFP8ScaledMMLinearKernel.is_supported()
            self.assertTrue(ok)

            mock_platform.is_out_of_tree.return_value = False
            ok, reason = AscendFP8ScaledMMLinearKernel.is_supported()
            self.assertFalse(ok)
            self.assertIn("out-of-tree", reason)

    @patch(_PROFILE_PATH)
    def test_is_supported_requires_engram_mxfp8_hardware(self, mock_profile):
        mock_profile.return_value.supports.return_value = False
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
        ) as mock_platform:
            mock_platform.is_out_of_tree.return_value = True
            ok, reason = AscendFP8ScaledMMLinearKernel.is_supported()
        self.assertFalse(ok)
        self.assertIn("MXFP8", reason)

    def test_can_implement_ptpc_quant_keys(self):
        ok, _ = AscendFP8ScaledMMLinearKernel.can_implement(make_config())
        self.assertTrue(ok)

    def test_can_implement_rejects_other_weight_keys(self):
        config = make_config()
        config.weight_quant_key = None
        ok, reason = AscendFP8ScaledMMLinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("per-channel", reason)

    def test_can_implement_rejects_other_activation_keys(self):
        config = make_config()
        config.activation_quant_key = None
        ok, reason = AscendFP8ScaledMMLinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("per-token", reason)

    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
    def test_can_implement_accepts_bf16_fp16_output(self, out_dtype):
        ok, _ = AscendFP8ScaledMMLinearKernel.can_implement(make_config(out_dtype))
        self.assertTrue(ok)

    def test_can_implement_rejects_other_output_dtypes(self):
        config = make_config(out_dtype=torch.float32)
        ok, reason = AscendFP8ScaledMMLinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("output dtype", reason)

    def test_can_implement_requires_k_multiple_of_group_size(self):
        config = make_config(weight_shape=(N, 100))
        ok, reason = AscendFP8ScaledMMLinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("reduction dim", reason)

    @patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config"
    )
    @patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
        return_value=0,
    )
    @patch(_PROFILE_PATH)
    def test_constructs_mxfp8_delegate(
        self, mock_profile, mock_scale_alg, mock_vllm
    ):
        mock_profile.return_value.supports.return_value = True
        config = make_config()
        mock_vllm.return_value = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description={"group_size": 32})
        )
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform.is_out_of_tree",
            return_value=True,
        ):
            kernel = AscendFP8ScaledMMLinearKernel(config, ["weight", "weight_scale"])
        self.assertEqual(kernel.mxfp8_method.group_size, _MXFP8_GROUP_SIZE)

    def test_apply_scaled_mm_not_implemented(self):
        kernel = object.__new__(AscendFP8ScaledMMLinearKernel)
        with self.assertRaisesRegex(NotImplementedError, "apply_weights"):
            kernel.apply_scaled_mm(
                A=torch.empty(1, K),
                B=torch.empty(K, N),
                out_dtype=torch.bfloat16,
                As=None,
                Bs=None,
                bias=None,
                output_shape=[1, N],
            )

    @patch("torch_npu.npu_quant_matmul")
    @patch(
        "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.AscendW8A8MXFP8DynamicLinearMethod"
    )
    def test_apply_weights_delegates_to_mxfp8_method(self, mock_method_cls, mock_matmul):
        mock_method = Mock()
        mock_method.apply.return_value = torch.empty(2, N)
        mock_method_cls.return_value = mock_method
        kernel = object.__new__(AscendFP8ScaledMMLinearKernel)
        kernel.mxfp8_method = mock_method
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.empty(K, N, dtype=torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.empty(N, 1, dtype=torch.float32), requires_grad=False)
        x = torch.randn(2, K, dtype=torch.bfloat16)

        out = kernel.apply_weights(layer, x)
        mock_method.apply.assert_called_once_with(layer, x, None)
        self.assertEqual(out.shape, (2, N))

    @patch("torch_npu.npu_dynamic_mx_quant")
    @patch("torch_npu.npu_format_cast", side_effect=lambda w, fmt, **kwargs: w.clone())
    @patch("vllm_ascend.utils._should_trans_nz", return_value=False)
    def test_process_weights_requantizes_ptpc_to_mxfp8(
        self, mock_should_trans_nz, mock_format_cast, mock_mx_quant
    ):
        """PTPC [K,N] fp8 + [N,1] fp32 scale must become MXFP8 weights."""
        weight = torch.randn(K, N, dtype=torch.bfloat16)
        scale = torch.randn(N, 1, dtype=torch.bfloat16).abs() + 0.1
        dense = (weight.float() * scale.float()).to(torch.bfloat16)  # [K, N]

        def fake_mx_quant(x, dst_type, scale_alg):
            packed = torch.randn(x.shape[0], x.shape[1] // 2, 2, dtype=torch.float8_e8m0fnu)
            return x.to(dst_type), packed

        mock_mx_quant.side_effect = fake_mx_quant
        kernel = object.__new__(AscendFP8ScaledMMLinearKernel)
        kernel.config = SimpleNamespace(input_dtype=torch.bfloat16)
        kernel.layer_param_names = ["weight", "weight_scale"]
        kernel.mxfp8_method = Mock(
            dynamic_mx_quant_scale_alg=0,
            process_weights_after_loading=Mock(),
        )

        layer = nn.Module()
        layer.weight = nn.Parameter(weight.to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(scale.to(torch.float32), requires_grad=False)

        kernel.process_weights_after_loading(layer)

        called_x = mock_mx_quant.call_args.args[0]
        self.assertEqual(called_x.shape, (N, K))
        torch.testing.assert_close(
            called_x.float(),
            dense.T.float(),
            rtol=0,
            atol=0.02,
        )
        self.assertEqual(layer.weight_scale.shape, (N, K // _MXFP8_GROUP_SIZE))
        kernel.mxfp8_method.process_weights_after_loading.assert_called_once_with(layer)


class TestAscendFP8ScaledMMInt8Kernel(TestBase):
    """The non-MXFP8 execution path for PTPC draft lm_head copies (A3/A2)."""

    @patch(_PROFILE_PATH)
    def test_is_supported_only_on_out_of_tree_platform(self, mock_profile):
        mock_profile.return_value = make_profile(mxfp8=False, backend=QuantizationBackendFamily.STANDARD)
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
        ) as mock_platform:
            mock_platform.is_out_of_tree.return_value = True
            ok, _ = AscendFP8ScaledMMInt8LinearKernel.is_supported()
            self.assertTrue(ok)

            mock_platform.is_out_of_tree.return_value = False
            ok, reason = AscendFP8ScaledMMInt8LinearKernel.is_supported()
            self.assertFalse(ok)
            self.assertIn("out-of-tree", reason)

    @patch(_PROFILE_PATH)
    def test_is_supported_not_on_mxfp8_hardware(self, mock_profile):
        """The MXFP8 kernel takes precedence where the capability exists."""
        mock_profile.return_value = make_profile(mxfp8=True, backend=QuantizationBackendFamily.STANDARD)
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
        ) as mock_platform:
            mock_platform.is_out_of_tree.return_value = True
            ok, reason = AscendFP8ScaledMMInt8LinearKernel.is_supported()
        self.assertFalse(ok)
        self.assertIn("MXFP8", reason)

    @patch(_PROFILE_PATH)
    def test_is_supported_rejects_compatibility_backend(self, mock_profile):
        mock_profile.return_value = make_profile(
            mxfp8=False, backend=QuantizationBackendFamily.COMPATIBILITY
        )
        with patch(
            "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
        ) as mock_platform:
            mock_platform.is_out_of_tree.return_value = True
            ok, reason = AscendFP8ScaledMMInt8LinearKernel.is_supported()
        self.assertFalse(ok)
        self.assertIn("standard quantization backend", reason)

    def test_is_supported_gates_are_mutually_exclusive_with_mxfp8_kernel(self):
        """Whatever the hardware, exactly one of the two kernels is supported."""
        for mxfp8 in (False, True):
            profile = make_profile(mxfp8=mxfp8, backend=QuantizationBackendFamily.STANDARD)
            with patch(_PROFILE_PATH, return_value=profile), patch(
                "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.current_platform"
            ) as mock_platform:
                mock_platform.is_out_of_tree.return_value = True
                mxfp8_ok, _ = AscendFP8ScaledMMLinearKernel.is_supported()
                int8_ok, _ = AscendFP8ScaledMMInt8LinearKernel.is_supported()
            self.assertNotEqual(mxfp8_ok, int8_ok)
            self.assertEqual(mxfp8_ok, mxfp8)

    def test_can_implement_ptpc_quant_keys(self):
        ok, _ = AscendFP8ScaledMMInt8LinearKernel.can_implement(make_config())
        self.assertTrue(ok)

    def test_can_implement_allows_k_not_multiple_of_mxfp8_group_size(self):
        """INT8 per-channel quantization has no MX group-size constraint."""
        ok, _ = AscendFP8ScaledMMInt8LinearKernel.can_implement(
            make_config(weight_shape=(N, 100))
        )
        self.assertTrue(ok)

    def test_can_implement_rejects_other_weight_keys(self):
        config = make_config()
        config.weight_quant_key = None
        ok, reason = AscendFP8ScaledMMInt8LinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("per-channel", reason)

    def test_can_implement_rejects_other_activation_keys(self):
        config = make_config()
        config.activation_quant_key = None
        ok, reason = AscendFP8ScaledMMInt8LinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("per-token", reason)

    def test_can_implement_rejects_other_output_dtypes(self):
        config = make_config(out_dtype=torch.float32)
        ok, reason = AscendFP8ScaledMMInt8LinearKernel.can_implement(config)
        self.assertFalse(ok)
        self.assertIn("output dtype", reason)

    def test_constructs_int8_delegate(self):
        kernel = AscendFP8ScaledMMInt8LinearKernel(make_config(), ["weight", "weight_scale"])
        self.assertIsInstance(kernel.int8_method, AscendW8A8DynamicLinearMethod)

    def test_apply_scaled_mm_not_implemented(self):
        kernel = object.__new__(AscendFP8ScaledMMInt8LinearKernel)
        with self.assertRaisesRegex(NotImplementedError, "apply_weights"):
            kernel.apply_scaled_mm(
                A=torch.empty(1, K),
                B=torch.empty(K, N),
                out_dtype=torch.bfloat16,
                As=None,
                Bs=None,
                bias=None,
                output_shape=[1, N],
            )

    @patch("torch_npu.npu_quant_matmul")
    @patch(
        "vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm.AscendW8A8DynamicLinearMethod"
    )
    def test_apply_weights_delegates_to_int8_method(self, mock_method_cls, mock_matmul):
        mock_method = Mock()
        mock_method.apply.return_value = torch.empty(2, N)
        mock_method_cls.return_value = mock_method
        kernel = object.__new__(AscendFP8ScaledMMInt8LinearKernel)
        kernel.int8_method = mock_method
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.empty(K, N, dtype=torch.int8), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.empty(N, dtype=torch.float32), requires_grad=False)
        x = torch.randn(2, K, dtype=torch.bfloat16)

        out = kernel.apply_weights(layer, x)
        mock_method.apply.assert_called_once_with(layer, x, None)
        self.assertEqual(out.shape, (2, N))

    @patch("vllm_ascend.utils._should_trans_nz", return_value=False)
    def test_process_weights_requantizes_ptpc_to_int8(self, mock_should_trans_nz):
        """PTPC [K,N] fp8 + [N,1] fp32 scale must become W8A8 int8 weights."""
        weight = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)  # [K, N]
        scale = torch.full((2, 1), 2.0, dtype=torch.float32)  # [N, 1]

        kernel = object.__new__(AscendFP8ScaledMMInt8LinearKernel)
        kernel.config = SimpleNamespace(input_dtype=torch.bfloat16)
        kernel.layer_param_names = ["weight", "weight_scale"]
        kernel.int8_method = AscendW8A8DynamicLinearMethod()

        layer = nn.Module()
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = nn.Parameter(scale, requires_grad=False)

        kernel.process_weights_after_loading(layer)

        # Expected per-channel symmetric requantization of the resolved [N, K].
        dense = weight.t().float() * scale.float()
        amax = dense.abs().amax(dim=1, keepdim=True)
        exp_scale = (amax / _INT8_MAX).clamp_min(_MIN_CHANNEL_SCALE).to(torch.bfloat16)
        exp_q = (
            torch.round(dense / exp_scale.float())
            .clamp_(_INT8_MIN, _INT8_MAX)
            .to(torch.int8)
        )
        # The W8A8 transform transposes the weight to [K, N] and flattens the
        # scales/offset to [N]; it also derives the fp32 scale copy.
        self.assertEqual(layer.weight.data.shape, (4, 2))
        self.assertEqual(layer.weight.data.dtype, torch.int8)
        self.assertTrue(torch.equal(layer.weight.data, exp_q.t().contiguous()))
        self.assertTrue(torch.equal(layer.weight_scale.data, exp_scale.flatten()))
        self.assertTrue(
            torch.equal(layer.weight_offset.data, torch.zeros(2, dtype=torch.bfloat16))
        )
        self.assertEqual(layer.weight_scale_fp32.dtype, torch.float32)


class TestFP8LmHeadKernelRegistration(TestBase):
    """The registration path that lets upstream kernel selection find us.

    Without it, init_fp8_linear_kernel finds no OOT entry and
    Fp8PtpcOnlineLinearMethod fails with ValueError.
    """

    def test_register_calls_upstream_oot_api(self):
        with patch(
            "vllm.model_executor.kernels.linear.register_linear_kernel"
        ) as mock_register:
            ok = register_ascend_fp8_lm_head_kernel()
        self.assertTrue(ok)
        self.assertEqual(
            mock_register.call_args_list,
            [
                call(
                    AscendFP8ScaledMMLinearKernel,
                    PlatformEnum.OOT,
                    kernel_type="fp8",
                ),
                call(
                    AscendFP8ScaledMMInt8LinearKernel,
                    PlatformEnum.OOT,
                    kernel_type="fp8",
                ),
            ],
        )

    def test_registration_populates_oot_table_and_marks_batch_invariant(self):
        ok = register_ascend_fp8_lm_head_kernel()
        self.assertTrue(ok)

        import vllm.model_executor.kernels.linear as kernels_linear

        self.assertIn(
            AscendFP8ScaledMMLinearKernel,
            kernels_linear._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT],
        )
        self.assertIn(
            AscendFP8ScaledMMInt8LinearKernel,
            kernels_linear._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT],
        )
        if not hasattr(kernels_linear, "fp8_scaled_mm_is_batch_invariant"):
            self.skipTest("vLLM build has no fp8_scaled_mm_is_batch_invariant")
        for kernel_cls in (AscendFP8ScaledMMLinearKernel, AscendFP8ScaledMMInt8LinearKernel):
            kernel = object.__new__(kernel_cls)
            self.assertTrue(kernels_linear.fp8_scaled_mm_is_batch_invariant(kernel))
