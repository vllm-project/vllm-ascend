# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the HIF8 C8 attention scheme (hif8_c8.py).

The scheme's whole contract is the checkpoint interface: static per-tensor
scales loaded via a custom weight loader, processed into descales and
reciprocals for the SFA backend, plus startup validation for
``--kv-cache-dtype hif8``. Everything here runs on plain CPU tensors.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase

_HIF8_C8_MODULE = "vllm_ascend.quantization.methods.kv_cache.hif8_c8"


class TestHif8QuantScaleWeightLoader(TestBase):
    """``_hif8_quant_scale_weight_loader`` enforces the per-tensor recipe."""

    def setUp(self):
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import _hif8_quant_scale_weight_loader

        self.loader = _hif8_quant_scale_weight_loader

    def test_bare_scalar_loads(self):
        param = nn.Parameter(torch.full((1,), float("nan")), requires_grad=False)
        self.loader(param, torch.tensor([0.125]))
        self.assertEqual(param.data.item(), 0.125)

    def test_uniform_per_head_vector_takes_first_element(self):
        # ModelSlim can export the per-tensor scale replicated across heads.
        # Attention-module parameters bypass ColumnParallelLinear sharding, so
        # every TP rank receives the full tensor and keeps the same first
        # element; the scale stays replicated across ranks.
        param = nn.Parameter(torch.full((1,), float("nan")), requires_grad=False)
        self.loader(param, torch.full((32,), 0.5))
        self.assertEqual(param.data.item(), 0.5)

    def test_non_uniform_scales_rejected(self):
        # Per-head scales are not a QK_HIF8_PER_TENSOR checkpoint; the kernel
        # has no per-head input, so reject at load time instead of quantizing
        # every head with the first head's scale.
        param = nn.Parameter(torch.ones(1), requires_grad=False)
        with self.assertRaises(ValueError) as ctx:
            self.loader(param, torch.tensor([1.0, 2.0, 1.0]))
        self.assertIn("per-tensor", str(ctx.exception))

    def test_empty_scale_rejected(self):
        param = nn.Parameter(torch.ones(1), requires_grad=False)
        with self.assertRaises(ValueError):
            self.loader(param, torch.empty(0))


class TestAscendHIF8KVCacheAttentionMethod(TestBase):
    """create_weights / process_weights_after_loading lifecycle on CPU."""

    def setUp(self):
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import AscendHIF8KVCacheAttentionMethod

        self.method = AscendHIF8KVCacheAttentionMethod()

    def test_create_weights_nan_sentinels(self):
        # The scale starts as NaN so a checkpoint that never provides it
        # cannot silently quantize with a valid-looking scale: the NaN flows
        # into the reciprocals the backend consumes.
        layer = nn.Module()
        self.method.create_weights(layer)
        for name in ("fa_q", "fa_k"):
            scale = getattr(layer, name).scale
            self.assertIsInstance(scale, nn.Parameter)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(tuple(scale.shape), (1,))
            self.assertTrue(torch.isnan(scale.data).all(), name)

    def test_create_weights_attaches_loader(self):
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import _hif8_quant_scale_weight_loader

        layer = nn.Module()
        self.method.create_weights(layer)
        self.assertIs(layer.fa_q.scale.weight_loader, _hif8_quant_scale_weight_loader)
        self.assertIs(layer.fa_k.scale.weight_loader, _hif8_quant_scale_weight_loader)

    def test_process_weights_after_loading_builds_reciprocals(self):
        layer = nn.Module()
        self.method.create_weights(layer)
        with torch.no_grad():
            layer.fa_q.scale.copy_(torch.tensor([2.0]))
            layer.fa_k.scale.copy_(torch.tensor([8.0]))
        self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.hif8_q_descale.item(), 2.0)
        self.assertEqual(layer.hif8_kv_descale.item(), 8.0)
        self.assertEqual(layer.hif8_q_scale_reciprocal.item(), 0.5)
        self.assertEqual(layer.hif8_kv_scale_reciprocal.item(), 0.125)
        for name in (
            "hif8_q_descale",
            "hif8_kv_descale",
            "hif8_q_scale_reciprocal",
            "hif8_kv_scale_reciprocal",
        ):
            self.assertIsInstance(getattr(layer, name), nn.Parameter, name)
        self.assertTrue(layer.hif8_quant_layer)

    def test_loaded_checkpoint_flows_to_backend_scales(self):
        """create -> loader -> process produces exactly what the impl reads."""
        layer = nn.Module()
        self.method.create_weights(layer)
        with torch.no_grad():
            layer.fa_q.scale.weight_loader(layer.fa_q.scale, torch.tensor([0.5]))
            # ModelSlim's replicated per-head export shape takes the same path.
            layer.fa_k.scale.weight_loader(layer.fa_k.scale, torch.full((48,), 0.5))
        self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.hif8_q_scale_reciprocal.item(), 2.0)
        self.assertEqual(layer.hif8_kv_scale_reciprocal.item(), 2.0)


class TestValidateSparseSfaHif8(TestBase):
    """Startup guards for ``--kv-cache-dtype hif8``."""

    @staticmethod
    def _vllm_config(pcp=1, dcp=1):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp)
        )

    @staticmethod
    def _profile(supported):
        return SimpleNamespace(supports_weight_format=lambda fmt: supported)

    def test_rejects_hardware_without_hif8(self):
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import validate_sparse_sfa_hif8

        with (
            patch(f"{_HIF8_C8_MODULE}.get_current_hardware_profile", return_value=self._profile(False)),
            self.assertRaises(ValueError) as ctx,
        ):
            validate_sparse_sfa_hif8(self._vllm_config(), [3])
        self.assertIn("950", str(ctx.exception))

    def test_rejects_dtype_without_recipes(self):
        # --kv-cache-dtype hif8 on a checkpoint without QK_HIF8_PER_TENSOR
        # layer recipes would run BF16 KV through the quantized kernel path.
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import validate_sparse_sfa_hif8

        with (
            patch(f"{_HIF8_C8_MODULE}.get_current_hardware_profile", return_value=self._profile(True)),
            self.assertRaises(ValueError) as ctx,
        ):
            validate_sparse_sfa_hif8(self._vllm_config(), [])
        self.assertIn("recipes", str(ctx.exception))

    def test_rejects_context_parallel(self):
        # PCP/DCP gather flows would have to move the packed HIF8 payload.
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import validate_sparse_sfa_hif8

        with patch(f"{_HIF8_C8_MODULE}.get_current_hardware_profile", return_value=self._profile(True)):
            with self.assertRaises(ValueError) as ctx:
                validate_sparse_sfa_hif8(self._vllm_config(pcp=2), [3])
            with self.assertRaises(ValueError):
                validate_sparse_sfa_hif8(self._vllm_config(dcp=2), [3])
        self.assertIn("context parallel", str(ctx.exception))

    def test_accepts_950_with_recipes(self):
        from vllm_ascend.quantization.methods.kv_cache.hif8_c8 import validate_sparse_sfa_hif8

        with patch(f"{_HIF8_C8_MODULE}.get_current_hardware_profile", return_value=self._profile(True)):
            validate_sparse_sfa_hif8(self._vllm_config(), [3, 5])
