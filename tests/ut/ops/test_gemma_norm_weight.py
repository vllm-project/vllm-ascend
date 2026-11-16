# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import operator
from unittest import TestCase
from unittest.mock import patch

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, get_gemma_rms_norm_weight


class TestGemmaNormWeight(TestCase):
    def setUp(self):
        # Test the real parameter/buffer lifecycle without selecting NPU kernels.
        self.dispatch = patch.object(GemmaRMSNorm, "dispatch_forward", return_value=lambda x: x)
        self.dispatch.start()
        self.addCleanup(self.dispatch.stop)

    def make_norm(self, dtype=torch.bfloat16):
        return AscendGemmaRMSNorm(8).to(dtype=dtype)

    def test_unprepared_weight_uses_original_addition(self):
        norm = self.make_norm()
        with torch.no_grad():
            norm.weight.copy_(torch.arange(8))
            self.assertFalse(norm._effective_weight_ready)
            torch.testing.assert_close(get_gemma_rms_norm_weight(norm), 1 + norm.weight, rtol=0, atol=0)

    def test_loaded_weight_rounding_and_reload_preserve_storage(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype), torch.inference_mode():
                norm = self.make_norm(dtype)
                for weight in (torch.linspace(-1.2, 0.7, 8), torch.linspace(-0.3, 1.2, 8)):
                    old = norm._effective_weight
                    norm.weight.weight_loader(norm.weight, weight)
                    expected = 1.0 + weight.to(dtype)
                    actual = get_gemma_rms_norm_weight(norm)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(norm.weight, weight.to(dtype), rtol=0, atol=0)
                    self.assertIs(actual, old)
                    self.assertIs(actual, norm.effective_weight)

    def test_reload_outside_original_inference_context(self):
        norm = self.make_norm()
        with torch.inference_mode():
            norm.weight.weight_loader(norm.weight, torch.ones(8))
        cached = norm._effective_weight
        norm.weight.weight_loader(norm.weight, torch.full((8,), 2.0))
        self.assertIs(norm._effective_weight, cached)
        torch.testing.assert_close(cached, torch.full_like(cached, 3.0), rtol=0, atol=0)

    def test_native_layerwise_reload_refreshes_original_cache_storage(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for prepared in (False, True):
                with self.subTest(dtype=dtype, prepared=prepared), torch.inference_mode():
                    norm = self.make_norm(dtype)
                    model = torch.nn.Sequential(norm)
                    # Record the constructor state, as the model loader does.
                    record_metadata_for_reloading(model)
                    initial = torch.full_like(norm.weight, 0.25)
                    if prepared:
                        norm.weight.weight_loader(norm.weight, initial)
                    else:
                        norm.weight.copy_(initial)
                    original_weight = norm.weight
                    original_cache = norm._effective_weight
                    address = original_cache.data_ptr()
                    for value in (None, 0.75, -0.5, None):
                        expected_weight = norm.weight.clone()
                        if value is not None:
                            expected_weight.fill_(value)
                        initialize_layerwise_reload(model)
                        if value is not None:
                            norm.weight.weight_loader(norm.weight, expected_weight)
                        finalize_layerwise_reload(model, model_config=None)
                        self.assertIs(norm.weight, original_weight)
                        self.assertIs(norm._effective_weight, original_cache)
                        self.assertEqual(original_cache.data_ptr(), address)
                        torch.testing.assert_close(norm.weight, expected_weight, rtol=0, atol=0)
                        torch.testing.assert_close(
                            get_gemma_rms_norm_weight(norm), 1.0 + expected_weight, rtol=0, atol=0
                        )
                        self.assertEqual(set(norm.state_dict()), {"weight"})
                        self.assertIn("_effective_weight", norm._non_persistent_buffers_set)

    def test_unprepared_receiver_keeps_transfer_buffer_and_runtime_fallback(self):
        sender, receiver = self.make_norm(), self.make_norm()
        sender.weight.weight_loader(sender.weight, torch.full((8,), 0.5))
        self.assertEqual(dict(sender.named_buffers()).keys(), dict(receiver.named_buffers()).keys())
        # A parameter/buffer transfer does not invoke the checkpoint loader.
        with torch.inference_mode():
            receiver.weight.copy_(sender.weight)
            receiver._effective_weight.copy_(sender._effective_weight)
            self.assertFalse(receiver._effective_weight_ready)
            torch.testing.assert_close(receiver.effective_weight, 1.0 + receiver.weight, rtol=0, atol=0)

    def test_scalar_checkpoint_weight(self):
        norm = AscendGemmaRMSNorm(1)
        norm.weight.weight_loader(norm.weight, torch.tensor(0.25))
        torch.testing.assert_close(norm._effective_weight, torch.tensor([1.25]), rtol=0, atol=0)

    def test_state_dict_contains_raw_weight_only_and_refreshes_on_assign(self):
        norm = self.make_norm()
        for assign in (False, True):
            with self.subTest(assign=assign):
                norm.load_state_dict({"weight": torch.full((8,), 0.25, dtype=torch.bfloat16)}, assign=assign)
                self.assertEqual(set(norm.state_dict()), {"weight"})
                torch.testing.assert_close(norm.state_dict()["weight"], torch.full_like(norm.weight, 0.25))
                torch.testing.assert_close(norm._effective_weight, 1 + norm.weight, rtol=0, atol=0)
                norm.weight.weight_loader(norm.weight, torch.zeros(8))
                torch.testing.assert_close(norm._effective_weight, torch.ones_like(norm.weight), rtol=0, atol=0)

    def test_dtype_conversion_recomputes_in_destination_precision(self):
        norm = self.make_norm(torch.float32)
        weight = torch.tensor([0.0039, 0.0078, 0.0156, 0.0312, -0.0039, -0.0078, -0.0156, -0.0312])
        norm.weight.weight_loader(norm.weight, weight)
        norm.bfloat16()
        torch.testing.assert_close(norm._effective_weight, 1.0 + weight.bfloat16(), rtol=0, atol=0)
        rounded = norm._effective_weight.float().clone()
        norm.float()
        torch.testing.assert_close(norm._effective_weight, 1.0 + weight.bfloat16().float(), rtol=0, atol=0)
        self.assertFalse(torch.equal(norm._effective_weight, rounded))
        norm.weight.weight_loader(norm.weight, torch.zeros(8))
        torch.testing.assert_close(norm._effective_weight, torch.ones(8), rtol=0, atol=0)

    def test_failed_weight_load_preserves_cache(self):
        norm = self.make_norm()
        norm.weight.weight_loader(norm.weight, torch.ones(8))
        cached = norm._effective_weight.clone()
        with self.assertRaises(AssertionError):
            norm.weight.weight_loader(norm.weight, torch.ones(7))
        torch.testing.assert_close(norm._effective_weight, cached, rtol=0, atol=0)

    def test_grad_enabled_retains_weight_gradient(self):
        norm = self.make_norm(torch.float32)
        norm.weight.weight_loader(norm.weight, torch.ones(8))
        norm.effective_weight.sum().backward()
        torch.testing.assert_close(norm.weight.grad, torch.ones(8))

    def test_unregistered_norm_fallback(self):
        # A fused consumer can also receive the unregistered upstream class.
        norm = torch.nn.Module()
        norm.register_parameter("weight", torch.nn.Parameter(torch.arange(8, dtype=torch.float32)))
        torch.testing.assert_close(get_gemma_rms_norm_weight(norm), 1.0 + norm.weight, rtol=0, atol=0)

    def test_residual_forward_and_fused_consumer_share_cache(self):
        norm = self.make_norm()
        norm.weight.weight_loader(norm.weight, torch.randn(8))
        for rows in (1, 4, 16, 128):
            with self.subTest(rows=rows), torch.inference_mode():
                x = torch.randn(rows, 8, dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                with patch("torch_npu.npu_add_rms_norm", create=True, return_value=(x, None, residual)) as op:
                    out, res = norm.forward_oot(x, residual)
                self.assertIs(out, x)
                self.assertIs(res, residual)
                self.assertIs(op.call_args.args[2], get_gemma_rms_norm_weight(norm))

    def test_no_residual_passes_raw_and_cached_weights_to_adaptor(self):
        norm = self.make_norm()
        norm.weight.weight_loader(norm.weight, torch.randn(8))
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        with (
            torch.inference_mode(),
            patch("vllm_ascend.ops.layernorm.DeviceOperator.npu_gemma_rms_norm", return_value=x) as op,
        ):
            self.assertIs(norm.forward_oot(x), x)
        self.assertIs(op.call_args.args[1], norm.weight)
        self.assertIs(op.call_args.kwargs["effective_weight"], norm._effective_weight)

    def test_compiled_consumer_observes_in_place_reload(self):
        norm = self.make_norm(torch.float32)
        norm.weight.weight_loader(norm.weight, torch.ones(8))
        captured = []

        def backend(graph, example_inputs):
            captured.append(graph)
            return graph.forward

        compiled = torch.compile(lambda x: x * get_gemma_rms_norm_weight(norm), backend=backend, fullgraph=True)
        x = torch.ones(2, 8)
        with torch.inference_mode():
            torch.testing.assert_close(compiled(x), x * 2)
            norm.weight.weight_loader(norm.weight, torch.full((8,), 2.0))
            torch.testing.assert_close(compiled(x), x * 3)
        self.assertEqual(len(captured), 1)
        self.assertFalse(
            any(node.target in (operator.add, torch.add, torch.Tensor.add) for node in captured[0].graph.nodes)
        )

    def test_compiled_consumer_observes_native_layerwise_reload(self):
        norm = self.make_norm(torch.float32)
        model = torch.nn.Sequential(norm)
        record_metadata_for_reloading(model)
        norm.weight.weight_loader(norm.weight, torch.full_like(norm.weight, 0.25))
        captured = []

        def backend(graph, example_inputs):
            captured.append(graph)
            return graph.forward

        compiled = torch.compile(lambda x: x * get_gemma_rms_norm_weight(norm), backend=backend, fullgraph=True)
        x = torch.ones(2, 8)
        with torch.inference_mode():
            torch.testing.assert_close(compiled(x), x * 1.25, rtol=0, atol=0)
            cached = norm.effective_weight
            for value in (0.75, -0.5):
                loaded = torch.full_like(norm.weight, value)
                initialize_layerwise_reload(model)
                norm.weight.weight_loader(norm.weight, loaded)
                finalize_layerwise_reload(model, model_config=None)
                self.assertIs(norm.effective_weight, cached)
                torch.testing.assert_close(compiled(x), x * (1 + value), rtol=0, atol=0)
        self.assertEqual(len(captured), 1)
        self.assertFalse(
            any(node.target in (operator.add, torch.add, torch.Tensor.add) for node in captured[0].graph.nodes)
        )
