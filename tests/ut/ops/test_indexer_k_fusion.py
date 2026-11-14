# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU validation tests; Triton import stubs do not execute kernel math."""

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_launcher(relative_path):
    # Only stub the compiler boundary, leaving the production Python launcher
    # and its real PyTorch shape/dtype/stride validation intact.
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = SimpleNamespace(
        jit=lambda fn: fn,
        autotune=lambda **kwargs: lambda fn: fn,
        Config=MagicMock(),
    )
    triton_utils.tl = SimpleNamespace(constexpr=object())
    spec = importlib.util.spec_from_file_location("_fusion_launcher_under_test", REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"vllm.triton_utils": triton_utils}):
        spec.loader.exec_module(module)
    return module


class TestIndexerKFusionValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_launcher("vllm_ascend/ops/triton/indexer_k_fusion.py")

    def setUp(self):
        self.args = dict(
            k_pre=torch.ones(3, 128, dtype=torch.bfloat16),
            weight=torch.ones(128, dtype=torch.bfloat16),
            eps=1e-6,
            cos=torch.ones(3, 64),
            sin=torch.zeros(3, 64),
            slots=torch.zeros(3, 2, dtype=torch.int32),
            k_cache=torch.zeros(2, 4, 1, 128, dtype=torch.int8),
            scale_cache=torch.zeros(2, 4, 1, 1, dtype=torch.float16),
        )

    def test_invalid_inputs_fail_before_launch_and_leave_cache_untouched(self):
        cases = [
            ("k_pre", torch.ones(3, 128), "BF16"),
            ("k_pre", torch.ones(3, 96, dtype=torch.bfloat16), "power of two"),
            ("k_pre", torch.ones(3, 256, dtype=torch.bfloat16)[:, ::2], "stride"),
            ("weight", torch.ones(127), "weight"),
            ("weight", torch.ones(128, dtype=torch.float16), "weight"),
            ("slots", torch.zeros(3, dtype=torch.int32), "slots"),
            ("slots", torch.zeros(3, 2), "slots"),
            ("cos", torch.ones(3, 64, dtype=torch.bfloat16), "FP32"),
            ("cos", torch.ones(2, 64), "cos/sin"),
            ("sin", torch.zeros(3, 32), "cos/sin"),
            ("k_cache", torch.zeros(2, 4, 2, 128, dtype=torch.int8), "k_cache"),
            ("k_cache", torch.zeros(2, 4, 1, 128), "INT8"),
            ("scale_cache", torch.zeros(2, 4, 1, 1), "FP16"),
            ("scale_cache", torch.zeros(2, 4, 1, dtype=torch.float16), "scale_cache"),
            ("k_cache", self.args["k_cache"][:1].expand(2, -1, -1, -1), "nonoverlapping"),
        ]
        for name, value, message in cases:
            with self.subTest(name=name, shape=value.shape):
                args = {**self.args, name: value}
                before = args["k_cache"].clone(), args["scale_cache"].clone()
                with self.assertRaisesRegex(ValueError, message):
                    self.module.indexer_k_norm_rope_int8_store(**args)
                torch.testing.assert_close(args["k_cache"], before[0], rtol=0, atol=0)
                torch.testing.assert_close(args["scale_cache"], before[1], rtol=0, atol=0)

    def test_invalid_epsilon(self):
        for eps in (0.0, -1e-6, float("inf"), float("nan")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "positive"):
                self.module.indexer_k_norm_rope_int8_store(**{**self.args, "eps": eps})

    def test_invalid_rotary_width(self):
        for width in (0, 3, 130):
            args = {**self.args, "cos": torch.ones(3, width), "sin": torch.zeros(3, width)}
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "R must"):
                self.module.indexer_k_norm_rope_int8_store(**args)

    def test_cpu_tensors_rejected_even_for_empty_batch(self):
        for tokens in (0, 3):
            for four_dimensional in (False, True):
                args = dict(self.args)
                for name in ("k_pre", "cos", "sin", "slots"):
                    args[name] = args[name][:tokens]
                if four_dimensional:
                    for name in ("cos", "sin"):
                        args[name] = args[name][:, None, None, :]
                with (
                    self.subTest(tokens=tokens, four_dimensional=four_dimensional),
                    self.assertRaisesRegex(ValueError, "one NPU"),
                ):
                    self.module.indexer_k_norm_rope_int8_store(**args)


class TestCompressorFusedNormValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_launcher("vllm_ascend/ops/triton/compressor/compressor_triton.py")

    def setUp(self):
        self.args = dict(
            kv=torch.empty(0, 128),
            scores=torch.empty(0, 128),
            state_cache=torch.zeros(2, 32, 256),
            metadata=torch.zeros(5, 1, dtype=torch.int32),
            out=torch.empty(0, 128, dtype=torch.bfloat16),
            max_query_len=0,
            num_cores=1,
        )

    def test_empty_input_preserves_output_and_state_storage(self):
        for dtype in (None, torch.bfloat16, torch.float32):
            weight = None if dtype is None else torch.ones(128, dtype=dtype)
            state = self.args["state_cache"].clone()
            actual = self.module.compressor_from_projected(**self.args, norm_w=weight)
            self.assertIs(actual, self.args["out"])
            torch.testing.assert_close(self.args["state_cache"], state, rtol=0, atol=0)

    def test_invalid_norm_weight_is_checked_on_empty_input(self):
        for weight in (torch.ones(127), torch.ones(128, dtype=torch.float16), torch.ones(256)[::2]):
            with (
                self.subTest(shape=weight.shape, dtype=weight.dtype),
                self.assertRaisesRegex(ValueError, "RMSNorm weight or epsilon"),
            ):
                self.module.compressor_from_projected(**self.args, norm_w=weight)

    def test_invalid_norm_epsilon(self):
        for eps in (0.0, -1e-6, float("inf"), float("nan")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "RMSNorm weight or epsilon"):
                self.module.compressor_from_projected(**self.args, norm_w=torch.ones(128), rms_eps=eps)


class TestCompressorModelNormDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Constructor/cache registration dependencies are outside this focused
        # test. Execute the actual pool_projected method on real CPU tensors.
        names = (
            "vllm.model_executor.layers.layernorm",
            "vllm.v1.kv_cache_interface",
            "vllm_ascend.attention.dsa_v41",
            "vllm_ascend.core.kv_cache_interface",
            "vllm_ascend.models.deepseek_v41.cache_config",
        )
        with patch.dict("sys.modules", {name: MagicMock() for name in names}):
            cls.module = _load_launcher("vllm_ascend/models/deepseek_v41/compressor.py")

    def test_bias_fallback_and_fused_output_are_each_normalized_once(self):
        for bias_loaded in (False, True):
            with self.subTest(bias_loaded=bias_loaded):
                raw = torch.ones(2, 128, dtype=torch.bfloat16)
                normalized = torch.full_like(raw, 3)
                norm = MagicMock(
                    return_value=normalized,
                    weight=torch.ones(128, dtype=torch.bfloat16),
                    variance_epsilon=1e-6,
                    bias_loaded=bias_loaded,
                )
                pooled = torch.empty(4, 128, dtype=torch.bfloat16)
                state = torch.zeros(2, 32, 1, 256)
                compressor = SimpleNamespace(
                    norm=norm,
                    state_cache=SimpleNamespace(kv_cache=[state]),
                    _ring_pooled=pooled,
                    _ring_num_cores=8,
                )
                metadata = SimpleNamespace(c2_ring_metadata=torch.zeros(5, 1, dtype=torch.int32), max_query_len=2)
                kernel_module = ModuleType("vllm_ascend.ops.triton.compressor.compressor_triton")
                kernel_module.compressor_from_projected = MagicMock(return_value=raw if bias_loaded else normalized)
                with patch.dict("sys.modules", {kernel_module.__name__: kernel_module}):
                    result = self.module.DeepseekV41Compressor.pool_projected(
                        compressor,
                        torch.ones(2, 128),
                        torch.zeros(2, 128),
                        metadata,
                    )
                self.assertIs(result, normalized)
                call = kernel_module.compressor_from_projected.call_args
                self.assertIs(call.kwargs["norm_w"], None if bias_loaded else norm.weight)
                self.assertEqual(call.kwargs["rms_eps"], norm.variance_epsilon)
                self.assertEqual(call.args[2].data_ptr(), state.data_ptr())
                self.assertEqual(call.args[4].data_ptr(), pooled.data_ptr())
                self.assertEqual(norm.call_count, int(bias_loaded))
                if bias_loaded:
                    norm.assert_called_once_with(raw)
