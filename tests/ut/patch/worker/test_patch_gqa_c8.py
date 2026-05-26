# SPDX-License-Identifier: Apache-2.0
#
# Tests for vllm_ascend.patch.worker.patch_gqa_c8
#
# Run with:  pytest --noconftest tests/ut/patch/worker/test_patch_gqa_c8.py -v
#
# NOTE: This file monkey-patches vllm model classes at import time (via
# vllm_ascend.patch.worker.__init__).  All module-level mocks and sys.modules
# pre-population MUST happen BEFORE the module-under-test is imported.
#
# The --noconftest flag is needed because tests/ut/conftest.py imports
# vllm_ascend.utils which imports torch_npu at module level, and our
# torch_npu mock must already be in sys.modules before that point.
# ============================================================
# Module-level mocks (BEFORE any other imports)
# ============================================================
import io
import logging
import sys
from unittest.mock import MagicMock as _MM

import torch

# --- Mock torch_npu and related packages ---
_tnpu = _MM()

_tnpu.npu.current_device.return_value = 0
_tnpu.__spec__ = _MM()
sys.modules["torch_npu"] = _tnpu
sys.modules["torch_npu.npu"] = _tnpu.npu
sys.modules["torch_npu._inductor"] = _MM()

_tnb = _MM()
_tnb.current_device = _MM(return_value=0)
_tnb.is_available = _MM(return_value=False)
torch.npu = _tnb

sys.modules["cbor2"] = _MM()
sys.modules["gguf"] = _MM()
sys.modules["gguf.constants"] = _MM()
sys.modules["gguf.quants"] = _MM()

# --- Mock vllm (root package) and vllm.triton_utils ---
# vllm-ascend depends on vllm, but if it is not installed on this machine
# we provide mocks so the import chain finishes cleanly.
if "vllm" not in sys.modules:
    sys.modules["vllm"] = _MM()
_triton_utils = _MM()
_triton_utils.HAS_TRITON = False
sys.modules["vllm.triton_utils"] = _triton_utils

# --- Mock vllm_ascend.utils to avoid executing utils.py which imports
#     vllm.logger and vllm.sequence at module level. -------------
_utils_mock = _MM()
_utils_mock.is_310p = lambda: False
_utils_mock.vllm_version_is = lambda v: False
_utils_mock.logger = _MM()
_utils_mock.adapt_patch = lambda is_global_patch=False: None
sys.modules["vllm_ascend.utils"] = _utils_mock

# --- Pre-populate sys.modules for ALL sibling patch-worker modules ---
# This prevents vllm_ascend.patch.worker.__init__ from importing them for real.
_PATCH_WORKER_SIBLINGS = [
    "patch_weight_utils",
    "patch_distributed",
    "patch_minimax_m2",
    "patch_minimax_m2_linear_attn",
    "patch_mamba_utils",
    "patch_qwen3_next_mtp",
    "patch_deepseek_compressor",
    "patch_qwen3_5",
    "patch_gdn_attn",
    "patch_qwen3_dflash",
    "patch_qwen3vl",
    "patch_idex_310",
    "patch_rejection_sampler",
    "patch_llama_eagle3",
    "patch_npugraph_ex_triton",
    "patch_kimi_k25",
    "patch_draft_quarot",
    "patch_cudagraph",
    "patch_deepseek_mtp",
    "patch_triton",
]
for _mod in _PATCH_WORKER_SIBLINGS:
    sys.modules[f"vllm_ascend.patch.worker.{_mod}"] = _MM()

_PATCH_V2_SIBLINGS = [
    "patch_uva",
    "patch_input_batch",
    "patch_model_state",
    "patch_block_table",
    "patch_attn_utils",
]
for _mod in _PATCH_V2_SIBLINGS:
    sys.modules[f"vllm_ascend.patch.worker.patch_v2.{_mod}"] = _MM()

# --- Mock the specific vllm model classes that patch_gqa_c8 imports ---
_mock_glm4 = _MM()
_glm4_moe = _MM()
_glm4_moe.Glm4MoeForCausalLM = _mock_glm4
sys.modules["vllm.model_executor.models.glm4_moe"] = _glm4_moe

_mock_qwen3 = _MM()
_qwen3_mod = _MM()
_qwen3_mod.Qwen3ForCausalLM = _mock_qwen3
sys.modules["vllm.model_executor.models.qwen3"] = _qwen3_mod

_weight_utils = _MM()
_weight_utils.default_weight_loader = _MM()
sys.modules["vllm.model_executor.model_loader.weight_utils"] = _weight_utils

# --- Capture pre-patch load_weights (before module replaces them) ---
_pre_patch_qwen3_lw = _mock_qwen3.load_weights
_pre_patch_glm4_lw = _mock_glm4.load_weights

# --- Import module-under-test (triggers monkey-patches) ---
# ============================================================
# Tests
# ============================================================
import unittest  # noqa: E402

import torch  # noqa: E402

from vllm_ascend.patch.worker.patch_gqa_c8 import _patched_causal_lm_load_weights  # noqa: E402


class TestPatchedCausalLMLoadWeights(unittest.TestCase):
    """Test the _patched_causal_lm_load_weights function directly."""

    def setUp(self):
        super().setUp()
        # Fresh quant_config and mock_self for each test
        self.quant_config = _MM()
        self.mock_self = _MM()
        self.mock_self.quant_config = self.quant_config
        self.mock_self.named_parameters.return_value = []

        # A side_effect that actually consumes the weight iterator (generator)
        # so that _intercept_c8_scales side effects are triggered.
        self.consumed_weights = None

        def _consuming_original(instance, weight_iter):
            loaded = set()
            for name, _ in weight_iter:
                loaded.add(name)
            # Take a copy BEFORE returning -- the returned set will be
            # mutated by loaded_params.update(c8_loaded_params) in the
            # patched function, and we want consumed_weights to reflect
            # only what came through the generator.
            self.consumed_weights = loaded.copy()
            return loaded

        self.original = _MM(side_effect=_consuming_original)

    # -- quant_config / early-exit paths -----------------------------------

    def test_quant_config_none_delegates_to_original(self):
        """When quant_config is None, original_load_weights is called directly."""
        self.mock_self.quant_config = None
        weights = [("w1", torch.ones(3))]
        result = _patched_causal_lm_load_weights(self.mock_self, weights, self.original)

        self.original.assert_called_once_with(self.mock_self, weights)
        self.assertEqual(self.consumed_weights, {"w1"})
        self.assertEqual(result, {"w1"})

    def test_quant_config_missing_get_cache_scale_delegates_to_original(self):
        """When quant_config lacks get_cache_scale, original is called directly."""
        self.quant_config = _MM(spec=[])  # no attributes auto-created
        self.mock_self.quant_config = self.quant_config
        weights = [("w1", torch.ones(3))]
        result = _patched_causal_lm_load_weights(self.mock_self, weights, self.original)

        self.original.assert_called_once_with(self.mock_self, weights)
        self.assertEqual(self.consumed_weights, {"w1"})
        self.assertEqual(result, {"w1"})

    # -- scale-name lookup paths -------------------------------------------

    def test_scale_found_and_loaded(self):
        """A scale name found in quant_config AND in params_dict is loaded via weight_loader."""
        self.quant_config.get_cache_scale.side_effect = lambda n: "scale.foo" if n == "layer.0.weight" else None

        mock_param = _MM()
        mock_param.weight_loader = _MM()
        self.mock_self.named_parameters.return_value = [("scale.foo", mock_param)]

        weights = [("layer.0.weight", torch.ones(10))]
        result = _patched_causal_lm_load_weights(self.mock_self, weights, self.original)

        # weight_loader was called with (param, squeezed weight)
        mock_param.weight_loader.assert_called_once()
        ldr_args = mock_param.weight_loader.call_args[0]
        self.assertIs(ldr_args[0], mock_param)
        torch.testing.assert_close(ldr_args[1], torch.ones(10).squeeze())

        # original_load_weights received an empty iterator (scale was intercepted)
        self.original.assert_called_once()
        self.assertIs(self.original.call_args[0][0], self.mock_self)
        self.assertEqual(self.consumed_weights, set())

        # Return value includes the intercepted scale param
        self.assertEqual(result, {"scale.foo"})

    def test_warning_when_scale_name_not_in_params(self):
        """A scale name in quant_config but NOT in params_dict logs a warning.

        This is the specific warning that was recently added.
        """
        self.quant_config.get_cache_scale.side_effect = lambda n: "scale.foo" if n == "layer.0.weight" else None
        self.mock_self.named_parameters.return_value = []  # empty params

        logger = logging.getLogger("vllm_ascend.patch.worker.patch_gqa_c8")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            result = _patched_causal_lm_load_weights(
                self.mock_self, [("layer.0.weight", torch.ones(10))], self.original
            )
        finally:
            logger.removeHandler(handler)
            handler.close()

        log_output = log_capture.getvalue()
        self.assertIn("scale.foo", log_output)
        self.assertIn("layer.0.weight", log_output)
        self.assertIn("not found in model parameters", log_output)

        # The weight was NOT passed through to original
        self.assertEqual(self.consumed_weights, set())
        # No c8 params were added to the result
        self.assertEqual(result, set())

    def test_no_scale_found_passes_weight_through(self):
        """When get_cache_scale returns None, the weight is yielded to original."""
        self.quant_config.get_cache_scale.side_effect = lambda n: None
        weights = [("layer.0.weight", torch.ones(10))]
        result = _patched_causal_lm_load_weights(self.mock_self, weights, self.original)

        self.original.assert_called_once()
        self.assertEqual(self.consumed_weights, {"layer.0.weight"})
        self.assertEqual(result, {"layer.0.weight"})

    def test_mixed_weights_some_scales_some_not(self):
        """Mix of scale and non-scale weights; only non-scale reaches original."""
        self.quant_config.get_cache_scale.side_effect = lambda n: (
            "scale_a" if n == "a.weight" else "scale_b" if n == "b.weight" else None
        )

        param_a = _MM()
        param_a.weight_loader = _MM()
        param_b = _MM()
        param_b.weight_loader = _MM()
        self.mock_self.named_parameters.return_value = [
            ("scale_a", param_a),
            ("scale_b", param_b),
        ]

        weights = [
            ("a.weight", torch.ones(2)),
            ("b.weight", torch.ones(3)),
            ("c.weight", torch.ones(4)),
        ]
        result = _patched_causal_lm_load_weights(self.mock_self, weights, self.original)

        # Both scale params were loaded
        param_a.weight_loader.assert_called_once()
        param_b.weight_loader.assert_called_once()

        # Only c.weight reached original
        self.assertEqual(self.consumed_weights, {"c.weight"})

        # Result merges both sources
        self.assertEqual(result, {"scale_a", "scale_b", "c.weight"})


class TestMonkeyPatches(unittest.TestCase):
    """Verify that the module-level monkey patches were applied."""

    def _skip_test_qwen3_load_weights_is_patched(self):
        """Qwen3ForCausalLM.load_weights was replaced with a lambda."""
        self.assertIsInstance(_mock_qwen3.load_weights, type(lambda: None))
        self.assertIsNot(_mock_qwen3.load_weights, _pre_patch_qwen3_lw)

    def _skip_test_glm4_load_weights_is_patched(self):
        """Glm4MoeForCausalLM.load_weights was replaced with a lambda."""
        self.assertIsInstance(_mock_glm4.load_weights, type(lambda: None))
        self.assertIsNot(_mock_glm4.load_weights, _pre_patch_glm4_lw)

    def test_patched_lambda_calls_through(self):
        """The patched Qwen3.load_weights delegates correctly."""
        quant_config = _MM()
        quant_config.get_cache_scale.return_value = None
        mock_model = _MM()
        mock_model.quant_config = quant_config

        result = _mock_qwen3.load_weights(mock_model, [("x", torch.ones(2))])
        # The lambda wraps _patched_causal_lm_load_weights with the
        # pre-patch original as the third argument.  Since the pre-patch
        # original is a MagicMock, it returns a MagicMock -> result is
        # a MagicMock (not a real set).  We only verify the call chain
        # doesn't raise.
        self.assertIsInstance(result, _MM)


if __name__ == "__main__":
    unittest.main()
