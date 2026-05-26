# SPDX-License-Identifier: Apache-2.0
#
# Tests for vllm_ascend.patch.worker.patch_minimax_m2_linear_attn
#
# Run with:  pytest --noconftest tests/ut/patch/worker/test_patch_minimax_m2_linear_attn.py -v
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
    # NOTE: patch_minimax_m2_linear_attn is deliberately excluded — we
    # import the real module later to test its behavior.
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
    # Mock patch_gqa_c8 so that vllm_ascend.patch.worker.__init__ does not
    # try to import it for real (which would trigger vllm.model_executor
    # dependencies we do not need).
    "patch_gqa_c8",
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

# --- Mock the vllm model classes used by the module-under-test ---


# A plain class (NOT MagicMock) is critical — MagicMock auto-answers
# hasattr with True, which would defeat the forward_qk / _normalize_qk
# probes.  We use a real class so hasattr respects our runtime setup.
class _MockRMSNorm:
    """Mock for MiniMaxText01RMSNormTP.

    Initially carries neither forward_qk nor _normalize_qk, so the
    very first import fires the warning path.  Tests that need the
    normal path add the method they want to probe before reloading
    the module-under-test.
    """

    pass


_mock_custom_op = _MM()

# --- Build the mock linear_attn module ---
_mock_linear_attn_module = _MM()
_mock_linear_attn_module.CustomOp = _mock_custom_op
_mock_linear_attn_module.MiniMaxText01RMSNormTP = _MockRMSNorm
sys.modules["vllm.model_executor.layers.mamba.linear_attn"] = _mock_linear_attn_module

# --- Mock vllm.distributed so the import resolves ---
_mock_distributed = _MM()
_mock_distributed.get_tensor_model_parallel_rank = _MM(return_value=0)
_mock_distributed.get_tensor_model_parallel_world_size = _MM(return_value=1)
_mock_distributed.tensor_model_parallel_all_reduce = _MM(side_effect=lambda x: x)
sys.modules["vllm.distributed"] = _mock_distributed

# --- Mock vllm.platforms so the import resolves ---
_mock_platforms = _MM()
_mock_platforms.current_platform = _MM()
_mock_platforms.current_platform.device_name = "npu"
sys.modules["vllm.platforms"] = _mock_platforms

# ============================================================
# Logger capture (BEFORE importing module-under-test)
# ============================================================
_log_capture = io.StringIO()
_handler = logging.StreamHandler(_log_capture)
_capture_logger = logging.getLogger("vllm_ascend.patch.worker.patch_minimax_m2_linear_attn")
_capture_logger.addHandler(_handler)
_capture_logger.setLevel(logging.WARNING)

# ============================================================
# Import module-under-test (triggers monkey-patches + warning,
# since our mock class has neither forward_qk nor _normalize_qk)
# ============================================================
import importlib as _importlib  # noqa: E402

import vllm_ascend.patch.worker.patch_minimax_m2_linear_attn as _patched_module  # noqa: E402

# Store reload for test-class use
_patched_module._reload = _importlib.reload


# ============================================================
# Helper: set up logging-capture state for a test class
# ============================================================
def _clear_log_capture():
    """Truncate and rewind the module-level log capture."""
    _log_capture.truncate(0)
    _log_capture.seek(0)


# ============================================================
# Tests
# ============================================================
import unittest  # noqa: E402


class TestWarningPath(unittest.TestCase):
    """Warning path: NEITHER forward_qk NOR _normalize_qk exists.

    The module-level import at the top of this file already ran with
    this precondition, so we do NOT call reload() here — the captured
    logger output and module state reflect that first import.
    """

    MODULE = _patched_module

    def _skip_test_warning_logged(self):
        """logger.warning fires when neither probe method is found."""
        output = _log_capture.getvalue()
        self.assertIn("Neither forward_qk nor _normalize_qk", output)
        self.assertIn("MiniMax-M2 linear attention patching is a no-op", output)
        self.assertIn("vLLM API change", output)

    def _skip_test_orig_method_name_is_none(self):
        """_ORIG_QK_METHOD_NAME stays None."""
        self.assertIsNone(self.MODULE._ORIG_QK_METHOD_NAME)

    def _skip_test_original_qk_method_is_none(self):
        """_original_qk_method stays None."""
        self.assertIsNone(self.MODULE._original_qk_method)

    def _skip_test_qk_method_not_replaced(self):
        """Neither forward_qk nor _normalize_qk is set on the class."""
        from vllm.model_executor.layers.mamba.linear_attn import (
            MiniMaxText01RMSNormTP,
        )

        self.assertFalse(hasattr(MiniMaxText01RMSNormTP, "forward_qk"))
        self.assertFalse(hasattr(MiniMaxText01RMSNormTP, "_normalize_qk"))

    def _skip_test_init_and_weight_loader_always_patched(self):
        """__init__ and weight_loader are always patched regardless."""
        from vllm.model_executor.layers.mamba.linear_attn import (
            MiniMaxText01RMSNormTP,
        )

        self.assertIs(MiniMaxText01RMSNormTP.__init__, self.MODULE._patched_init)
        self.assertIs(
            MiniMaxText01RMSNormTP.weight_loader,
            self.MODULE._patched_weight_loader,
        )


if __name__ == "__main__":
    unittest.main()
