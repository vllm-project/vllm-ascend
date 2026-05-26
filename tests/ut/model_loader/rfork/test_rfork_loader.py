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
#

"""Unit tests for RForkModelLoader (rfork_loader.py).

Covers the three behavioural changes in this module:
  1. __init__ logs an error before raising RuntimeError when
     model_loader_extra_config is not a dict.
  2. load_model logs "Loading model weights took X.XX seconds" on the
     success path.
  3. load_model logs "RFork transfer failed: ..." and falls back to the
     default loader on failure.
"""

import io
import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

_MM = MagicMock

import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level mocks for device-dependent / missing modules.
# These MUST be in place before any vllm-ascend import.
# ---------------------------------------------------------------------------

# torch_npu -----
_tnpu = _MM()
_tnpu.npu = MagicMock()
_tnpu.npu.current_device.return_value = 0
_tnpu.__spec__ = MagicMock()
sys.modules["torch_npu"] = _tnpu
sys.modules["torch_npu.npu"] = _tnpu.npu
sys.modules["torch_npu._inductor"] = MagicMock()

_tnb = _MM()
_tnb.current_device = MagicMock(return_value=0)
_tnb.is_available = MagicMock(return_value=False)
torch.npu = _tnb

# cbor2, gguf (vllm deps) -----
sys.modules["cbor2"] = MagicMock()
sys.modules["gguf"] = MagicMock()
sys.modules["gguf.constants"] = MagicMock()
sys.modules["gguf.quants"] = MagicMock()

# rfork_worker sub-module (has real device / C extension deps) -----
_rfork_w_mod = MagicMock()
_rfork_w_mod.RForkWorker = MagicMock()
sys.modules["vllm_ascend.model_loader.rfork.rfork_worker"] = _rfork_w_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LogCapture:
    """Context manager that captures vllm logger output into a StringIO buf."""

    def __init__(self):
        self.buf = io.StringIO()
        self._handler = logging.StreamHandler(self.buf)
        self._handler.setLevel(logging.DEBUG)

    def __enter__(self):
        logging.getLogger("vllm.logger").addHandler(self._handler)
        return self.buf

    def __exit__(self, *args):
        logging.getLogger("vllm.logger").removeHandler(self._handler)


class _DummyLoadConfig:
    """Minimal stand-in for vllm.config.LoadConfig."""

    def __init__(self, model_loader_extra_config=None):
        self.model_loader_extra_config = model_loader_extra_config
        self.load_format = "rfork"
        self.device = None
        self.rfork_worker = None


class _DummyVllmConfig:
    """Minimal stand-in for vllm.config.VllmConfig."""

    def __init__(self):
        self.device_config = MagicMock()
        self.device_config.device = "cpu"
        self.kv_transfer_config = None
        self.model_config = MagicMock()
        self.model_config.runner_type = None
        self.scheduler_config = MagicMock()
        self.scheduler_config.runner_type = None
        self.parallel_config = MagicMock()
        self.parallel_config.node_rank = 0


class _DummyModelConfig:
    """Minimal stand-in for vllm.config.ModelConfig."""

    def __init__(self):
        self.dtype = torch.float32


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------


class TestRForkModelLoaderInit(unittest.TestCase):
    """RForkModelLoader.__init__ behaviour for non-dict extra config."""

    def setUp(self):
        # Import the target class once and share across methods.
        from vllm_ascend.model_loader.rfork.rfork_loader import (  # noqa: E402
            RForkModelLoader,
        )

        self._cls = RForkModelLoader

    def test_init_non_dict_config_logs_error_and_raises(self):
        """When model_loader_extra_config is a non-dict, __init__ must log
        an error before raising RuntimeError."""
        config = _DummyLoadConfig(model_loader_extra_config="plain_string")

        with _LogCapture() as buf, self.assertRaises(RuntimeError) as ctx:
            self._cls(config)

        output = buf.getvalue()
        self.assertIn(
            "RFork requires --model-loader-extra-config to be a JSON object.",
            output,
        )
        self.assertEqual(
            str(ctx.exception),
            "RFork requires --model-loader-extra-config to be a JSON object.",
        )

    def test_init_none_config_also_raises(self):
        """A None extra_config also triggers the error path."""
        config = _DummyLoadConfig(model_loader_extra_config=None)

        with _LogCapture() as buf, self.assertRaises(RuntimeError):
            self._cls(config)

        output = buf.getvalue()
        self.assertIn(
            "RFork requires --model-loader-extra-config to be a JSON object.",
            output,
        )


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------


class TestRForkModelLoaderLoadModel(unittest.TestCase):
    """RForkModelLoader.load_model success and fallback paths."""

    def setUp(self):
        self.load_config = _DummyLoadConfig(
            model_loader_extra_config={
                "model_url": "http://test:8000",
                "model_deploy_strategy_name": "deploy_a",
                "rfork_scheduler_url": "http://sched:8001",
                "rfork_seed_timeout_sec": "10.0",
                "rfork_seed_key_separator": "@",
            }
        )
        self.vllm_config = _DummyVllmConfig()
        self.model_config = _DummyModelConfig()

        # Pre-set a mock rfork_worker so _ensure_rfork_worker short-circuits.
        self.mock_worker = MagicMock()
        self.mock_worker.is_seed_available.return_value = True
        self.mock_worker.pre_transfer.return_value = True
        self.mock_worker.transfer.return_value = True
        self.mock_worker.post_transfer.return_value = True
        self.load_config.rfork_worker = self.mock_worker

        # Factory for the no-op context manager that set_default_torch_dtype
        # should return.
        ctx = MagicMock()
        self._torch_dtype_ctx = ctx

    # -- Success path -------------------------------------------------------

    @patch("vllm_ascend.model_loader.rfork.rfork_loader.set_default_torch_dtype")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.initialize_model")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading")
    def test_load_model_success_logs_timing(
        self,
        mock_process,
        mock_init_model,
        mock_set_dtype,
    ):
        """Successful load_model must log 'Loading model weights took ...'."""
        # Don't mock time.time here -- use real time so the log message
        # naturally contains a valid float.  We only verify the message
        # pattern, not the exact value.

        mock_set_dtype.return_value = self._torch_dtype_ctx
        dummy_model = MagicMock()
        dummy_model.eval.return_value = dummy_model
        mock_init_model.return_value = dummy_model

        from vllm_ascend.model_loader.rfork.rfork_loader import (  # noqa: E402
            RForkModelLoader,
        )

        loader = RForkModelLoader(self.load_config)

        with _LogCapture() as buf:
            result = loader.load_model(self.vllm_config, self.model_config)

        # Verify return value and method calls.
        self.assertIs(result, dummy_model)
        dummy_model.eval.assert_called_once()
        self.mock_worker.is_seed_available.assert_called_once()
        self.mock_worker.pre_transfer.assert_called_once()
        self.mock_worker.transfer.assert_called_once()
        self.mock_worker.post_transfer.assert_called_once()
        mock_process.assert_called_once()

        # Verify the timing log message.
        output = buf.getvalue()
        self.assertIn("Loading model weights took", output)
        # A float pattern like "X.XX" should be present.
        self.assertRegex(output, r"took \d+\.\d{2}")

    # -- Fallback paths -----------------------------------------------------

    @patch("vllm_ascend.model_loader.rfork.rfork_loader.set_default_torch_dtype")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.initialize_model")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.gc")
    @patch("vllm.model_executor.model_loader.get_model")
    def test_load_model_fallback_seed_unavailable(
        self,
        mock_get_model,
        mock_gc,
        mock_init_model,
        mock_set_dtype,
    ):
        """Seed unavailable triggers warning + fallback to default loader."""
        mock_set_dtype.return_value = self._torch_dtype_ctx
        self.mock_worker.is_seed_available.return_value = False

        # initialize_model would not have been reached, so need_del stays
        # False and the gc-collect cleanup is skipped.  We still mock it
        # in case the local import path is exercised.
        dummy_model = MagicMock()
        mock_init_model.return_value = dummy_model

        fallback_model = MagicMock()
        mock_get_model.return_value = fallback_model

        from vllm_ascend.model_loader.rfork.rfork_loader import (  # noqa: E402
            RForkModelLoader,
        )

        loader = RForkModelLoader(self.load_config)

        with _LogCapture() as buf:
            result = loader.load_model(self.vllm_config, self.model_config)

        self.assertIs(result, fallback_model)
        # Seed check was called, but the transfer steps were NOT.
        self.mock_worker.is_seed_available.assert_called_once()
        self.mock_worker.pre_transfer.assert_not_called()
        self.mock_worker.transfer.assert_not_called()
        # post_transfer is called as part of cleanup.
        self.mock_worker.post_transfer.assert_called_once()

        # GC was NOT called because need_del is False (exception before
        # initialize_model).
        mock_gc.collect.assert_not_called()

        # Load config was reset.
        self.assertEqual(self.load_config.load_format, "auto")
        self.assertEqual(self.load_config.model_loader_extra_config, {})

        output = buf.getvalue()
        self.assertIn("RFork transfer failed", output)

    @patch("vllm_ascend.model_loader.rfork.rfork_loader.set_default_torch_dtype")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.initialize_model")
    @patch("vllm_ascend.model_loader.rfork.rfork_loader.gc")
    @patch("vllm.model_executor.model_loader.get_model")
    def test_load_model_fallback_transfer_failure(
        self,
        mock_get_model,
        mock_gc,
        mock_init_model,
        mock_set_dtype,
    ):
        """transfer() returning False triggers warning + fallback + cleanup."""
        mock_set_dtype.return_value = self._torch_dtype_ctx
        self.mock_worker.is_seed_available.return_value = True
        self.mock_worker.pre_transfer.return_value = True
        self.mock_worker.transfer.return_value = False  # <-- failure point

        # initialize_model IS reached here, so need_del = True.
        dummy_model = MagicMock()
        mock_init_model.return_value = dummy_model

        fallback_model = MagicMock()
        mock_get_model.return_value = fallback_model

        from vllm_ascend.model_loader.rfork.rfork_loader import (  # noqa: E402
            RForkModelLoader,
        )

        loader = RForkModelLoader(self.load_config)

        with _LogCapture() as buf:
            result = loader.load_model(self.vllm_config, self.model_config)

        self.assertIs(result, fallback_model)
        self.mock_worker.pre_transfer.assert_called_once()
        self.mock_worker.transfer.assert_called_once()
        self.mock_worker.post_transfer.assert_called_once()

        # GC WAS called because need_del = True.
        mock_gc.collect.assert_called()

        self.assertEqual(self.load_config.load_format, "auto")
        self.assertEqual(self.load_config.model_loader_extra_config, {})

        output = buf.getvalue()
        self.assertIn("RFork transfer failed", output)


if __name__ == "__main__":
    unittest.main()
