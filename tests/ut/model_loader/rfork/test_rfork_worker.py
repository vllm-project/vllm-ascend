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

"""Unit tests for RForkWorker (rfork_worker.py).

Covers the five behavioural points in this module:
  1. pre_transfer exception log contains device_id=...
  2. transfer exception log contains device_id=...
  3. start_seed_service logs warning with device_id when pre_transfer fails
  4. start_seed_service: heartbeat thread starts when port > 0;
     BUG: seed_service_started = True set unconditionally (even when port <= 0)
  5. start_seed_service skips when seed_service_started is already True
"""

import io
import logging
import sys
import threading
import unittest
from unittest.mock import MagicMock as _MM
from unittest.mock import patch

import torch

# ---------------------------------------------------------------------------
# Module-level mocks for device-dependent / missing modules.
# These MUST be in place before any vllm-ascend import.
# ---------------------------------------------------------------------------

# torch_npu -----
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

# cbor2, gguf (vllm deps) -----
sys.modules["cbor2"] = _MM()
sys.modules["gguf"] = _MM()
sys.modules["gguf.constants"] = _MM()
sys.modules["gguf.quants"] = _MM()

# yr.datasystem, uvicorn, fastapi, requests (rfork_worker deps) -----
_yr = _MM()
_yr.datasystem = _MM()
sys.modules["yr"] = _yr
sys.modules["yr.datasystem"] = _yr.datasystem
sys.modules["uvicorn"] = _MM()
sys.modules["fastapi"] = _MM()
sys.modules["fastapi.responses"] = _MM()
_req = _MM()
_req.get = _MM()
_req.post = _MM()
sys.modules["requests"] = _req

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRForkWorker(unittest.TestCase):
    """RForkWorker pre_transfer, transfer, and start_seed_service behaviour."""

    DEVICE_ID = 7

    def setUp(self):
        # Patch names imported into rfork_worker's module namespace.
        self._rfork_server_patch = patch("vllm_ascend.model_loader.rfork.rfork_worker.start_rfork_server")
        self._backend_patch = patch("vllm_ascend.model_loader.rfork.rfork_worker.RForkTransferBackend")
        self._protocol_patch = patch("vllm_ascend.model_loader.rfork.rfork_worker.RForkSeedProtocol")

        self.mock_start_rfork_server = self._rfork_server_patch.start()
        self.mock_start_rfork_server.return_value = 9999

        self.mock_backend_cls = self._backend_patch.start()
        self.mock_protocol_cls = self._protocol_patch.start()

        # Configure default backend mock.
        self.mock_backend = _MM()
        self.mock_backend.is_initialized.return_value = True
        self.mock_backend.register_memory_region.return_value = True
        self.mock_backend.recv_from_source.return_value = True
        self.mock_backend_cls.return_value = self.mock_backend

        # Configure default protocol mock.
        self.mock_protocol = _MM()
        self.mock_protocol.get_local_seed_key.return_value = "test_local_key"
        self.mock_protocol_cls.return_value = self.mock_protocol

        # Import the target class (inside test code so module-level mocks
        # are already installed).
        from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker  # noqa: E402

        self.worker = RForkWorker(
            disaggregation_mode="test_mode",
            node_rank=0,
            tp_rank=0,
            device_id=self.DEVICE_ID,
            scheduler_url="http://scheduler:8000",
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_a",
        )

    def tearDown(self):
        self._rfork_server_patch.stop()
        self._backend_patch.stop()
        self._protocol_patch.stop()

    # -- pre_transfer ---------------------------------------------------------

    def test_pre_transfer_success(self):
        """pre_transfer returns True and sets ready_to_start_seed_service."""
        model = _MM()
        result = self.worker.pre_transfer(model)

        self.assertTrue(result)
        self.assertTrue(self.worker.ready_to_start_seed_service)
        self.mock_backend.is_initialized.assert_called_once()
        self.mock_backend.register_memory_region.assert_called_once_with(model)

    def test_pre_transfer_logs_device_id_on_assertion_error(self):
        """pre_transfer logs exception with device_id on AssertionError."""
        self.mock_backend.is_initialized.side_effect = AssertionError("backend not ready")
        model = _MM()

        with _LogCapture() as buf:
            result = self.worker.pre_transfer(model)

        self.assertFalse(result)
        self.assertFalse(self.worker.ready_to_start_seed_service)
        output = buf.getvalue()
        self.assertIn(f"device_id={self.DEVICE_ID}", output)
        self.assertIn("Pre-transfer failed", output)

    # -- transfer -------------------------------------------------------------

    def test_transfer_success(self):
        """transfer calls recv_from_source with seed info and returns True."""
        self.worker.rfork_seed = {
            "seed_ip": "192.168.1.100",
            "seed_port": 4321,
        }
        model = _MM()
        result = self.worker.transfer(model)

        self.assertTrue(result)
        self.mock_backend.recv_from_source.assert_called_once_with(
            model=model,
            seed_instance_ip="192.168.1.100",
            seed_instance_service_port=4321,
            local_seed_key=self.mock_protocol.get_local_seed_key(),
        )

    def test_transfer_logs_device_id_on_assertion_error(self):
        """transfer logs exception with device_id on AssertionError."""
        self.mock_backend.is_initialized.side_effect = AssertionError("backend not ready")
        model = _MM()

        with _LogCapture() as buf:
            result = self.worker.transfer(model)

        self.assertFalse(result)
        output = buf.getvalue()
        self.assertIn(f"device_id={self.DEVICE_ID}", output)
        self.assertIn("Transfer failed", output)

    # -- start_seed_service ---------------------------------------------------

    def test_start_seed_service_skips_when_already_started(self):
        """start_seed_service returns early when seed_service_started is True."""
        self.worker.seed_service_started = True
        model = _MM()

        with _LogCapture() as buf:
            self.worker.start_seed_service(model)

        output = buf.getvalue()
        self.assertIn("already started, skipping", output)
        # pre_transfer (register_memory_region) must not be called.
        self.mock_backend.register_memory_region.assert_not_called()

    def test_start_seed_service_warning_on_pre_transfer_failure(self):
        """start_seed_service logs warning with device_id when pre_transfer
        fails."""
        self.worker.ready_to_start_seed_service = False
        self.mock_backend.register_memory_region.return_value = False
        model = _MM()

        with _LogCapture() as buf:
            self.worker.start_seed_service(model)

        output = buf.getvalue()
        self.assertIn(f"device_id={self.DEVICE_ID}", output)
        self.assertIn("pre_transfer failed", output)
        # start_rfork_server must NOT be called when pre_transfer fails.
        self.mock_start_rfork_server.assert_not_called()

    def test_start_seed_service_starts_heartbeat_when_port_gt_0(self):
        """start_seed_service starts heartbeat thread when port > 0."""
        self.worker.ready_to_start_seed_service = True
        self.worker.seed_service_started = False
        port = 8888
        self.mock_start_rfork_server.return_value = port
        model = _MM()

        self.worker.start_seed_service(model)

        # Heartbeat thread is created, is a daemon, and targets report_seed.
        self.assertIsNotNone(self.worker.rfork_heartbeat_thread)
        self.assertIsInstance(self.worker.rfork_heartbeat_thread, threading.Thread)
        self.assertTrue(self.worker.rfork_heartbeat_thread.daemon)
        self.mock_protocol.report_seed.assert_called_once_with(port)
        self.assertTrue(self.worker.seed_service_started)

    def test_start_seed_service_sets_started_even_when_port_le_0(self):
        """BUG: seed_service_started is set unconditionally even when port <= 0,
        and no heartbeat thread is started."""
        self.worker.ready_to_start_seed_service = True
        self.worker.seed_service_started = False
        self.mock_start_rfork_server.return_value = 0  # trigger the bug
        model = _MM()

        self.worker.start_seed_service(model)

        # Bug: seed_service_started is True even though port <= 0.
        self.assertTrue(self.worker.seed_service_started)
        # No heartbeat thread should exist.
        self.assertFalse(hasattr(self.worker, "rfork_heartbeat_thread"))
        self.mock_protocol.report_seed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
