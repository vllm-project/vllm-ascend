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

"""Unit tests for seed_server.py.

Covers the timeout path in start_rfork_server():
  1. When port_queue.get times out, the function logs
     "[RFork Seed] start server error for seed_key=%s: ..." and returns -1.
"""

import io
import logging
import queue
import sys
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

# uvicorn + fastapi (needed to import seed_server) -----
sys.modules["uvicorn"] = _MM()
sys.modules["uvicorn.Config"] = _MM()
sys.modules["uvicorn.Server"] = _MM()

_fa = _MM()
_fa.FastAPI = _MM()

sys.modules["fastapi"] = _fa
sys.modules["fastapi.responses"] = _fa.responses

# requests (seed_server dep) -----
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


class TestStartRforkServer(unittest.TestCase):
    """Tests for seed_server.start_rfork_server."""

    def test_timeout_returns_minus_one_and_logs_seed_key(self):
        """start_rfork_server returns -1 and logs seed_key when
        port_queue.get raises queue.Empty."""
        seed_key = "http://model:8000$deploy_x$mode_a$0$0"

        with patch("queue.Queue.get", side_effect=queue.Empty("timeout")):
            from vllm_ascend.model_loader.rfork.seed_server import (  # noqa: E402
                start_rfork_server,
            )

            with _LogCapture() as buf:
                port = start_rfork_server(
                    local_seed_key=seed_key,
                    rfork_transfer_engine_info={},
                    health_timeout_sec=30.0,
                )

        self.assertEqual(port, -1)
        output = buf.getvalue()
        self.assertIn(seed_key, output)
        self.assertIn("start server error", output)

    def test_timeout_logs_queue_exception_message(self):
        """start_rfork_server log contains the exception str when
        port_queue.get times out."""
        seed_key = "my_test_seed"

        with patch("queue.Queue.get", side_effect=queue.Empty("timeout")):
            from vllm_ascend.model_loader.rfork.seed_server import (  # noqa: E402
                start_rfork_server,
            )

            with _LogCapture() as buf:
                port = start_rfork_server(
                    local_seed_key=seed_key,
                    rfork_transfer_engine_info={},
                    health_timeout_sec=30.0,
                )

        self.assertEqual(port, -1)
        output = buf.getvalue()
        self.assertIn("timeout", output)


if __name__ == "__main__":
    unittest.main()
